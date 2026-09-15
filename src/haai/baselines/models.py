from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..agents.llm import CallLog, LanguageModel, ModelUnavailable, resolve_model
from ..agents.parsing import parse_completion
from ..agents.schemas import VULNERABILITY_SCHEMA, ParseTelemetry
from ..config import Config
from ..logging_utils import get_logger

LOGGER = get_logger()

STRUCTURED_FIELDS: Tuple[str, ...] = (
    "cvss",
    "epss",
    "poc_available",
    "cwe_risk_prior",
    "attack_technique_prior",
    "affected_software_breadth",
)


def structured_matrix(frame: pd.DataFrame) -> np.ndarray:
    columns = []
    for name in STRUCTURED_FIELDS:
        values = frame[name]
        if values.dtype == bool:
            columns.append(values.to_numpy(dtype=float))
        else:
            columns.append(pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype=float))
    return np.column_stack(columns)


class CVSSOnlyBaseline:
    name = "cvss_only"

    def fit(self, train: pd.DataFrame, config: Config, rng: np.random.Generator):
        return self

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        return frame["cvss"].to_numpy(dtype=float) / 10.0


class CalibratedCVSSEPSSBaseline:
    name = "cvss_epss_calibrated"

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=1_000)

    def _matrix(self, frame: pd.DataFrame) -> np.ndarray:
        return np.column_stack(
            [frame["cvss"].to_numpy(dtype=float), frame["epss"].to_numpy(dtype=float)]
        )

    def fit(self, train: pd.DataFrame, config: Config, rng: np.random.Generator):
        matrix = self.scaler.fit_transform(self._matrix(train))
        self.model.fit(matrix, train["label"].to_numpy(dtype=int))
        return self

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.scaler.transform(self._matrix(frame)))[:, 1]


class XGBoostBaseline:
    name = "xgboost"

    def __init__(self):
        self.model = None

    def fit(self, train: pd.DataFrame, config: Config, rng: np.random.Generator):
        matrix = structured_matrix(train)
        labels = train["label"].to_numpy(dtype=int)
        try:
            from xgboost import XGBClassifier

            positive = max(int((labels == 1).sum()), 1)
            negative = max(int((labels == 0).sum()), 1)
            self.model = XGBClassifier(
                n_estimators=config.baseline.xgboost_estimators,
                max_depth=config.baseline.xgboost_max_depth,
                learning_rate=config.baseline.xgboost_learning_rate,
                subsample=config.baseline.xgboost_subsample,
                scale_pos_weight=negative / positive,
                eval_metric="logloss",
                tree_method="hist",
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingClassifier

            LOGGER.warning("xgboost is not installed; using HistGradientBoostingClassifier")
            self.model = HistGradientBoostingClassifier(
                max_depth=config.baseline.xgboost_max_depth,
                learning_rate=config.baseline.xgboost_learning_rate,
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
        self.model.fit(matrix, labels)
        return self

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(structured_matrix(frame))[:, 1]


class SecureBERTBaseline:
    name = "securebert_cve"

    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = "cpu"

    def _require(self):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as error:
            raise ModelUnavailable(
                "the SecureBERT-CVE baseline fine-tunes a transformer encoder; install "
                "requirements-llm.txt"
            ) from error
        return torch, AutoModelForSequenceClassification, AutoTokenizer

    def fit(self, train: pd.DataFrame, config: Config, rng: np.random.Generator):
        torch, AutoModelForSequenceClassification, AutoTokenizer = self._require()
        reference = resolve_model(config.baseline.securebert_model)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(reference)
            self.model = AutoModelForSequenceClassification.from_pretrained(reference, num_labels=2)
        except Exception as error:
            raise ModelUnavailable(
                f"could not load '{config.baseline.securebert_model}'; place the weights under "
                "$HAAI_MODEL_DIR or authenticate with the model host"
            ) from error
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.train()

        texts = train["description"].astype(str).tolist()
        labels = train["label"].to_numpy(dtype=int)
        counts = np.bincount(labels, minlength=2).astype(float)
        weights = torch.tensor(
            (counts.sum() / np.maximum(counts, 1.0)), dtype=torch.float32, device=self.device
        )
        loss_function = torch.nn.CrossEntropyLoss(weight=weights)
        optimiser = torch.optim.AdamW(
            self.model.parameters(), lr=config.baseline.securebert_learning_rate
        )
        order = np.arange(len(texts))
        size = config.baseline.securebert_batch_size
        for epoch in range(config.baseline.securebert_epochs):
            rng.shuffle(order)
            total = 0.0
            for start in range(0, len(order), size):
                index = order[start : start + size]
                encoded = self.tokenizer(
                    [texts[i] for i in index],
                    truncation=True,
                    padding=True,
                    max_length=config.baseline.securebert_max_length,
                    return_tensors="pt",
                ).to(self.device)
                target = torch.tensor(labels[index], device=self.device)
                optimiser.zero_grad()
                logits = self.model(**encoded).logits
                loss = loss_function(logits, target)
                loss.backward()
                optimiser.step()
                total += float(loss.item())
            LOGGER.info(
                "SecureBERT epoch %d/%d mean loss %.4f",
                epoch + 1,
                config.baseline.securebert_epochs,
                total / max(len(order) / size, 1),
            )
        self.model.eval()
        return self

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        import torch

        texts = frame["description"].astype(str).tolist()
        size = self.config.baseline.securebert_batch_size
        probabilities: List[float] = []
        with torch.no_grad():
            for start in range(0, len(texts), size):
                encoded = self.tokenizer(
                    texts[start : start + size],
                    truncation=True,
                    padding=True,
                    max_length=self.config.baseline.securebert_max_length,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**encoded).logits
                probabilities.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy().tolist())
        return np.asarray(probabilities, dtype=float)


class TextEncoder:
    def __init__(self, config: Config):
        self.config = config
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise ModelUnavailable(
                "the RAG-LLM baseline needs a transformer encoder; install requirements-llm.txt"
            ) from error
        reference = resolve_model(config.baseline.retrieval_encoder)
        self.torch = torch
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(reference)
            self.model = AutoModel.from_pretrained(reference)
        except Exception as error:
            raise ModelUnavailable(
                f"could not load the retrieval encoder '{config.baseline.retrieval_encoder}'"
            ) from error
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        torch = self.torch
        vectors: List[np.ndarray] = []
        size = self.config.baseline.retrieval_batch_size
        with torch.no_grad():
            for start in range(0, len(texts), size):
                encoded = self.tokenizer(
                    list(texts[start : start + size]),
                    truncation=True,
                    padding=True,
                    max_length=self.config.baseline.retrieval_max_length,
                    return_tensors="pt",
                ).to(self.device)
                hidden = self.model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                pooled = torch.nn.functional.normalize(pooled, dim=-1)
                vectors.append(pooled.cpu().numpy())
        return np.vstack(vectors)


class RAGLLMBaseline:
    name = "rag_llm"

    def __init__(self, config: Config, model: LanguageModel, telemetry: ParseTelemetry, log: CallLog):
        self.config = config
        self.model = model
        self.telemetry = telemetry
        self.log = log
        self.encoder: Optional[TextEncoder] = None
        self.index: Optional[np.ndarray] = None
        self.corpus: Optional[pd.DataFrame] = None

    def fit(self, train: pd.DataFrame, config: Config, rng: np.random.Generator):
        self.encoder = TextEncoder(config)
        self.corpus = train.reset_index(drop=True)
        self.index = self.encoder.encode(self.corpus["description"].astype(str).tolist())
        LOGGER.info("RAG index built over %d retrieval documents", len(self.corpus))
        return self

    def _context(self, neighbours: np.ndarray) -> str:
        lines = []
        for position in neighbours:
            record = self.corpus.iloc[int(position)]
            lines.append(
                f"- {record['cve_id']} (CVSS {float(record['cvss']):.1f}, EPSS "
                f"{float(record['epss']):.4f}, {record['cwe']}): "
                f"{'confirmed exploited' if int(record['label']) == 1 else 'no confirmed exploitation'}. "
                f"{str(record['description'])[:220]}"
            )
        return "\n".join(lines)

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        if self.encoder is None or self.index is None:
            raise RuntimeError("the RAG baseline must be fitted before scoring")
        queries = self.encoder.encode(frame["description"].astype(str).tolist())
        similarity = queries @ self.index.T
        k = self.config.fidelity.rag_neighbours
        ranked = np.argsort(-similarity, axis=1)[:, :k]
        prompts: List[str] = []
        for position, record in enumerate(frame.to_dict("records")):
            prompts.append(
                "SYSTEM: You are a vulnerability analyst. Use the retrieved historical records to "
                "judge whether the target vulnerability will be exploited in the wild.\n\n"
                f"RETRIEVED RECORDS:\n{self._context(ranked[position])}\n\n"
                f"TARGET:\n{record['cve_id']} (CVSS {float(record['cvss']):.1f}, EPSS "
                f"{float(record['epss']):.4f}, {record['cwe']})\n"
                f"{str(record['description'])[:400]}\n\n"
                'OUTPUT FORMAT (JSON):\n{\n  "risk_score" : float [0,1],\n'
                '  "confidence"  : float [0,1],\n  "reasoning"   : string\n}\n'
            )
        scores: List[float] = []
        size = self.config.runtime.inference_batch_size
        for start in range(0, len(prompts), size):
            completions = self.model.generate(
                prompts[start : start + size], agent=self.name, log=self.log
            )
            for completion in completions:
                parsed = parse_completion(
                    completion.text,
                    VULNERABILITY_SCHEMA,
                    self.telemetry,
                    score_field="risk_score",
                    max_retries=0,
                )
                scores.append(float(parsed.values["risk_score"]))
        return np.asarray(scores, dtype=float)


class SingleAgentBaseline:
    name = "single_agent"

    def __init__(self, config: Config, model: LanguageModel, telemetry: ParseTelemetry, log: CallLog):
        self.config = config
        self.model = model
        self.telemetry = telemetry
        self.log = log

    def fit(self, train: pd.DataFrame, config: Config, rng: np.random.Generator):
        return self

    def score(self, pairs: pd.DataFrame) -> np.ndarray:
        prompts: List[str] = []
        for record in pairs.to_dict("records"):
            prompts.append(
                "SYSTEM: You are a single security analyst responsible for both threat "
                "intelligence and organisational context. Produce one combined risk score.\n\n"
                "INPUT:\n"
                f"CVE-ID           : {record.get('cve_id', '')}\n"
                f"CVSS             : {float(record.get('cvss', 0.0)):.1f}\n"
                f"EPSS             : {float(record.get('epss', 0.0)):.4f}\n"
                f"PoC Available    : {'yes' if bool(record.get('poc_available', False)) else 'no'}\n"
                f"CWE              : {record.get('cwe', '')}\n"
                f"ATT&CK           : {record.get('attack_technique', '') or 'none mapped'}\n"
                f"Asset ID         : {record.get('asset_id', '')}\n"
                f"Zone             : {record.get('zone', '')}\n"
                f"Criticality      : {float(record.get('criticality', 0.0)):.2f}\n"
                f"Reachable        : {'yes' if bool(record.get('external_reachable', False)) else 'no'}\n"
                f"Control coverage : {float(record.get('control_coverage', 0.0)):.2f}\n\n"
                'OUTPUT FORMAT (JSON):\n{\n  "risk_score" : float [0,1],\n'
                '  "confidence"  : float [0,1],\n  "reasoning"   : string\n}\n'
            )
        scores: List[float] = []
        size = self.config.runtime.inference_batch_size
        for start in range(0, len(prompts), size):
            completions = self.model.generate(
                prompts[start : start + size], agent=self.name, log=self.log
            )
            for completion in completions:
                parsed = parse_completion(
                    completion.text,
                    VULNERABILITY_SCHEMA,
                    self.telemetry,
                    score_field="risk_score",
                    max_retries=0,
                )
                scores.append(float(parsed.values["risk_score"]))
        return np.asarray(scores, dtype=float)


class FlatMultiAgentBaseline:
    name = "flat_multi_agent"

    def score(self, risk: np.ndarray, exposure: np.ndarray) -> np.ndarray:
        return 0.5 * np.asarray(risk, dtype=float) + 0.5 * np.asarray(exposure, dtype=float)
