from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from ..config import Config
from ..logging_utils import get_logger
from .llm import CallLog, Completion, LanguageModel
from .parsing import ParsedOutput, parse_completion
from .prompts import CONTEXTUAL_FIELD_LABELS, RenderedPrompt, render_contextual_prompt
from .schemas import CONTEXTUAL_SCHEMA, ParseTelemetry
from .vulnerability_agent import AgentOutput, field_attention

LOGGER = get_logger()

ATTENTION_FIELDS: Tuple[str, ...] = tuple(name for name, _ in CONTEXTUAL_FIELD_LABELS)


def render_rows(pairs: pd.DataFrame, risk_scores: np.ndarray) -> List[RenderedPrompt]:
    rendered: List[RenderedPrompt] = []
    records = pairs.to_dict("records")
    for record, risk in zip(records, np.asarray(risk_scores, dtype=float)):
        values = {
            "cve_id": record.get("cve_id", ""),
            "risk_score_input": f"{float(risk):.3f}",
            "asset_id": record.get("asset_id", ""),
            "zone": record.get("zone", ""),
            "adjacent_zones": str(record.get("adjacency_degree", 0)),
            "external_reachable": "yes" if bool(record.get("external_reachable", False)) else "no",
            "criticality": f"{float(record.get('criticality', 0.0)):.2f}",
            "installed_software": record.get("matched_product", ""),
            "controls": f"{float(record.get('control_coverage', 0.0)):.2f} coverage",
            "patch_history": (
                f"disruption {float(record.get('patch_disruption_history', 0.0)):.2f}, "
                f"recency {float(record.get('patch_recency', 0.0)):.2f}"
            ),
        }
        rendered.append(render_contextual_prompt(values))
    return rendered


class ContextualAgent:
    name = "contextual_agent"

    def __init__(self, config: Config, model: LanguageModel, telemetry: ParseTelemetry, log: CallLog):
        self.config = config
        self.model = model
        self.telemetry = telemetry
        self.log = log

    def assess(
        self,
        pairs: pd.DataFrame,
        risk_scores: np.ndarray,
        need_attention: bool = False,
        batch_size: int | None = None,
    ) -> AgentOutput:
        prompts = render_rows(pairs, risk_scores)
        size = batch_size or self.config.runtime.inference_batch_size
        exposure: List[float] = []
        regression: List[float] = []
        confidences: List[float] = []
        logprobs: List[float] = []
        flags: List[bool] = []
        parsed_all: List[ParsedOutput] = []
        completions_all: List[Completion] = []
        attention_rows: List[np.ndarray] = []

        for start in range(0, len(prompts), size):
            chunk = prompts[start : start + size]
            completions = self.model.generate(
                [item.text for item in chunk],
                agent=self.name,
                log=self.log,
                need_attention=need_attention,
            )
            for prompt, completion in zip(chunk, completions):
                def retry(prompt_text=prompt.text) -> str:
                    again = self.model.generate(
                        [prompt_text + "\n\nRespond with the JSON object only."],
                        agent=self.name,
                        log=self.log,
                    )
                    return again[0].text

                parsed = parse_completion(
                    completion.text,
                    CONTEXTUAL_SCHEMA,
                    self.telemetry,
                    retry=retry,
                    score_field="exposure_score",
                    max_retries=self.config.fidelity.repair_retry_attempts,
                )
                exposure.append(float(parsed.values["exposure_score"]))
                regression.append(float(parsed.values["regression_risk"]))
                confidences.append(float(parsed.values["confidence"]))
                logprobs.append(completion.field_logprob("exposure_score"))
                flags.append(parsed.flagged())
                parsed_all.append(parsed)
                completions_all.append(completion)
                if need_attention:
                    attention_rows.append(field_attention(prompt, completion, ATTENTION_FIELDS))
            LOGGER.info(
                "%s scored %d/%d pairs", self.name, min(start + size, len(prompts)), len(prompts)
            )

        return AgentOutput(
            score=np.asarray(exposure, dtype=float),
            secondary=np.asarray(regression, dtype=float),
            reported_confidence=np.asarray(confidences, dtype=float),
            sequence_logprob=np.asarray(logprobs, dtype=float),
            flags=np.asarray(flags, dtype=bool),
            parsed=parsed_all,
            prompts=prompts,
            completions=completions_all,
            attention=np.vstack(attention_rows) if attention_rows else None,
            fields=ATTENTION_FIELDS,
        )
