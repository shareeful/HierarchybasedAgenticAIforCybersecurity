import requests
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta


@dataclass
class VulnerabilityRecord:
    cve_id: str
    cvss: float
    epss: float
    poc_available: int
    attack_technique: str
    cwe: str
    affected_software: List[str]
    description: str = ""
    label: int = 0


@dataclass
class AssetRecord:
    asset_id: str
    zone: str
    criticality: str
    sw_versions: Dict[str, str]
    controls: List[str]
    patch_history: List[Dict]


@dataclass
class ExposurePair:
    vulnerability: VulnerabilityRecord
    asset: AssetRecord


def fetch_nvd_cves(start_date: str, end_date: str, results_per_page: int = 100) -> List[Dict]:
    base_url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
    params = {
        "pubStartDate": f"{start_date}T00:00:00.000",
        "pubEndDate": f"{end_date}T23:59:59.999",
        "resultsPerPage": results_per_page,
        "startIndex": 0
    }
    all_cves = []
    try:
        response = requests.get(base_url, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            all_cves.extend(data.get("vulnerabilities", []))
            total = data.get("totalResults", 0)
            while len(all_cves) < min(total, 5000):
                params["startIndex"] = len(all_cves)
                response = requests.get(base_url, params=params, timeout=30)
                if response.status_code != 200:
                    break
                batch = response.json().get("vulnerabilities", [])
                if not batch:
                    break
                all_cves.extend(batch)
    except Exception:
        pass
    return all_cves


def fetch_epss_scores(cve_ids: List[str]) -> Dict[str, float]:
    scores = {}
    batch_size = 100
    for i in range(0, len(cve_ids), batch_size):
        batch = cve_ids[i:i + batch_size]
        params = {"cve": ",".join(batch)}
        try:
            response = requests.get("https://api.first.org/data/v1/epss", params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                for item in data.get("data", []):
                    scores[item["cve"]] = float(item.get("epss", 0.0))
        except Exception:
            pass
    return scores


def fetch_cisa_kev() -> List[str]:
    exploited_cves = []
    try:
        response = requests.get(
            "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json",
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            exploited_cves = [v["cveID"] for v in data.get("vulnerabilities", [])]
    except Exception:
        pass
    return exploited_cves


def parse_nvd_record(raw: Dict) -> Optional[VulnerabilityRecord]:
    try:
        cve = raw.get("cve", {})
        cve_id = cve.get("id", "")
        metrics = cve.get("metrics", {})
        cvss = 0.0
        for key in ["cvssMetricV31", "cvssMetricV30", "cvssMetricV2"]:
            if key in metrics and metrics[key]:
                cvss = metrics[key][0].get("cvssData", {}).get("baseScore", 0.0)
                break
        weaknesses = cve.get("weaknesses", [])
        cwe = ""
        if weaknesses:
            descs = weaknesses[0].get("description", [])
            if descs:
                cwe = descs[0].get("value", "")
        configs = cve.get("configurations", [])
        affected_sw = []
        for config in configs:
            for node in config.get("nodes", []):
                for match in node.get("cpeMatch", []):
                    cpe = match.get("criteria", "")
                    parts = cpe.split(":")
                    if len(parts) > 4:
                        affected_sw.append(f"{parts[3]}:{parts[4]}")
        descriptions = cve.get("descriptions", [])
        desc = ""
        for d in descriptions:
            if d.get("lang") == "en":
                desc = d.get("value", "")
                break
        return VulnerabilityRecord(
            cve_id=cve_id,
            cvss=float(cvss),
            epss=0.0,
            poc_available=0,
            attack_technique="",
            cwe=cwe,
            affected_software=list(set(affected_sw)),
            description=desc
        )
    except Exception:
        return None


def build_vulnerability_feature_vector(record: VulnerabilityRecord) -> Dict:
    return {
        "cve_id": record.cve_id,
        "cvss": record.cvss,
        "epss": record.epss,
        "poc_available": record.poc_available,
        "attack_technique": record.attack_technique,
        "cwe": record.cwe,
        "affected_software": record.affected_software,
        "description": record.description,
        "label": record.label
    }


def build_asset_feature_vector(asset: AssetRecord) -> Dict:
    disruption_count = sum(1 for p in asset.patch_history if p.get("caused_disruption", False))
    total_patches = len(asset.patch_history)
    disruption_rate = disruption_count / max(total_patches, 1)
    zone_map = {"DMZ": 1.0, "cloud": 0.8, "internal": 0.4, "isolated": 0.1}
    crit_map = {"critical": 1.0, "high": 0.75, "medium": 0.5, "low": 0.25}
    return {
        "asset_id": asset.asset_id,
        "zone": asset.zone,
        "zone_score": zone_map.get(asset.zone, 0.5),
        "criticality": asset.criticality,
        "criticality_score": crit_map.get(asset.criticality, 0.5),
        "sw_versions": asset.sw_versions,
        "controls": asset.controls,
        "num_controls": len(asset.controls),
        "disruption_rate": disruption_rate,
        "total_patches": total_patches
    }


def generate_exposure_pairs(
    vulnerabilities: List[VulnerabilityRecord],
    assets: List[AssetRecord]
) -> List[ExposurePair]:
    pairs = []
    for vuln in vulnerabilities:
        for asset in assets:
            installed = set(asset.sw_versions.keys())
            affected = set(vuln.affected_software)
            if installed & affected:
                pairs.append(ExposurePair(vulnerability=vuln, asset=asset))
    return pairs


def validate_records(
    vulnerabilities: List[VulnerabilityRecord],
    assets: List[AssetRecord]
) -> Tuple[List[VulnerabilityRecord], List[AssetRecord]]:
    valid_vulns = [v for v in vulnerabilities if v.cve_id and v.cvss >= 0]
    valid_assets = [a for a in assets if a.asset_id and a.zone and a.criticality]
    if valid_vulns:
        epss_values = np.array([v.epss for v in valid_vulns])
        cvss_values = np.array([v.cvss for v in valid_vulns])
        epss_mean, epss_std = epss_values.mean(), epss_values.std() + 1e-9
        cvss_mean, cvss_std = cvss_values.mean(), cvss_values.std() + 1e-9
        filtered = []
        for v in valid_vulns:
            z_epss = abs((v.epss - epss_mean) / epss_std)
            z_cvss = abs((v.cvss - cvss_mean) / cvss_std)
            if z_epss <= 3 and z_cvss <= 3:
                filtered.append(v)
        valid_vulns = filtered
    return valid_vulns, valid_assets


def load_dataset_from_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def save_dataset(data: List[Dict], path: str):
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    return df


def prepare_training_data(
    vulnerabilities: List[VulnerabilityRecord],
    assets: List[AssetRecord],
    kev_ids: List[str]
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    for v in vulnerabilities:
        v.label = 1 if v.cve_id in kev_ids else 0

    va_data = []
    for v in vulnerabilities:
        risk_score = 1.0 if v.label == 1 else min(v.epss * 1.5 + v.cvss / 20.0, 1.0)
        va_data.append({
            "input": build_vulnerability_feature_vector(v),
            "risk_score": round(risk_score, 4),
            "attack_technique": v.attack_technique,
            "label": v.label
        })

    pairs = generate_exposure_pairs(vulnerabilities, assets)
    ca_data = []
    for pair in pairs:
        vuln_feat = build_vulnerability_feature_vector(pair.vulnerability)
        asset_feat = build_asset_feature_vector(pair.asset)
        exposure = (
            asset_feat["zone_score"] * 0.45 +
            asset_feat["criticality_score"] * 0.35 +
            (1.0 - min(asset_feat["num_controls"] / 5.0, 1.0)) * 0.20
        )
        regression_risk = asset_feat["disruption_rate"]
        ca_data.append({
            "vulnerability": vuln_feat,
            "asset": asset_feat,
            "exposure_score": round(exposure, 4),
            "regression_risk": round(regression_risk, 4)
        })

    supervisor_data = []
    for i in range(min(len(va_data), len(ca_data))):
        risk = va_data[i]["risk_score"]
        exp = ca_data[i]["exposure_score"]
        reg = ca_data[i]["regression_risk"]
        adj = (risk * 0.6 + exp * 0.4) * (1 - 0.4 * reg)
        if adj > 0.75:
            action = "patch"
        elif adj > 0.5:
            action = "compensate"
        elif adj > 0.3:
            action = "defer"
        else:
            action = "accept"
        supervisor_data.append({
            "va_output": va_data[i],
            "ca_output": ca_data[i],
            "action": action,
            "control_type": _map_cwe_to_control(va_data[i]["input"].get("cwe", ""))
        })

    return va_data, ca_data, supervisor_data


def _map_cwe_to_control(cwe: str) -> str:
    mapping = {
        "CWE-89": "input_validation",
        "CWE-79": "input_validation",
        "CWE-22": "access_control",
        "CWE-287": "access_control",
        "CWE-798": "access_control",
        "CWE-200": "access_control",
        "CWE-502": "patch_management",
        "CWE-78": "patch_management",
        "CWE-125": "patch_management",
        "CWE-119": "patch_management",
    }
    for key, val in mapping.items():
        if key in cwe:
            return val
    return "network_segmentation"
