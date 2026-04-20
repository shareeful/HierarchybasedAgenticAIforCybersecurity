import argparse
import json
import sys
import random
from pathlib import Path

from data import AssetRecord, VulnerabilityRecord
from pipeline import HierVulExPipeline, OutcomeSignals, collect_and_prepare_data


def build_example_assets():
    return [
        AssetRecord(
            asset_id="web-server-01", zone="DMZ", criticality="critical",
            sw_versions={"apache": "2.4.51", "openssl": "1.1.1t", "php": "8.0.20"},
            controls=["waf", "ids"],
            patch_history=[{"patch_id": "P001", "caused_disruption": False, "date": "2023-06-01"},
                           {"patch_id": "P002", "caused_disruption": True, "date": "2023-09-15"}]
        ),
        AssetRecord(
            asset_id="db-server-01", zone="internal", criticality="high",
            sw_versions={"mysql": "8.0.32", "linux-kernel": "5.15.0"},
            controls=["firewall", "ids", "dlp"],
            patch_history=[{"patch_id": "P003", "caused_disruption": False, "date": "2023-07-01"}]
        ),
        AssetRecord(
            asset_id="app-server-02", zone="cloud", criticality="medium",
            sw_versions={"nginx": "1.24.0", "nodejs": "18.17.0", "openssl": "3.0.9"},
            controls=["firewall"],
            patch_history=[{"patch_id": "P005", "caused_disruption": True, "date": "2023-08-10"}]
        )
    ]


def build_example_vulnerabilities():
    return [
        VulnerabilityRecord(
            cve_id="CVE-2024-1234", cvss=9.8, epss=0.72, poc_available=1,
            attack_technique="T1190", cwe="CWE-89",
            affected_software=["apache", "openssl"],
            description="Critical SQL injection in Apache web server.", label=1
        ),
        VulnerabilityRecord(
            cve_id="CVE-2024-5678", cvss=7.5, epss=0.18, poc_available=0,
            attack_technique="T1059", cwe="CWE-22",
            affected_software=["nodejs", "nginx"],
            description="Path traversal in Node.js application.", label=0
        ),
        VulnerabilityRecord(
            cve_id="CVE-2024-9999", cvss=5.3, epss=0.04, poc_available=0,
            attack_technique="T1078", cwe="CWE-287",
            affected_software=["mysql"],
            description="Authentication bypass in MySQL under specific configurations.", label=0
        )
    ]


def run(args):
    print("\n=== HierVulEx: Hierarchical Multi-AI-Agent Cybersecurity Risk Assessment ===\n")

    pipeline = HierVulExPipeline(
        use_local_models=args.use_models,
        va_model=args.va_model,
        ca_model=args.ca_model,
        sa_model=args.sa_model
    )

    if args.load_state and Path(args.load_state).exists():
        pipeline.load_state(args.load_state)
        print(f"Loaded state from {args.load_state}")

    assets = build_example_assets()

    if args.fetch_data:
        vulnerabilities, _, kev_ids = collect_and_prepare_data(
            start_date=args.start_date, end_date=args.end_date, assets=assets
        )
        if not vulnerabilities:
            print("No vulnerabilities fetched.")
            sys.exit(1)
    else:
        vulnerabilities = build_example_vulnerabilities()

    print(f"Processing {len(vulnerabilities)} vulnerabilities against {len(assets)} assets...\n")
    results = pipeline.run_batch(vulnerabilities, assets)

    print("=== RISK ASSESSMENT RESULTS ===\n")
    for i, r in enumerate(results):
        print(f"[{i+1}] {r.get('cve_id', 'N/A')}")
        if r.get("status") == "ESCALATED":
            print("     STATUS: ESCALATED TO HUMAN REVIEW")
        else:
            print(f"     Joint Risk:    {r.get('joint_risk', 0):.4f}")
            print(f"     Adjusted Risk: {r.get('adjusted_risk', 0):.4f}")
            print(f"     Action:        {r.get('action', '').upper()}")
            print(f"     Control:       {r.get('control_type', '').replace('_', ' ')}")
            print(f"     VA Flagged:    {r.get('va_flagged', False)}")
            print(f"     CA Flagged:    {r.get('ca_flagged', False)}")
            print(f"     [ANALYST] {r.get('technical_explanation', '')[:120]}...")
            print(f"     [MANAGER] {r.get('plain_explanation', '')}")
        print()

    if args.simulate_outcomes:
        print("=== SIMULATING OUTCOMES AND UPDATING POLICIES ===\n")
        random.seed(42)
        for r in results:
            if r.get("status") == "DECIDED":
                outcomes = OutcomeSignals(
                    s_p=random.randint(0, 1), s_e=random.randint(0, 1),
                    s_r=round(random.uniform(0.1, 0.8), 3), s_d=random.randint(0, 1)
                )
                reward = pipeline.observe_outcome_and_update(r, outcomes)
                print(f"  {r.get('cve_id')}: reward={reward:.4f} "
                      f"[s_p={outcomes.s_p} s_e={outcomes.s_e} s_r={outcomes.s_r:.3f} s_d={outcomes.s_d}]")

    if args.save_state:
        pipeline.save_state(args.save_state)
        print(f"\nState saved to {args.save_state}")

    if args.save_audit:
        pipeline.audit.save(args.save_audit)
        print(f"Audit trail saved to {args.save_audit}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {args.output}")

    print("\n=== DONE ===")


def main():
    parser = argparse.ArgumentParser(description="HierVulEx Risk Assessment Pipeline")
    parser.add_argument("--fetch-data", action="store_true")
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default="2024-12-31")
    parser.add_argument("--use-models", action="store_true")
    parser.add_argument("--va-model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--ca-model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--sa-model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--simulate-outcomes", action="store_true")
    parser.add_argument("--save-state", default=None)
    parser.add_argument("--load-state", default=None)
    parser.add_argument("--save-audit", default=None)
    parser.add_argument("--output", default=None)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
