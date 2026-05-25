"""
Main pipeline orchestrator for the substitution LM.

Runs steps 1–5 in order, skipping any step whose output already exists.
Delete the relevant cache / data file to force a step to re-run.

Usage:
    python run_pipeline.py [--steps 1,2,3,4,5] [--eval] [--mechanism a|b|c|all]

Steps:
    1  Parse corpus with SpaCy → cache/parse_train.jsonl + cache/parse_test.jsonl
    2  Extract n-gram nodes    → data/node_table.pkl
    3  Build co-occurrence DAG → data/dag.pkl
    4  ConceptNet properties   → data/properties.pkl
    5  Assemble model          → data/model.pkl

--eval : run PPL/BPB/SVDR evaluation after build (requires step 5 complete)
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import config as C

C.DATA_DIR.mkdir(parents=True, exist_ok=True)
C.CACHE_DIR.mkdir(parents=True, exist_ok=True)

STEP_SCRIPTS = {
    1: ROOT / "pipeline" / "01_parse.py",
    2: ROOT / "pipeline" / "02_extract_ngrams.py",
    3: ROOT / "pipeline" / "03_build_dag.py",
    4: ROOT / "pipeline" / "04_property_lookup.py",
    5: ROOT / "pipeline" / "05_build_model.py",
}

STEP_OUTPUTS = {
    1: C.PARSE_CACHE_TRAIN,
    2: C.NODE_TABLE_PATH,
    3: C.DAG_PATH,
    4: C.PROPERTIES_PATH,
    5: C.MODEL_PATH,
}



def run_step(n: int):
    script = STEP_SCRIPTS[n]
    output = STEP_OUTPUTS[n]

    if output.exists():
        print(f"  Step {n}: output exists ({output.name}) — skipping")
        return

    t0 = time.time()
    result = subprocess.run([sys.executable, str(script)], check=False)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  Step {n} FAILED (exit {result.returncode}) — aborting pipeline.")
        sys.exit(result.returncode)

    print(f"  Step {n} complete ({elapsed:.0f}s)\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", default="1,2,3,4,5",
                        help="Comma-separated list of steps to run")
    parser.add_argument("--eval", action="store_true",
                        help="Run PPL/BPB/SVDR after build")
    parser.add_argument("--mechanism", default="all",
                        choices=["a", "b", "c", "all"])
    args = parser.parse_args()

    steps = [int(s.strip()) for s in args.steps.split(",")]

    print("=" * 60)
    print("  Substitution LM Pipeline")
    print("=" * 60)
    print(f"  Dataset   : {C.HF_DATASET} / {C.HF_CONFIG}")
    print(f"  Data dir  : {C.DATA_DIR}")
    print(f"  Cache dir : {C.CACHE_DIR}")
    print()

    for s in steps:
        print(f"── Step {s}: {STEP_SCRIPTS[s].name} {'─'*(45 - len(STEP_SCRIPTS[s].name))}")
        run_step(s)

    if args.eval:
        if not C.MODEL_PATH.exists():
            print("Model not found — run step 5 first.")
            sys.exit(1)
        print("── Evaluation ──────────────────────────────────────────")
        subprocess.run(
            [sys.executable, str(ROOT / "eval" / "ppl_bpb.py"),
             "--mechanism", args.mechanism],
            check=True,
        )


if __name__ == "__main__":
    main()
