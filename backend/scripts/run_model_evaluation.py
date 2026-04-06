from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.database import SessionLocal, init_db
from app.services.model_evaluation_service import evaluation_service


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SONA model evaluation and persist prediction logs.")
    parser.add_argument("--input-type", default="all", choices=["all", "face", "voice"])
    parser.add_argument("--output-json", default="", help="Optional path to save the evaluation response.")
    args = parser.parse_args()

    init_db()
    with SessionLocal() as db:
        result = evaluation_service.run_evaluation(db, input_type=args.input_type)

    print(json.dumps(result, indent=2))
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
