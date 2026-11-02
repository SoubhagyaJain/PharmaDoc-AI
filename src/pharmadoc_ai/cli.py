"""Reproducible local commands. Use --help on any subcommand."""

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

from pharmadoc_ai.config import ROOT, Settings
from pharmadoc_ai.errors import DocumentError


def _json(value):
    print(json.dumps(value, indent=2, default=str))


def create_examples(records, root: Path):
    import pymupdf

    from pharmadoc_ai.data.synthetic import create_scan

    destination = ROOT / "data/examples"
    destination.mkdir(parents=True, exist_ok=True)
    selected = {}
    for category, filename in (
        ("prescription", "prescription_digital"),
        ("clinical_form", "clinical_form"),
        ("regulatory", "regulatory"),
    ):
        available = [r for r in records if r["split"] == "train" and r["category"] == category]
        record = available[min(2, len(available) - 1)]
        shutil.copyfile(root / record["path"], destination / f"{filename}.pdf")
        (destination / f"{filename}.labels.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
        selected[category] = destination / f"{filename}.pdf"
    create_scan(selected["prescription"], destination / "prescription_scan.pdf")
    with (
        pymupdf.open(selected["prescription"]) as digital,
        pymupdf.open(destination / "prescription_scan.pdf") as scanned,
        pymupdf.open() as mixed,
    ):
        mixed.insert_pdf(digital, from_page=0, to_page=0)
        mixed.insert_pdf(scanned, from_page=0, to_page=0)
        mixed.save(destination / "mixed_pages.pdf", deflate=True)
    with pymupdf.open(selected["prescription"]) as doc:
        doc[0].get_pixmap(dpi=110).save(destination / "prescription.png")


def build_parser():
    parser = argparse.ArgumentParser(
        description="PharmaDoc AI: synthetic healthcare document intelligence"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="Generate the partitioned synthetic corpus")
    generate.add_argument("--output", type=Path, default=ROOT / "data/generated")
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--per-family", type=int, default=10)
    generate.add_argument("--examples", action="store_true")
    audit = commands.add_parser("audit", help="Check labels, duplicate hashes, and split leakage")
    audit.add_argument("manifest", type=Path)
    audit.add_argument(
        "--visual", action="store_true", help="Also report perceptually similar first pages"
    )
    inspect = commands.add_parser("inspect", help="Inspect page routes without OCR")
    inspect.add_argument("path", type=Path)
    process = commands.add_parser("process", help="Extract a document to evidence-linked JSON")
    process.add_argument("path", type=Path)
    process.add_argument("--output", type=Path)
    _processing_flags(process)
    train = commands.add_parser("train", help="Train local classifier and/or statistical NER")
    train.add_argument("manifest", type=Path)
    train.add_argument("--model", choices=["classifier", "ner", "all"], default="all")
    train.add_argument("--epochs", type=int, default=25)
    evaluate = commands.add_parser(
        "evaluate", help="Measure extraction; test requires explicit opt-in"
    )
    evaluate.add_argument("manifest", type=Path)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--source", choices=["gold", "digital", "scan"], default="digital")
    evaluate.add_argument("--split", choices=["validation", "test"], default="validation")
    evaluate.add_argument("--limit", type=int, default=0)
    evaluate.add_argument("--prescriptions-only", action="store_true")
    evaluate.add_argument("--allow-test", action="store_true")
    _processing_flags(evaluate)
    benchmark = commands.add_parser("benchmark-ocr", help="Paired OCR / preprocessing experiment")
    benchmark.add_argument("manifest", type=Path)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--split", choices=["validation", "test"], default="validation")
    benchmark.add_argument("--pages", type=int, default=30)
    benchmark.add_argument(
        "--engines", nargs="+", choices=["tesseract", "paddle"], default=["tesseract", "paddle"]
    )
    benchmark.add_argument(
        "--profiles",
        nargs="+",
        choices=["raw", "contrast", "adaptive", "deskew"],
        default=["raw", "contrast", "adaptive", "deskew"],
    )
    benchmark.add_argument("--allow-test", action="store_true")
    commands.add_parser("doctor", help="Inspect tools without running model inference")
    warmup = commands.add_parser(
        "warmup", help="Download/load one local engine and run a synthetic smoke check"
    )
    warmup.add_argument("--engine", choices=["paddle", "llm"], required=True)
    annotate = commands.add_parser(
        "export-labels", help="Export ten training-only Label Studio tasks"
    )
    annotate.add_argument("manifest", type=Path)
    annotate.add_argument("--output", type=Path, default=ROOT / "output/annotation_tasks.json")
    serve = commands.add_parser("serve", help="Start FastAPI on localhost")
    serve.add_argument("--port", type=int, default=8000)
    ui = commands.add_parser("ui", help="Start the local Streamlit reviewer")
    ui.add_argument("--port", type=int, default=8501)
    return parser


def _processing_flags(parser):
    parser.add_argument("--engine", choices=["tesseract", "paddle"], default="tesseract")
    parser.add_argument("--method", choices=["rules", "ner", "llm"], default="rules")
    parser.add_argument(
        "--profile", choices=["raw", "contrast", "adaptive", "deskew"], default="raw"
    )
    parser.add_argument("--force-ocr", action="store_true")


def main():
    args = build_parser().parse_args()
    settings = Settings()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    if hasattr(args, "method"):
        settings = settings.model_copy(
            update={
                "ocr_engine": args.engine,
                "extraction_method": args.method,
                "preprocessing": args.profile,
                "force_ocr": args.force_ocr,
            }
        )
    try:
        if args.command == "generate":
            from pharmadoc_ai.data.quality import check_dataset
            from pharmadoc_ai.data.synthetic import generate_dataset

            records = generate_dataset(args.output, args.seed, args.per_family)
            report = check_dataset(records, args.output)
            if args.examples:
                create_examples(records, args.output)
            _json(report)
            if not report["passed"]:
                return 1
        elif args.command == "audit":
            from pharmadoc_ai.data.quality import check_dataset

            records = json.loads(args.manifest.read_text(encoding="utf-8"))["records"]
            report = check_dataset(records, args.manifest.parent)
            if args.visual:
                from pharmadoc_ai.data.quality import visual_duplicate_candidates

                report["visual_duplicate_candidates"] = visual_duplicate_candidates(
                    records, args.manifest.parent
                )
            _json(report)
            return 0 if report["passed"] else 1
        elif args.command == "inspect":
            from pharmadoc_ai.ingestion import inspect_document

            _json([page.model_dump() for page in inspect_document(args.path, settings)])
        elif args.command == "process":
            from pharmadoc_ai.pipeline import process_document

            result = process_document(args.path, settings)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(result.model_dump_json(indent=2), encoding="utf-8")
                _json(
                    {
                        "output": args.output,
                        "status": result.status,
                        "human_review_required": result.human_review_required,
                    }
                )
            else:
                print(result.model_dump_json(indent=2))
        elif args.command == "train":
            records = json.loads(args.manifest.read_text(encoding="utf-8"))["records"]
            output = ROOT / "artifacts/training.json"
            report = json.loads(output.read_text(encoding="utf-8")) if output.is_file() else {}
            if args.model in ("classifier", "all"):
                from pharmadoc_ai.classification import train_classifier

                report["classifier"] = train_classifier(records, settings.classifier_path)
                print("Classifier training complete.", file=sys.stderr, flush=True)
            if args.model in ("ner", "all"):
                from pharmadoc_ai.extraction.ner import train_ner

                report["ner"] = train_ner(records, settings.ner_path, args.epochs)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2), encoding="utf-8")
            _json(report)
        elif args.command == "evaluate":
            from pharmadoc_ai.evaluation.run import evaluate

            _json(
                evaluate(
                    args.manifest,
                    args.output,
                    settings,
                    args.split,
                    args.source,
                    args.limit,
                    args.prescriptions_only,
                    args.allow_test,
                )
            )
        elif args.command == "benchmark-ocr":
            from pharmadoc_ai.evaluation.run import benchmark_ocr

            if not 1 <= args.pages <= 100:
                raise ValueError("Benchmark pages must be between 1 and 100.")
            report = benchmark_ocr(
                args.manifest,
                args.output,
                settings,
                args.split,
                args.engines,
                args.profiles,
                args.pages,
                args.allow_test,
            )
            _json({key: value for key, value in report.items() if key != "measurements"})
        elif args.command == "doctor":
            from pharmadoc_ai.ocr.tesseract import available

            _json(
                {
                    "python": sys.version,
                    "tesseract_available": available(settings),
                    "tesseract_cmd": settings.tesseract_cmd,
                    "paddle_python_present": settings.paddle_python.is_file(),
                    "classifier_present": settings.classifier_path.is_file(),
                    "ner_present": (settings.ner_path / "config.cfg").is_file(),
                    "ollama_model": settings.ollama_model,
                    "inference": "not_run",
                }
            )
        elif args.command == "warmup":
            from pharmadoc_ai.pipeline import process_document

            configured = settings.model_copy(
                update={
                    "ocr_engine": "paddle" if args.engine == "paddle" else "tesseract",
                    "extraction_method": "llm" if args.engine == "llm" else "rules",
                    "ocr_timeout_seconds": 180,
                    "llm_timeout_seconds": 90,
                    "document_timeout_seconds": 300,
                }
            )
            path = (
                ROOT
                / "data/examples"
                / (
                    "prescription_scan.pdf"
                    if args.engine == "paddle"
                    else "prescription_digital.pdf"
                )
            )
            result = process_document(path, configured)
            _json(
                {
                    "engine": args.engine,
                    "status": result.status,
                    "seconds": result.processing_duration_seconds,
                    "review_reasons": result.review_reasons,
                    "metadata": result.metadata,
                }
            )
        elif args.command == "export-labels":
            from pharmadoc_ai.data.annotation import export_label_studio

            records = json.loads(args.manifest.read_text(encoding="utf-8"))["records"]
            tasks = export_label_studio(records, args.output)
            _json({"tasks": len(tasks), "output": args.output, "partition": "train"})
        elif args.command == "serve":
            import uvicorn

            uvicorn.run("pharmadoc_ai.api:app", host="127.0.0.1", port=args.port)
        elif args.command == "ui":
            return subprocess.call(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    str(ROOT / "src/pharmadoc_ai/ui.py"),
                    "--server.address",
                    "127.0.0.1",
                    "--server.port",
                    str(args.port),
                    "--browser.gatherUsageStats",
                    "false",
                    "--server.maxUploadSize",
                    "20",
                ]
            )
    except DocumentError as exc:
        _json({"error": exc.code, "message": str(exc)})
        return 1
    except (ValueError, OSError):
        # Do not echo parser errors or filesystem exceptions that might contain document data.
        _json(
            {
                "error": "command_failed",
                "message": "Invalid command input or inaccessible local file.",
            }
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
