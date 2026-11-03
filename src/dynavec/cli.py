"""Command-line diagnostics for dynavec."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from typing import Any

from . import __version__
from .client import Dynavec
from .config import DynavecConfig


def _parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(prog="dynavec")
	subparsers = parser.add_subparsers(dest="command")
	subparsers.add_parser("version", help="show the installed dynavec version")
	doctor = subparsers.add_parser("doctor", help="check AWS credentials and resource access")
	doctor.add_argument("--bucket", help="S3 Vectors bucket name")
	doctor.add_argument("--index", help="S3 Vectors index name")
	doctor.add_argument("--table", help="DynamoDB table name")
	doctor.add_argument("--region", help="AWS region")
	doctor.add_argument("--profile", help="AWS profile name")

	export_cmd = subparsers.add_parser(
		"export", help="dump vectors and documents for a namespace to JSONL"
	)
	export_cmd.add_argument(
		"--namespace",
		"-n",
		default="default",
		help="namespace to export (default: 'default')",
	)
	export_cmd.add_argument(
		"--output",
		"-o",
		help="output JSONL file path (default: stdout)",
	)
	export_cmd.add_argument("--bucket", help="S3 Vectors bucket name (or DYNAVEC_VECTOR_BUCKET)")
	export_cmd.add_argument("--index", help="S3 Vectors index name (or DYNAVEC_INDEX)")
	export_cmd.add_argument("--table", help="DynamoDB table name (or DYNAVEC_TABLE)")
	export_cmd.add_argument("--dimension", type=int, help="vector dimension (default: 1536)")
	export_cmd.add_argument("--region", help="AWS region")
	export_cmd.add_argument("--profile", help="AWS profile name")

	import_cmd = subparsers.add_parser(
		"import", help="restore vectors and documents for a namespace from JSONL"
	)
	import_cmd.add_argument(
		"--namespace",
		"-n",
		default="default",
		help="target namespace (default: 'default')",
	)
	import_cmd.add_argument(
		"--input",
		"-i",
		help="input JSONL file path (default: stdin)",
	)
	import_cmd.add_argument(
		"--batch-size",
		type=int,
		default=100,
		help="batch size for upserting records (default: 100)",
	)
	import_cmd.add_argument("--bucket", help="S3 Vectors bucket name (or DYNAVEC_VECTOR_BUCKET)")
	import_cmd.add_argument("--index", help="S3 Vectors index name (or DYNAVEC_INDEX)")
	import_cmd.add_argument("--table", help="DynamoDB table name (or DYNAVEC_TABLE)")
	import_cmd.add_argument(
		"--dimension", type=int, help="vector dimension (inferred from input if omitted)"
	)
	import_cmd.add_argument("--region", help="AWS region")
	import_cmd.add_argument("--profile", help="AWS profile name")

	mcp = subparsers.add_parser("mcp", help="run the FastMCP server for AI clients")
	mcp.add_argument(
		"--transport",
		choices=["stdio", "sse"],
		default="stdio",
		help="Transport mode (default: stdio)",
	)
	mcp.add_argument(
		"--port",
		type=int,
		default=8000,
		help="Port for SSE transport (default: 8000)",
	)

	eval_cmd = subparsers.add_parser("eval", help="evaluation regression gating and trend commands")
	eval_sub = eval_cmd.add_subparsers(dest="eval_command")
	eval_check = eval_sub.add_parser(
		"check", help="compare eval run against baseline and fail on regression"
	)
	eval_check.add_argument("--current", "-c", required=True, help="path to current eval run JSON file")
	eval_check.add_argument(
		"--baseline", "-b", required=False, help="path to baseline eval run JSON file"
	)
	eval_check.add_argument(
		"--max-drop",
		type=float,
		default=0.05,
		help="max allowed relative drop (default: 0.05)",
	)
	eval_check.add_argument(
		"--threshold",
		action="append",
		dest="thresholds",
		metavar="METRIC=VAL",
		help="per-metric threshold override, e.g. --threshold recall@10=0.02 (repeatable)",
	)
	eval_check.add_argument(
		"--absolute",
		action="store_true",
		help="use absolute drop threshold instead of relative",
	)
	eval_check.add_argument(
		"--store",
		help="path to EvalRunStore JSON file to record/persist runs",
	)
	eval_check.add_argument(
		"--tag",
		action="append",
		dest="tags",
		metavar="KEY=VALUE",
		help="metadata tags to attach to the recorded run (repeatable)",
	)
	eval_check.add_argument("--json", action="store_true", help="output report as JSON")
	eval_check.add_argument(
		"--allow-missing-baseline",
		action="store_true",
		help="succeed with warning if baseline file does not exist",
	)
	return parser


def _session(profile: str | None, region: str | None) -> Any:
	import boto3

	kwargs: dict[str, str] = {}
	if profile:
		kwargs["profile_name"] = profile
	if region:
		kwargs["region_name"] = region
	return boto3.Session(**kwargs)


def _resolve_resources(args: argparse.Namespace) -> tuple[str, str, str]:
	import os

	bucket = args.bucket or os.environ.get("DYNAVEC_VECTOR_BUCKET") or os.environ.get("DYNAVEC_BUCKET")
	index = args.index or os.environ.get("DYNAVEC_INDEX")
	table = args.table or os.environ.get("DYNAVEC_TABLE")

	missing = []
	if not bucket:
		missing.append("--bucket")
	if not index:
		missing.append("--index")
	if not table:
		missing.append("--table")

	if missing:
		raise ValueError(
			f"Missing required resource configuration: {', '.join(missing)} "
			"(provide via flags or DYNAVEC_* environment variables)."
		)
	return str(bucket), str(index), str(table)


def _export(args: argparse.Namespace) -> int:
	try:
		bucket, index, table = _resolve_resources(args)
	except ValueError as exc:
		print(f"[FAIL] {exc}", file=sys.stderr)
		return 1

	session = _session(args.profile, args.region)
	dimension = args.dimension or 1536
	config = DynavecConfig(
		vector_bucket=bucket,
		index=index,
		table=table,
		dimension=dimension,
		region=args.region,
	)
	db = Dynavec(config, boto_session=session)

	try:
		if args.output and args.output != "-":
			count = db.export_namespace(args.output, namespace=args.namespace)
			print(f"Exported {count} documents from namespace '{args.namespace}' to {args.output}")
		else:
			count = db.export_namespace(sys.stdout, namespace=args.namespace)
			print(f"Exported {count} documents from namespace '{args.namespace}'", file=sys.stderr)
		return 0
	except Exception as exc:
		print(f"[FAIL] Export failed: {exc}", file=sys.stderr)
		return 1


def _import(args: argparse.Namespace) -> int:
	import json

	try:
		bucket, index, table = _resolve_resources(args)
	except ValueError as exc:
		print(f"[FAIL] {exc}", file=sys.stderr)
		return 1

	session = _session(args.profile, args.region)
	dimension = args.dimension

	if dimension is None and args.input and args.input != "-":
		try:
			with open(args.input, encoding="utf-8") as f:
				for line in f:
					line = line.strip()
					if line:
						first_item = json.loads(line)
						if "vector" in first_item and isinstance(first_item["vector"], list):
							dimension = len(first_item["vector"])
							break
		except Exception:
			pass

	if dimension is None:
		try:
			client = session.client("s3vectors", region_name=args.region)
			idx = client.get_index(vectorBucketName=bucket, indexName=index)
			dimension = idx.get("index", {}).get("dimension")
		except Exception:
			pass

	if dimension is None:
		dimension = 1536

	config = DynavecConfig(
		vector_bucket=bucket,
		index=index,
		table=table,
		dimension=dimension,
		region=args.region,
	)
	db = Dynavec(config, boto_session=session)

	try:
		if args.input and args.input != "-":
			count = db.import_namespace(
				args.input, namespace=args.namespace, batch_size=args.batch_size
			)
			print(f"Imported {count} documents into namespace '{args.namespace}' from {args.input}")
		else:
			count = db.import_namespace(
				sys.stdin, namespace=args.namespace, batch_size=args.batch_size
			)
			print(f"Imported {count} documents into namespace '{args.namespace}'")
		return 0
	except Exception as exc:
		print(f"[FAIL] Import failed: {exc}", file=sys.stderr)
		return 1


def _check(label: str, callback: Callable[[], str]) -> bool:
	try:
		detail = callback()
	except Exception as exc:  # noqa: BLE001
		print(f"[FAIL] {label}\n       {exc}")
		return False
	print(f"[PASS] {label}")
	if detail:
		print(f"       {detail}")
	return True


def _doctor(args: argparse.Namespace) -> int:
	print("Dynavec doctor\n")
	session = None
	checks_passed = True

	def get_session() -> Any:
		nonlocal session
		if session is None:
			session = _session(args.profile, args.region)
		return session

	checks_passed &= _check(
		"AWS credentials / STS identity",
		lambda: _identity(get_session()),
	)

	if args.bucket or args.index:
		if not args.bucket or not args.index:
			print("[FAIL] S3 Vectors configuration\n       --bucket and --index must be provided together")
			checks_passed = False
		else:
			checks_passed &= _check(
				f"S3 Vectors index: {args.index}",
				lambda: _check_s3vectors(get_session(), args.bucket, args.index, args.region),
			)

	if args.table:
		checks_passed &= _check(
			f"DynamoDB table: {args.table}",
			lambda: _check_dynamodb(get_session(), args.table, args.region),
		)

	print("\nDoctor checks passed." if checks_passed else "\nDoctor checks failed.")
	return 0 if checks_passed else 1


def _identity(session: Any) -> str:
	identity = session.client("sts").get_caller_identity()
	return f"Account: {identity.get('Account', 'unknown')}"


def _check_s3vectors(session: Any, bucket: str, index: str, region: str | None) -> str:
	client = session.client("s3vectors", region_name=region)
	client.get_index(vectorBucketName=bucket, indexName=index)
	return f"Bucket: {bucket}"


def _check_dynamodb(session: Any, table: str, region: str | None) -> str:
	session.client("dynamodb", region_name=region).describe_table(TableName=table)
	return "Accessible"


def _eval_check(args: argparse.Namespace) -> int:
	import json
	from pathlib import Path

	from .eval.gate import check_regression
	from .eval.trend import EvalRun, EvalRunStore

	current_path = Path(args.current)
	if not current_path.exists():
		print(f"[FAIL] Current eval file not found: {args.current}", file=sys.stderr)
		return 2

	tag_dict: dict[str, str] = {}
	if args.tags:
		for item in args.tags:
			if "=" in item:
				k, v = item.split("=", 1)
				tag_dict[k.strip()] = v.strip()

	if args.store:
		try:
			import time
			import uuid

			from .eval.gate import _extract_metrics

			store = EvalRunStore.load_json(args.store)
			curr_metrics, c_run_id = _extract_metrics(current_path)
			run_id = c_run_id or uuid.uuid4().hex[:12]
			run = EvalRun(
				run_id=run_id,
				ts=time.time(),
				tags=tag_dict,
				metrics=curr_metrics,
			)
			store._runs.append(run)
			store.save_json(args.store)
		except Exception as exc:
			print(f"[WARN] Failed to update trend store at {args.store}: {exc}", file=sys.stderr)

	if not args.baseline:
		if args.allow_missing_baseline:
			print(
				"[WARN] No baseline provided and --allow-missing-baseline set. Skipping regression check."
			)
			return 0
		print(
			"[FAIL] Missing required --baseline argument (or pass --allow-missing-baseline).",
			file=sys.stderr,
		)
		return 2

	baseline_path = Path(args.baseline)
	if not baseline_path.exists():
		if args.allow_missing_baseline:
			print(
				f"[WARN] Baseline file not found: {args.baseline}. Bootstrapping mode: skipping regression check."
			)
			return 0
		print(f"[FAIL] Baseline file not found: {args.baseline}", file=sys.stderr)
		return 2

	metric_thresholds: dict[str, float] = {}
	if args.thresholds:
		for item in args.thresholds:
			if "=" in item:
				m, val = item.split("=", 1)
				try:
					metric_thresholds[m.strip()] = float(val.strip())
				except ValueError:
					print(f"[FAIL] Invalid threshold value for {m}: {val}", file=sys.stderr)
					return 2

	try:
		report = check_regression(
			current=current_path,
			baseline=baseline_path,
			max_drop=args.max_drop,
			metric_thresholds=metric_thresholds,
			relative=not args.absolute,
		)
	except Exception as exc:
		print(f"[FAIL] Regression check error: {exc}", file=sys.stderr)
		return 2

	if args.json:
		print(json.dumps(report.to_dict(), indent=2))
	else:
		print(report.summary_table())

	return 0 if report.passed else 1


def main(argv: list[str] | None = None) -> int:
	args = _parser().parse_args(argv)
	if args.command == "version":
		print(__version__)
		return 0
	if args.command == "doctor":
		return _doctor(args)
	if args.command == "export":
		return _export(args)
	if args.command == "import":
		return _import(args)
	if args.command == "mcp":
		from .mcp.server import create_mcp_server

		server = create_mcp_server()
		if args.transport == "sse":
			server.settings.port = args.port
			server.run(transport="sse")
		else:
			server.run(transport="stdio")
		return 0
	if args.command == "eval":
		if getattr(args, "eval_command", None) == "check":
			return _eval_check(args)
		_parser().parse_args(["eval", "--help"])
		return 0
	_parser().print_help()
	return 0


if __name__ == "__main__":
	sys.exit(main())
