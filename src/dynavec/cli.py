"""Command-line diagnostics for dynavec."""

from __future__ import annotations

import argparse
import sys


def _parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(prog="dynavec")
	subparsers = parser.add_subparsers(dest="command")
	doctor = subparsers.add_parser("doctor", help="check AWS credentials and resource access")
	doctor.add_argument("--bucket", help="S3 Vectors bucket name")
	doctor.add_argument("--index", help="S3 Vectors index name")
	doctor.add_argument("--table", help="DynamoDB table name")
	doctor.add_argument("--region", help="AWS region")
	doctor.add_argument("--profile", help="AWS profile name")
	return parser


def _session(profile: str | None, region: str | None):
	import boto3

	kwargs = {}
	if profile:
		kwargs["profile_name"] = profile
	if region:
		kwargs["region_name"] = region
	return boto3.Session(**kwargs)


def _check(label: str, callback) -> bool:
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

	def get_session():
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


def _identity(session) -> str:
	identity = session.client("sts").get_caller_identity()
	return f"Account: {identity.get('Account', 'unknown')}"


def _check_s3vectors(session, bucket: str, index: str, region: str | None) -> str:
	client = session.client("s3vectors", region_name=region)
	client.get_index(vectorBucketName=bucket, indexName=index)
	return f"Bucket: {bucket}"


def _check_dynamodb(session, table: str, region: str | None) -> str:
	session.client("dynamodb", region_name=region).describe_table(TableName=table)
	return "Accessible"


def main(argv: list[str] | None = None) -> int:
	args = _parser().parse_args(argv)
	if args.command == "doctor":
		return _doctor(args)
	_parser().print_help()
	return 0


if __name__ == "__main__":
	sys.exit(main())
