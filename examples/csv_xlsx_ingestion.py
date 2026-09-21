"""Example: Ingesting CSV and Excel (.xlsx) spreadsheets into dynavec.

Each data row becomes its own Record (first row = header), so rows stay
independently searchable once chunked and embedded.
"""

from pathlib import Path

from dynavec import Dynavec, DynavecConfig
from dynavec.ingest import CsvSource, XlsxSource, ingest


def main():
    cfg = DynavecConfig(
        vector_bucket="my-vectors",
        index="spreadsheets",
        table="dynavec_spreadsheets",
        dimension=1536,
        auto_provision=False,
    )
    db = Dynavec(cfg)

    # Ingest a CSV file
    csv_file = Path("data.csv")
    if csv_file.exists():
        csv_count = ingest(db, CsvSource(csv_file), namespace="csv-rows")
        print(f"Ingested {csv_count} chunks from {csv_file}")

    # Ingest an Excel workbook
    xlsx_file = Path("sheets.xlsx")
    if xlsx_file.exists():
        xlsx_count = ingest(db, XlsxSource(xlsx_file), namespace="xlsx-rows")
        print(f"Ingested {xlsx_count} chunks from {xlsx_file}")


if __name__ == "__main__":
    main()
