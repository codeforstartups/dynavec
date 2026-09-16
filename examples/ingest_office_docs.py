"""Example: Ingesting Word (.docx), PowerPoint (.pptx), and Excel (.xlsx) files into dynavec."""

from pathlib import Path
from dynavec import Dynavec, DynavecConfig
from dynavec.ingest import DocxSource, PptxSource, XlsxSource, ingest

def main():
    cfg = DynavecConfig(
        vector_bucket="my-vectors",
        index="office-docs",
        table="dynavec_docs",
        dimension=1536,
        auto_provision=False,
    )
    db = Dynavec(cfg)

    # Ingest Word Document
    docx_file = Path("sample.docx")
    if docx_file.exists():
        docx_count = ingest(db, DocxSource(docx_file), namespace="word-docs")
        print(f"Ingested {docx_count} chunks from {docx_file}")

    # Ingest PowerPoint Presentation
    pptx_file = Path("presentation.pptx")
    if pptx_file.exists():
        pptx_count = ingest(db, PptxSource(pptx_file), namespace="slides")
        print(f"Ingested {pptx_count} chunks from {pptx_file}")

    # Ingest Excel Sheet
    xlsx_file = Path("sheets.xlsx")
    if xlsx_file.exists():
        xlsx_count = ingest(db, XlsxSource(xlsx_file), namespace="spreadsheets")
        print(f"Ingested {xlsx_count} chunks from {xlsx_file}")

if __name__ == "__main__":
    main()
