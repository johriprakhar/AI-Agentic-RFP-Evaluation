import pymupdf

def extract_pdf_pages(pdf_bytes):
    """Extract text page-by-page so evidence can be associated with source pages."""
    pages = []
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as pdf:
        for index, page in enumerate(pdf):
            pages.append({"page": index + 1, "text": page.get_text("text").strip()})
    return pages
