import os
import io
import pymupdf
import pytesseract
from PIL import Image
from langchain_core.tools import tool
from typing import Optional


@tool
def document_ingestion_tool(file_bytes: bytes, filename: str) -> dict:
    """
    Reads PDF or image bytes and returns clean text for LLM usage.

    Args:
        file_bytes: Raw bytes of the input file (PDF or image).
        filename: The filename (used to determine file extension).

    Returns:
        A dict containing the extracted `text`, the `source_type` and a
        `confidence` score for the extraction.
    """
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        text = ""

        for page in doc:
            text += page.get_text()

        return {
            "text": text.strip(),
            "source_type": "pdf_text",
            "confidence": 0.95
        }

    elif ext in [".png", ".jpg", ".jpeg"]:
        image = Image.open(io.BytesIO(file_bytes))
        text = pytesseract.image_to_string(image)

        return {
            "text": text.strip(),
            "source_type": "ocr_image",
            "confidence": 0.85
        }

    else:
        raise ValueError("Unsupported file type")
