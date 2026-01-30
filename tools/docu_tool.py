# lang_agent/tools/docu_tool.py
import os
from typing import Optional
from PIL import Image
import pytesseract
import fitz  
from langchain_core.tools import tool


@tool
def document_ingestion_tool(file_path: str) -> dict:
    """
    Reads a PDF or image file and returns clean text for LLM usage.

    Args:
        file_path: Path to the input file (PDF or image).

    Returns:
        A dict containing the extracted `text`, the `source_type`, and a
        `confidence` score for the extraction.
    """
    print(f"📄 document_ingestion_tool CALLED with: {file_path}")

    filename = os.path.basename(file_path)
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()

            return {
                "text": text.strip(),
                "source_type": "pdf_text",
                "confidence": 0.95
            }
        except Exception as e:
            return {
                "text": "",
                "source_type": "pdf_text",
                "confidence": 0.0,
                "error": f"Failed to read PDF: {e}"
            }

    elif ext in [".png", ".jpg", ".jpeg"]:
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            return {
                "text": text.strip(),
                "source_type": "ocr_image",
                "confidence": 0.85
            }
        except Exception as e:
            return {
                "text": "",
                "source_type": "ocr_image",
                "confidence": 0.0,
                "error": f"Failed to read image: {e}"
            }

    else:
        return {
            "text": "",
            "source_type": "unsupported",
            "confidence": 0.0,
            "error": f"Unsupported file type: {ext}"
        }














# import os
# import io
# import pymupdf
# import pytesseract
# from PIL import Image
# from langchain_core.tools import tool
# from typing import Optional


# @tool
# def document_ingestion_tool(file_bytes: bytes, filename: str) -> dict:
#     """
#     Reads PDF or image bytes and returns clean text for LLM usage.

#     Args:
#         file_bytes: Raw bytes of the input file (PDF or image).
#         filename: The filename (used to determine file extension).

#     Returns:
#         A dict containing the extracted `text`, the `source_type` and a
#         `confidence` score for the extraction.
#     """
#     ext = os.path.splitext(filename)[1].lower()

#     if ext == ".pdf":
#         doc = pymupdf.open(stream=file_bytes, filetype="pdf")
#         text = ""

#         for page in doc:
#             text += page.get_text()

#         return {
#             "text": text.strip(),
#             "source_type": "pdf_text",
#             "confidence": 0.95
#         }

#     elif ext in [".png", ".jpg", ".jpeg"]:
#         image = Image.open(io.BytesIO(file_bytes))
#         text = pytesseract.image_to_string(image)

#         return {
#             "text": text.strip(),
#             "source_type": "ocr_image",
#             "confidence": 0.85
#         }

#     else:
#         raise ValueError("Unsupported file type")
