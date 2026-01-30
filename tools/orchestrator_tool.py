from smolagents import tool
from tools.docu_tool import document_ingestion_tool
from tools.entity_extraction_tool import entity_extraction_tool


@tool
def orchestrator_tool(filepath: str) -> dict:
    """
    Run `document_ingestion_tool` then `entity_extraction_tool` on a local file.

    Args:
        filepath: Path to a local file (PDF or image) accessible to this process.

    Returns:
        A dict with keys `ingestion` (raw ingestion output) and `entities` (structured output).
    """
    # Read file bytes from the given filepath
    with open(filepath, "rb") as f:
        file_bytes = f.read()

    # infer filename from filepath
    import os
    filename = os.path.basename(filepath)

    ingest = document_ingestion_tool(file_bytes, filename)
    text = ingest.get("text", "")
    entities = entity_extraction_tool(text)
    return {"ingestion": ingest, "entities": entities}
