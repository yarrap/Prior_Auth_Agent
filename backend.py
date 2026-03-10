from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from typing import Optional
import shutil
import uuid

from agent import run_agent, resume_agent

app = FastAPI(title="Prior Authorization Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".txt", ".doc", ".docx"}


class TextInput(BaseModel):
    text: str


class ResumeInput(BaseModel):
    thread_id:   str
    decision:    str                    # "approve" | "reject" | "edit"
    edited_args: Optional[dict] = None  # only needed when decision == "edit"


@app.get("/")
async def root():
    return {
        "status": "running",
        "message": "Prior Authorization Agent API is running!",
        "endpoints": {
            "POST /process":      "Send plain text",
            "POST /process-file": "Upload PDF / image / text file",
            "POST /resume":       "Resume a paused agent (approve / reject / edit)"
        }
    }


@app.post("/process")
async def process_text(input_data: TextInput):
    """Process plain text typed in the chat box."""
    temp_file = None
    try:
        print(f"📝 Received text input: {input_data.text[:100]}...")

        temp_file = TEMP_DIR / f"{uuid.uuid4().hex}.txt"
        temp_file.write_text(input_data.text)

        result = run_agent(str(temp_file))

        # ✅ result is now a dict — NOT result['messages'][-1]
        if result["interrupted"]:
            return {
                "status":            "interrupted",
                "thread_id":         result["thread_id"],
                "pending_tool_call": result["pending_tool_call"],
                "message":           f"Agent paused before: {result['pending_tool_call']['name']}"
            }

        return {
            "status":         "success",
            "agent_response": result["agent_response"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

    finally:
        if temp_file and temp_file.exists():
            temp_file.unlink()


@app.post("/process-file")
async def process_file(
    file: UploadFile = File(...),
    note: Optional[str] = Form(None)
):
    """
    Upload a document (PDF, PNG, JPG, TXT, DOC, DOCX).
    Returns either:
      - { status: "interrupted", thread_id, pending_tool_call }
      - { status: "success", agent_response }
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    unique_name = f"{uuid.uuid4().hex}{ext}"
    temp_path   = TEMP_DIR / unique_name

    try:
        print(f"📄 Received file: {file.filename} ({file.content_type})")
        with temp_path.open("wb") as out:
            shutil.copyfileobj(file.file, out)

        if note and note.strip():
            print(f"📝 User note: {note.strip()}")

        result = run_agent(str(temp_path))

        # ✅ result is now a dict — NOT result['messages'][-1]
        if result["interrupted"]:
            tc = result["pending_tool_call"]
            return {
                "status":            "interrupted",
                "thread_id":         result["thread_id"],
                "pending_tool_call": tc,
                "message":           f"Agent wants to read: {tc['args'].get('file_path', file.filename)}"
            }

        return {
            "status":         "success",
            "filename":       file.filename,
            "agent_response": result["agent_response"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

    finally:
        # Only clean up if NOT interrupted — file still needed for resume
        if temp_path.exists() and not (
            'result' in locals() and result.get("interrupted")
        ):
            temp_path.unlink()


@app.post("/resume")
async def resume(input_data: ResumeInput):
    """
    Resume a paused agent after human decision.

    Body:
      {
        "thread_id":   "abc-123",
        "decision":    "approve" | "reject" | "edit",
        "edited_args": { "file_path": "new/path.pdf" }  // only for edit
      }
    """
    if input_data.decision not in ("approve", "reject", "edit"):
        raise HTTPException(
            status_code=400,
            detail="decision must be one of: approve | reject | edit"
        )

    if input_data.decision == "edit" and not input_data.edited_args:
        raise HTTPException(
            status_code=400,
            detail="edited_args is required when decision is 'edit'"
        )

    try:
        result = resume_agent(
            thread_id=input_data.thread_id,
            decision=input_data.decision,
            edited_args=input_data.edited_args
        )

        if result["interrupted"]:
            return {
                "status":            "interrupted",
                "thread_id":         result["thread_id"],
                "pending_tool_call": result["pending_tool_call"],
                "message":           f"Agent paused again before: {result['pending_tool_call']['name']}"
            }

        return {
            "status":         "success",
            "agent_response": result["agent_response"]
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resuming agent: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("Starting Prior Authorization Agent API")
    print("="*60)
    print("API available at: http://localhost:8000")
    print("API docs at:      http://localhost:8000/docs")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)



