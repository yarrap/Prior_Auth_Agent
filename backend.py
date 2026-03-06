from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path

# Import your agent function
from agent import run_agent

app = FastAPI(title="Prior Authorization Agent API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create temp directory for storing text inputs
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)


class TextInput(BaseModel):
    """User's text input"""
    text: str


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "message": "Prior Authorization Agent API is running!",
        "usage": "POST /process with {\"text\": \"your message here\"}"
    }


@app.post("/process")
async def process_text(input_data: TextInput):
    """
    Process user's text input directly
    
    User sends their text directly (pasted or typed).
    
    Example request:
    {
        "text": "Patient John Doe needs prior auth for MRI scan. 
                 He has chronic lower back pain. CPT code 72148."
    }
    
    Returns:
        Agent's analysis and recommendations
    """
    
    temp_file = None
    
    try:
        print(f"📝 Received text input: {input_data.text[:100]}...")
        
        # Save user's text to a temporary file
        # (Your agent currently needs a file path)
        temp_file = TEMP_DIR / "user_input.txt"
        temp_file.write_text(input_data.text)
        
        # Run your agent with the text file
        result = run_agent(str(temp_file))
        
        # Extract the agent's response
        ai_message = result['messages'][-1]
        
        # Clean up temp file
        temp_file.unlink()
        
        return {
            "status": "success",
            "agent_response": ai_message.content,
            "message_count": len(result['messages'])
        }
        
    except Exception as e:
        # Clean up temp file if error occurs
        if temp_file and temp_file.exists():
            temp_file.unlink()
        
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("Starting Prior Authorization Agent API")
    print("="*60)
    print("API available at: http://localhost:8000")
    print("API docs at: http://localhost:8000/docs")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)