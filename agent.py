# lang_agent/agent.py
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from tools.docu_tool import document_ingestion_tool
from tools.entity_extraction_tool import extract_entities
from tools.filter_policies_tool import filter_policies_simple
from tools.extraction_tool import evaluate_with_mistral_and_export_csv
from langchain.chat_models import init_chat_model
from prompt import system_prompt
import os
from pathlib import Path

print("Starting agent...")

# Initialize LLM
model = init_chat_model("gpt-5-nano")

# Create agent with tools
agent_executor = create_agent(
    model=model,
    tools=[document_ingestion_tool, extract_entities, filter_policies_simple, evaluate_with_mistral_and_export_csv],
    system_prompt=system_prompt
)

def run_agent(input_data):
    """
    Run the agent with either a file path OR text content
    
    Args:
        input_data: Either a file path (str) or text content (str)
    
    Returns:
        Agent response
    """
    print("➡️ Invoking agent...")
    
    # Check if it's a file path
    if Path(input_data).exists() and Path(input_data).suffix == '.pdf':
        # It's a PDF file - use the document tool
        message_content = f"Process a prior authorization request using this document: {input_data}"
    else:
        # It's text content or a txt file - handle directly
        if Path(input_data).exists() and Path(input_data).suffix == '.txt':
            # Read the text file
            with open(input_data, 'r') as f:
                text_content = f.read()
        else:
            # It's direct text content
            text_content = input_data
        
        # Send text directly to the agent (skip document ingestion tool)
        message_content = f"""Process this prior authorization request:

{text_content}

Please extract the relevant entities, evaluate against policies, and provide your analysis."""

    response = agent_executor.invoke({
        "messages": [
            HumanMessage(content=message_content)
        ]
    })
    return response


# Only run test code when executing directly
if __name__ == "__main__":
    # File path for testing
    file_path = "data/input_data/document_03.pdf"
    
    # Execute agent
    result = run_agent(file_path)

    # Print all messages for debug
    print("\n🧾 ALL MESSAGES:")
    for i, msg in enumerate(result["messages"]):
        print(f"\n--- Message {i} ({type(msg).__name__}) ---")
        print(msg.content)

    # Print the final agent output
    ai_message = result['messages'][-1]
    print("\n✅ Agent output:")
    print(ai_message.content)












