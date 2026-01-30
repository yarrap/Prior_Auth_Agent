from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from tools.docu_tool import document_ingestion_tool
from tools.extraction_tool import evaluate_with_mistral_and_export_csv
from tools.filter_policies_tool import filter_policies_simple   
from tools.entity_extraction_tool import extract_entities
from langchain.chat_models import init_chat_model
from prompt import system_prompt
import json

# Define local inputs used by the runner
with open("new_structured_policies.json", "r") as f:
    all_policies = json.load(f)

with open("data/input_data/document_03.pdf", "rb") as f:
    file_bytes = f.read()

csv_path = "prior_auth_human_review.csv"
filename = "document_03.pdf"

model = init_chat_model("gpt-5-nano")
# model = init_chat_model(
#     "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     model_provider="huggingface",
#     temperature=0.7,
#     max_tokens=1024
# )

agent_executor = create_agent(
    model=model,
    tools=[document_ingestion_tool, extract_entities, filter_policies_simple, evaluate_with_mistral_and_export_csv],
    system_prompt=system_prompt  
)

# Invoke the agent
def run_agent(file_bytes, filename, csv_path, all_policies):
    response = agent_executor.invoke({
        "messages": [
            HumanMessage(content=f"""
            Process a prior authorization request:
            1. Extract text from PDF: {filename}
            2. Evaluate against policies
            3. Save results to: {csv_path}
            """)
        ]
    })
    return response

result = run_agent(file_bytes, "document_03.pdf", csv_path, all_policies)

ai_message = result['messages'][-1]  

print(ai_message.content)