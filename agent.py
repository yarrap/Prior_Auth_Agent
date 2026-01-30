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

print("Starting agent...")

# File path for testing
file_path = "data/input_data/document_03.pdf"

# Initialize LLM
model = init_chat_model("gpt-5-nano")

# Create agent with only the ingestion tool for testing
agent_executor = create_agent(
    model=model,
    tools=[document_ingestion_tool, extract_entities, filter_policies_simple, evaluate_with_mistral_and_export_csv],
    system_prompt=system_prompt
)

def run_agent(file_path):
    """
    Run the agent with a tool-invocation style prompt to ensure the
    document ingestion tool is called exactly once.
    """
    print("➡️ Invoking agent...")

    response = agent_executor.invoke({
        "messages": [
            HumanMessage(content=f"Process a prior authorization request using this document: {file_path}")
        ]
    })
    return response

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




















# # lang_agent/agent.py
# from langchain.agents import create_agent
# from langchain_core.messages import HumanMessage
# from tools.docu_tool import document_ingestion_tool
# from tools.entity_extraction_tool import extract_entities
# from langchain.chat_models import init_chat_model
# from prompt import system_prompt
# import json
# import os

# # # Load structured policies
# # with open("new_structured_policies.json", "r") as f:
# #     all_policies = json.load(f)
# # print(f"Loaded {len(all_policies)} structured policies.")
# print("Starting agent...")

# # File paths
# file_path = "data/input_data/document_03.pdf"
# csv_path = "prior_auth_human_review.csv"

# # Initialize LLM
# model = init_chat_model("gpt-5-nano")

# # Create agent with only the first tool for testing
# agent_executor = create_agent(
#     model=model,
#     tools=[document_ingestion_tool],
#     system_prompt=system_prompt
#     # verbose=True
# )

# # Run the agent
# def run_agent(file_path):
#     # Agent receives instruction to call the document ingestion tool itself
#     print("➡️ Invoking agent...")
#     response = agent_executor.invoke({
#         "messages": [
#             HumanMessage(content=f"""
#             You are a medical document agent. 
#             Please use your tools to extract the text from this document:
#             {file_path}
#             Return the extracted text and metadata like source_type and confidence.
#             """)
#         ]
#     })
#     return response

# # Execute
# result = run_agent(file_path)

# print("\n🧾 ALL MESSAGES:")
# for i, msg in enumerate(result["messages"]):
#     print(f"\n--- Message {i} ({type(msg).__name__}) ---")
#     print(msg.content)

# # Print agent's final message
# ai_message = result['messages'][-1]
# print("Agent output:\n")
# print(ai_message.content)
















# # # from langchain.agents import create_agent
# # # from langchain_core.messages import HumanMessage
# # # from tools.docu_tool import document_ingestion_tool
# # # from tools.extraction_tool import evaluate_with_mistral_and_export_csv
# # # from tools.filter_policies_tool import filter_policies_simple   
# # # from tools.entity_extraction_tool import extract_entities
# # # from langchain.chat_models import init_chat_model
# # # from prompt import system_prompt
# # # import json

# # # # Define local inputs used by the runner
# # # with open("new_structured_policies.json", "r") as f:
# # #     all_policies = json.load(f)

# # # with open("data/input_data/document_03.pdf", "rb") as f:
# # #     file_bytes = f.read()

# # # csv_path = "prior_auth_human_review.csv"
# # # filename = "document_03.pdf"

# # # model = init_chat_model("gpt-5-nano")
# # # # model = init_chat_model(
# # # #     "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
# # # #     model_provider="huggingface",
# # # #     temperature=0.7,
# # # #     max_tokens=1024
# # # # )

# # # agent_executor = create_agent(
# # #     model=model,
# # #     tools=[document_ingestion_tool, extract_entities, filter_policies_simple, evaluate_with_mistral_and_export_csv],
# # #     system_prompt=system_prompt  
# # # )

# # # # Invoke the agent
# # # def run_agent(file_bytes, filename, csv_path, all_policies):
# # #     response = agent_executor.invoke({
# # #         "messages": [
# # #             HumanMessage(content=f"""
# # #             Process a prior authorization request:
# # #             1. Extract text from PDF: {filename}
# # #             2. Evaluate against policies
# # #             3. Save results to: {csv_path}
# # #             """)
# # #         ]
# # #     })
# # #     return response

# # # result = run_agent(file_bytes, "document_03.pdf", csv_path, all_policies)

# # # ai_message = result['messages'][-1]  

# # # print(ai_message.content)


# # # lang_agent/agent.py
# # from langchain.agents import create_agent
# # from langchain_core.messages import HumanMessage
# # from tools.docu_tool import document_ingestion_tool
# # from tools.extraction_tool import evaluate_with_mistral_and_export_csv
# # from tools.filter_policies_tool import filter_policies_simple   
# # from tools.entity_extraction_tool import extract_entities
# # from langchain.chat_models import init_chat_model
# # from prompt import system_prompt
# # import json
# # import os
# # import sys

# # # Load structured policies
# # with open("new_structured_policies.json", "r") as f:
# #     all_policies = json.load(f)

# # print(f"Loaded {len(all_policies)} structured policies.")
# # # sys.exit()

# # # File paths
# # file_path = "data/input_data/document_03.pdf"
# # csv_path = "prior_auth_human_review.csv"

# # # Initialize LLM
# # model = init_chat_model("gpt-5-nano")

# # # Create agent with tools
# # agent_executor = create_agent(
# #     model=model,
# #     tools=[document_ingestion_tool, extract_entities],
# #     system_prompt=system_prompt  
# # )
# # # filter_policies_simple, evaluate_with_mistral_and_export_csv

# # # Run the agent
# # def run_agent(file_path, csv_path, all_policies):
# #     # Step 1: Extract text using the document ingestion tool
# #     # doc_output = document_ingestion_tool(file_path)
# #     doc_output = document_ingestion_tool.invoke({
# #     "file_path": file_path})
# #     print("Tool called: document_ingestion_tool")
# #     print(f"Source type: {doc_output['source_type']}, Confidence: {doc_output['confidence']}")
# #     print(f"Extracted text preview: {doc_output['text'][:500]}...\n")  # Show first 500 chars

# #     # Step 2: Agent invokes the other tools via messages
# #     response = agent_executor.invoke({
# #         "messages": [
# #             HumanMessage(content=f"""
# #             Process a prior authorization request:
# #             1. Extracted text from {file_path}:
# #             {doc_output['text'][:500]}...
# #             2. Evaluate against policies
# #             3. Save results to: {csv_path}
# #             """)
# #         ]
# #     })
# #     return response

# # # Execute
# # result = run_agent(file_path, csv_path, all_policies)

# # # Print agent's final message
# # ai_message = result['messages'][-1]
# # print("Agent output:\n")
# # print(ai_message.content)
