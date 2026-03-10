# lang_agent/agent.py
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode
from prompt import system_prompt
from tools.docu_tool import document_ingestion_tool
from tools.entity_extraction_tool import extract_entities
from tools.filter_policies_tool import filter_policies_simple
from tools.extraction_tool import evaluate_with_mistral_and_export_csv
from pathlib import Path
import uuid

print("Starting agent...")

# ── LLM ──
model = init_chat_model("gpt-5-nano")
checkpointer = InMemorySaver()

# ── Bind all tools to the model ──
all_tools = [
    document_ingestion_tool,
    extract_entities,
    filter_policies_simple,
    evaluate_with_mistral_and_export_csv
]
model_with_tools = model.bind_tools(all_tools)

# ── Tool nodes ──
# document_ingestion_tool gets its OWN node so we can interrupt before it selectively
# All other tools share a node and run freely without interruption
ingest_node    = ToolNode([document_ingestion_tool])
remaining_node = ToolNode([extract_entities, filter_policies_simple, evaluate_with_mistral_and_export_csv])


# ── Agent node: calls the LLM ──
def call_model(state: MessagesState):
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}


# ── Router: after model responds, decide which node to go to ──
def route_after_model(state: MessagesState):
    last = state["messages"][-1]

    # No tool calls → we're done
    if not hasattr(last, "tool_calls") or not last.tool_calls:
        return END

    # If ANY tool call is document_ingestion_tool → go to ingest_node (will be interrupted)
    for tc in last.tool_calls:
        if tc["name"] == "document_ingestion_tool":
            return "ingest_node"

    # All other tools → go to remaining_node (no interrupt)
    return "remaining_node"


# ── Build the graph ──
builder = StateGraph(MessagesState)

builder.add_node("agent",          call_model)
builder.add_node("ingest_node",    ingest_node)     # ← interrupt_before targets this node
builder.add_node("remaining_node", remaining_node)  # ← runs freely, no interrupt

builder.add_edge(START,            "agent")
builder.add_conditional_edges("agent", route_after_model, ["ingest_node", "remaining_node", END])
builder.add_edge("ingest_node",    "agent")
builder.add_edge("remaining_node", "agent")

# ✅ interrupt_before targets the node name, NOT the tool name
agent_executor = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["ingest_node"]  # ONLY interrupts before document_ingestion_tool
)


# ── Run agent (first call) ──
def run_agent(input_data):
    """
    Start the agent with a file path or text content.
    If the agent hits an interrupt (before document_ingestion_tool),
    it pauses and returns the thread_id + pending tool call info.

    Args:
        input_data: File path (PDF/PNG/JPG/TXT) or plain text string

    Returns:
        dict with keys:
            - thread_id         (str)  always returned — needed to resume
            - interrupted       (bool) True if paused at interrupt
            - pending_tool_call (dict) what the agent wants to run next
            - agent_response    (str)  final response if not interrupted
    """
    print("➡️ Invoking agent...")

    # Build the message content
    if Path(input_data).exists() and Path(input_data).suffix in ['.pdf', '.png', '.jpg', '.jpeg']:
        message_content = f"Process a prior authorization request using this document: {input_data}"
    else:
        if Path(input_data).exists() and Path(input_data).suffix == '.txt':
            with open(input_data, 'r') as f:
                text_content = f.read()
        else:
            text_content = input_data

        message_content = f"""Process this prior authorization request:

{text_content}

Please extract the relevant entities, evaluate against policies, and provide your analysis."""

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # First invoke — will pause at ingest_node if document_ingestion_tool is called
    agent_executor.invoke(
        {"messages": [HumanMessage(content=message_content)]},
        config
    )

    # Check if we hit an interrupt
    snapshot = agent_executor.get_state(config)
    is_interrupted = bool(snapshot.next)

    # If interrupted, peek at what tool call is pending
    pending_tool_call = None
    if is_interrupted:
        last_msg = snapshot.values["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            tc = last_msg.tool_calls[0]
            pending_tool_call = {
                "name": tc["name"],
                "args": tc["args"],
                "id":   tc["id"]
            }
        print(f"⏸️  Interrupted before: {snapshot.next} | Pending: {pending_tool_call}")
        return {
            "thread_id":         thread_id,
            "interrupted":       True,
            "pending_tool_call": pending_tool_call,
            "agent_response":    None
        }

    # Not interrupted — agent ran to completion
    final_msg = snapshot.values["messages"][-1]
    return {
        "thread_id":         thread_id,
        "interrupted":       False,
        "pending_tool_call": None,
        "agent_response":    final_msg.content
    }


# ── Resume agent after human decision ──
def resume_agent(thread_id: str, decision: str, edited_args: dict = None):
    """
    Resume a paused agent after a human makes a decision.

    Args:
        thread_id:    The thread_id returned by run_agent()
        decision:     "approve" | "reject" | "edit"
        edited_args:  Required if decision == "edit". New args for the tool.
                      e.g. {"file_path": "new/path.pdf"}

    Returns:
        dict with keys:
            - thread_id      (str)
            - interrupted    (bool) True if paused again at another interrupt
            - agent_response (str)  final response when done
    """
    print(f"▶️  Resuming thread {thread_id} | Decision: {decision}")
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = agent_executor.get_state(config)

    if not snapshot.next:
        raise ValueError(f"Thread {thread_id} is not paused — nothing to resume.")

    if decision == "approve":
        # ── APPROVE: run the tool exactly as the LLM planned ──
        # For interrupt_before on a node, resume by invoking with None (no Command needed)
        result = agent_executor.invoke(None, config)

    elif decision == "reject":
        # ── REJECT: skip the tool, inject a ToolMessage with rejection feedback ──
        last_msg = snapshot.values["messages"][-1]
        tool_call_id = last_msg.tool_calls[0]["id"]

        # Inject a fake ToolMessage as if the tool ran but returned a rejection
        agent_executor.update_state(
            config,
            {"messages": [ToolMessage(
                content="Tool call rejected by human reviewer. Do not retry this tool. Proceed with available information.",
                tool_call_id=tool_call_id
            )]},
            as_node="ingest_node"  # pretend ingest_node ran and returned this
        )
        result = agent_executor.invoke(None, config)

    elif decision == "edit":
        # ── EDIT: modify the tool args before running ──
        if not edited_args:
            raise ValueError("edited_args must be provided when decision is 'edit'")

        last_msg = snapshot.values["messages"][-1]

        # Rebuild the tool calls list with updated args
        updated_tool_calls = []
        for tc in last_msg.tool_calls:
            if tc["name"] == "document_ingestion_tool":
                updated_tool_calls.append({**tc, "args": edited_args})
            else:
                updated_tool_calls.append(tc)

        # Patch the last AI message with new tool call args
        last_msg.tool_calls = updated_tool_calls
        agent_executor.update_state(
            config,
            {"messages": [last_msg]},
            as_node="agent"
        )
        result = agent_executor.invoke(None, config)

    else:
        raise ValueError(f"Invalid decision '{decision}'. Must be: approve | reject | edit")

    # Check if we hit another interrupt after resuming
    snapshot = agent_executor.get_state(config)
    is_interrupted = bool(snapshot.next)

    pending_tool_call = None
    if is_interrupted:
        last_msg = snapshot.values["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            tc = last_msg.tool_calls[0]
            pending_tool_call = {"name": tc["name"], "args": tc["args"], "id": tc["id"]}

    final_response = result["messages"][-1].content if not is_interrupted else None

    return {
        "thread_id":         thread_id,
        "interrupted":       is_interrupted,
        "pending_tool_call": pending_tool_call,
        "agent_response":    final_response
    }


# ── CLI test ──
if __name__ == "__main__":
    file_path = "data/input_data/document_03.pdf"

    print("\n" + "="*60)
    print("STEP 1: Starting agent...")
    print("="*60)
    result = run_agent(file_path)

    print(f"\n⏸️  Interrupted: {result['interrupted']}")
    print(f"📋 Pending: {result['pending_tool_call']}")

    if result['interrupted']:
        print("\nAgent wants to run:")
        print(f"  Tool: {result['pending_tool_call']['name']}")
        print(f"  Args: {result['pending_tool_call']['args']}")

        decision = input("\nDecision [approve / reject / edit]: ").strip().lower()
        edited_args = None

        if decision == "edit":
            new_path = input("Enter new file path: ").strip()
            edited_args = {"file_path": new_path}

        print("\n" + "="*60)
        print("STEP 2: Resuming agent...")
        print("="*60)
        final = resume_agent(result['thread_id'], decision, edited_args)

        print(f"\n⏸️  Interrupted again: {final['interrupted']}")
        print("\n✅ Agent output:")
        print(final['agent_response'])

    else:
        print("\n✅ Agent completed without interrupt:")
        print(result['agent_response'])






