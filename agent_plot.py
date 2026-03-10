from agent import agent_executor

graph_image = agent_executor.get_graph().draw_mermaid_png()
with open("result/graph.png", "wb") as f:
    f.write(graph_image)

print("✅ Saved to result/graph.png")