from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from typing import TypedDict, Annotated
import operator
import wikipedia

llm = ChatOllama(model="llama3.2", temperature=0.3)

# ── Define Tools ───────────────────────────────────────
@tool
def search_wikipedia(query: str) -> str:
    """Search wikipedia for information about a topic"""
    try:
        return wikipedia.summary(query, sentences=8)
    except:
        return f"No results found for {query}"

@tool
def calculator(expression: str) -> str:
    """Calculate a math expression like '100 * 2'"""
    try:
        return str(eval(expression))
    except:
        return "Could not calculate"

tools = [search_wikipedia, calculator]

# ── Bind tools to LLM ─────────────────────────────────
# This tells LLM it CAN use these tools
llm_with_tools = llm.bind_tools(tools)

# ── Define State ───────────────────────────────────────
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]

# ── Define Nodes ───────────────────────────────────────
def llm_node(state: AgentState):
    """LLM thinks and decides whether to use a tool"""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# ToolNode automatically runs whatever tool LLM decided to use
tool_node = ToolNode(tools)

# ── Decision Function ──────────────────────────────────
def should_use_tool(state: AgentState) -> str:
    """Check if LLM wants to use a tool or is done"""
    last_message = state["messages"][-1]
    
    # If LLM called a tool → go to tool node
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print(f"\n🔧 Using tool: {last_message.tool_calls[0]['name']}")
        return "use_tool"
    
    # Otherwise → done
    print("\n✅ Agent finished!")
    return "end"

# ── Build Graph ────────────────────────────────────────
graph = StateGraph(AgentState)

graph.add_node("llm", llm_node)
graph.add_node("tools", tool_node)

graph.set_entry_point("llm")

# Conditional — did LLM want a tool?
graph.add_conditional_edges(
    "llm",
    should_use_tool,
    {
        "use_tool": "tools",  # yes → run tool
        "end": END            # no → finish
    }
)

# After tool runs → go back to LLM
graph.add_edge("tools", "llm")

app = graph.compile()

# ── Run ────────────────────────────────────────────────
from langchain_core.messages import HumanMessage

print("=== LangGraph Agent with Tools ===\n")

while True:
    question = input("You: ")
    if question.lower() == "quit":
        break
    
    result = app.invoke({
        "messages": [HumanMessage(content=question)]
    })
    
    print(f"\nAI: {result['messages'][-1].content}\n")
