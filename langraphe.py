from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from typing import TypedDict

llm = OllamaLLM(model="llama3.2")
parser=StrOutputParser()

class ResearchState(TypedDict):
    topic : str
    research :str
    analysis :str
    report : str
    
def researcher_node(state:ResearchState) -> ResearchState:
    print("\n🔍 Researcher Agent working...")
    
    try:
        search_results = wikipedia.summary(state["topic"], sentences=10)
    except:
        search_results = f"No results found for {state['topic']}"
    
    prompt = ChatPromptTemplate.from_template("""
You are a researcher. Summarize these search results as bullet points.
Topic: {topic}
Results: {results}
""")
    chain = prompt | llm | parser
    research = chain.invoke({
        "topic": state["topic"],
        "results": search_results
    })
    
    print("✅ Research done!")
    # Write to shared state
    return {"research": research}

def analyst_node(state: ResearchState) -> ResearchState:
    print("\n🧠 Analyst Agent working...")
    
    prompt = ChatPromptTemplate.from_template("""
You are an analyst. Analyze this research and extract key insights.
Research: {research}
Provide: key insights, trends, critical facts.
""")
    chain = prompt | llm | parser
    analysis = chain.invoke({"research": state["research"]})
    
    print("✅ Analysis done!")
    return {"analysis": analysis}

def writer_node(state: ResearchState) -> ResearchState:
    print("\n✍️  Writer Agent working...")
    
    prompt = ChatPromptTemplate.from_template("""
You are a writer. Write a professional report.
Topic: {topic}
Analysis: {analysis}

Write with:
1. Executive Summary
2. Key Findings  
3. Conclusion
""")
    chain = prompt | llm | parser
    report = chain.invoke({
        "topic": state["topic"],
        "analysis": state["analysis"]
    })
    
    print("✅ Report done!")
    return {"report": report}

graph = StateGraph(ResearchState)

graph.add_node("researcher", researcher_node)
graph.add_node("analyst", analyst_node)
graph.add_node("writer", writer_node)

# Add edges — defines the flow
graph.set_entry_point("researcher")        
graph.add_edge("researcher", "analyst")   
graph.add_edge("analyst", "writer")        
graph.add_edge("writer", END)             

app = graph.compile()

while True:
    topic = input("Research topic (or 'quit'): ")
    if topic.lower() == "quit":
        break

    # Run the graph with initial state
    result = app.invoke({"topic": topic, "research": "", "analysis": "", "report": ""})

    print(f"\n{'='*50}")
    print("FINAL REPORT:")
    print('='*50)
    print(result["report"])

    # Save to file
    filename = f"report_{topic[:20].replace(' ', '_')}.txt"
    with open(filename, "w") as f:
        f.write(result["report"])
    print(f"\n📄 Saved to {filename}")