from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(model="llama3.2", temperature=0.3)
search = DuckDuckGoSearchRun()
parser = StrOutputParser()

# ── Agent 1: Researcher ────────────────────────────────
researcher_prompt = ChatPromptTemplate.from_template("""
You are a research agent. Your job is to search and collect information.

Search query: {input}

Search the web and collect all relevant information.
Present findings as clear bullet points.
Include key facts, statistics, and recent developments.
""")

researcher_chain = researcher_prompt | llm | parser

def researcher_agent(query):
    print("\n🔍 Researcher Agent working...")
    # Search web for info
    try:
        search_results = search.invoke(query)
    except Exception:
        search_results = "Web search unavailable."
    # Analyze search results
        search_results = sear
    findings = researcher_chain.invoke({
        "input": f"Query: {query}\n\nSearch Results:\n{search_results}"
    })
    print("✅ Research complete!")
    return findings

# ── Agent 2: Analyst ───────────────────────────────────
analyst_prompt = ChatPromptTemplate.from_template("""
You are an analyst agent. Your job is to analyze research findings.

Research findings:
{input}

Analyze the above research and provide:
1. Key insights
2. Important trends
3. Critical facts
4. Remove any irrelevant information
Present as structured analysis.
""")

analyst_chain = analyst_prompt | llm | parser

def analyst_agent(research):
    print("\n🧠 Analyst Agent working...")
    analysis = analyst_chain.invoke({"input": research})
    print("✅ Analysis complete!")
    return analysis

# ── Agent 3: Writer ────────────────────────────────────
writer_prompt = ChatPromptTemplate.from_template("""
You are a writer agent. Your job is to write clear professional reports.

Analysis to write report from:
{input}

Original topic: {topic}

Write a professional report with:
1. Executive Summary
2. Key Findings
3. Trends and Insights
4. Conclusion
Make it clear, concise and professional.
""")

writer_chain = writer_prompt | llm | parser

def writer_agent(analysis, topic):
    print("\n✍️  Writer Agent working...")
    report = writer_chain.invoke({
        "input": analysis,
        "topic": topic
    })
    print("✅ Report complete!")
    return report

# ── Orchestrator ───────────────────────────────────────
def research_pipeline(topic):
    print(f"\n{'='*50}")
    print(f"Research Topic: {topic}")
    print(f"{'='*50}")

    # Step 1 — Research
    research = researcher_agent(topic)
    
    # Step 2 — Analyze
    analysis = analyst_agent(research)
    
    # Step 3 — Write
    report = writer_agent(analysis, topic)

    # Save report to file
    filename = f"report_{topic[:20].replace(' ', '_')}.txt"
    with open(filename, "w") as f:
        f.write(f"RESEARCH REPORT: {topic}\n")
        f.write("="*50 + "\n\n")
        f.write(report)
    
    print(f"\n📄 Report saved to: {filename}")
    return report

# ── Main ───────────────────────────────────────────────
print("=== Multi-Agent Research Tool ===\n")

while True:
    topic = input("Research topic (or 'quit'): ")
    if topic.lower() == "quit":
        break
    report = research_pipeline(topic)
    print(f"\n{'='*50}")
    print("FINAL REPORT:")
    print('='*50)
    print(report)
   2. Yes, and don't ask again 