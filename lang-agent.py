from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import PromptTemplate
from langchain.agents import create_agent



# ── Setup RAG ──────────────────────────────────────────
loader = PyPDFLoader("docs/comgen.pdf")
docs = loader.load()
docs = [doc for doc in docs if doc.metadata["page"] != 0]

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    embedding=embedder,
    documents=chunks,
    persist_directory="./chroma_agent"
)

# ── Define Tools ───────────────────────────────────────
@tool
def doc_search(query: str) -> str:
    """Search the uploaded documents for relevant information."""
    results = vectorstore.similarity_search(query, k=3)
    if not results:
        return "Nothing found in documents"
    return "\n\n".join(doc.page_content for doc in results)

@tool
def calculator(expression: str) -> str:
    """Perform math calculations. Input must be a valid math expression like '100 * 2'"""
    try:
        result = eval(expression)
        return str(result)
    except:
        return "Could not calculate that"

search = DuckDuckGoSearchRun()

tools = [doc_search, calculator, search]

# ── Define Agent Prompt ────────────────────────────────
prompt = PromptTemplate.from_template("""You are a helpful assistant with access to tools.

You have access to the following tools:
{tools}

Use this format:
Question: the input question
Thought: think about what to do
Action: tool name (one of [{tool_names}])
Action Input: input to the tool
Observation: result of the tool
... (repeat Thought/Action/Observation as needed)
Thought: I now know the final answer
Final Answer: your answer here

Begin!

Question: {input}
Thought: {agent_scratchpad}""")

# ── Create Agent ───────────────────────────────────────
llm = ChatOllama(model="llama3.2", temperature=0.3)


# ── Run ────────────────────────────────────────────────
print("\n=== Ask My Doc Agent ===")
print("Type 'quit' to exit\n")
agent = create_agent(tools=tools, model=llm, system_prompt=""""You are a document assistant.
STRICT RULES:
- ALWAYS use doc_search tool first for any question
- ALWAYS use web_search tool if doc_search returns nothing
- ALWAYS use calculator for any math
- NEVER answer from memory""")

while True:
    question = input("You: ")
    if question.lower() == "quit":
        break
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    print(f"\nAI: {result['messages']}\n")
