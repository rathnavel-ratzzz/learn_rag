from langchain_community.document_loaders import PyPDFLoader
# New way
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser



loader=PyPDFLoader("docs/comgen.pdf")
docs=loader.load()
doc = [doc for doc in docs if doc.metadata["page"] != 0]
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # characters per chunk
    chunk_overlap=50,     # overlap between chunks
)

chunks = splitter.split_documents(doc)
embeder=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore=Chroma.from_documents(embedding=embeder,documents=chunks,persist_directory="./chroma_langchain")



# Step 5 — Connect Ollama
print("Connecting to Ollama...")
llm = OllamaLLM(model="llama3.2")

# Step 6 — Build prompt template
prompt_template = """You are a document assistant.
Answer ONLY from the context below.
If not in context say "I don't know".

Context:
{context}

Question: {question}
Answer:"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("\n=== LangChain Ask My Doc ===")
print("Type 'quit' to exit\n")

while True:
    question = input("You: ")
    if question.lower() == "quit":
        break
    answer = chain.invoke(question)
    print(f"AI: {answer}\n")
