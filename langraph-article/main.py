from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph,END
from typing import TypedDict
llm=ChatOllama(model="llama3.2", temperature=0.3)
llm.bind_tools([DuckDuckGoSearchRun])
persistent_directory="./chroma_langchain"
chat_history = ChatMessageHistory()
search=DuckDuckGoSearchRun()
SYSTEM_PROMPT = """You are a Extraordinary article knowledgeable assistant. You are capable of give the  key information from the article content provided to you """

embedder=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
retriever=None
store =None

class ArticleState(TypedDict):
    query:str
    cache: list
    db_result :str
    results:str
    web_result :str
    answer :str
    
def SearchDbNode(state:ArticleState)->ArticleState:
    
    if retriever is None:
        return {"db_result": None}
    results = retriever.invoke(state["query"])
    if not results:
        return {"db_result":None}
    
    return {"db_result":results}
    
def searchWeb(state: ArticleState) -> ArticleState:
    global store, retriever
    try:
        ;results = search.invoke(f"site:medium.com {state['query']}")
    except Exception:
        results = "Web search unavailable"
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.create_documents([results])
    
    if store is None:
        store = Chroma.from_documents(
            embedding=embedder,
            documents=chunks,
            persist_directory=persistent_directory
        )
    else:
        store.add_documents(chunks)
    
    retriever = store.as_retriever(search_kwargs={"k": 3})
    return {"web_result": results}  # ← always returns dict!
def AnswerNode(state:ArticleState)->ArticleState:
    prompt="""The context for the query is given. Now answer the query based on this {context} 
     query:{query}"""
    template=PromptTemplate(template=prompt,input_variables=["context","query"])
    chain=(template|llm|StrOutputParser())
    if state["web_result"] is None:
        return  {"answer" :chain.invoke({"context":state["db_result"],"query":state["query"]})}
    else:
        return {"answer" :chain.invoke({"context":state["web_result"],"query":state["query"]})}
    

def decision(state: ArticleState)->ArticleState:
    if state["db_result"] is None or state["db_result"] == "":
        return "web"
    else:
        return "answer"
    
graph=StateGraph(ArticleState)
graph.add_node("db",SearchDbNode)
graph.add_node("web",searchWeb)
graph.add_node("answer",AnswerNode)

graph.set_entry_point("db")
graph.add_conditional_edges("db",decision,
                            {"web":"web",
                             "answer":"answer"})
graph.add_edge("web","answer")
graph.add_edge("answer",END)
app = graph.compile()
def main():
    global store, retriever
    print("1.Load the article.")
    print("2.Query on article.")
    store = None
    while True:
        choice=input("Enter your choice (1-3, or 'quit' to exit): ")
        if choice=="1":
            input_path=input("Enter the path to the article (PDF): ")
            loader=PyPDFLoader(input_path)
            docs=loader.load()
            splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
            chunks=splitter.split_documents(docs)
            store=Chroma.from_documents(embedding=embedder,documents=chunks,persist_directory=persistent_directory)
            retriever=store.as_retriever(search_kwargs={"k":3})
            print("stored...")
        if choice == "2":
            input_qeury=input("query about the articles")
            prompt=PromptTemplate(template="You are good fixing the typo and making the query easy to understand by llm. The query is {query}",input_variables=["query"])
            chain = ( prompt | llm | StrOutputParser() )
            result= chain.invoke({"query":input_qeury})
            answer = app.invoke({"query": result, "db_result": "", "web_result": "", "answer": "", "cache": [], "results": ""})
            print(answer["answer"])
            
            
        
    
if __name__ == '__main__':
    main()