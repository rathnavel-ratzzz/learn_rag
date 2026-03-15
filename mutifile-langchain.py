from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from pathlib import Path
from langchain.agents import create_agent
# Read Docs --> chunks using the chuking startegy -> embedding using huggig face -> vector Db ->reteriver -> ollama model to give answer based on reteriver

directory_path = './docs'

all_chunks=[]
for file_path in Path(directory_path).iterdir():
    if file_path.is_file():
        if file_path.name.lower().split(".")[1]=="pdf":
            pdf_=PyPDFLoader(file_path)
            docs=pdf_.load()
            splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=150)
            all_chunks.extend(splitter.split_documents(docs))  

        elif file_path.name.lower().split(".")[1]=="txt":
            with open(file_path) as file:
                text=file.read()
            splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=150)
            chunks = splitter.split_text(text)
            all_chunks.extend([Document(page_content=c) for c in chunks])  


embedder=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
vectorstore=Chroma.from_documents(documents=all_chunks,embedding=embedder,persist_directory="./multi-chroma")
reteriver=vectorstore.as_retriever(search_kwargs={"k":3})
#question and context
promptTemplate_="""You are good at explaining the question based on provided context. Make sure you only provide knowledge about the known things. If it is out of context I can't provide info on that
example 1: context: 1. sun is yellow,
                     2. big star in our solar systems
                     3.not yellow in color it is white
            question: what is the biggest star in solar system.
            answer: the big star is suin in our...

example 2: context: 1. sun is yellow,
                     2. big star in our solar systems
                     3.not yellow in color it is white
            question: what is the 2nd big star..
            answer: i dont have info on that
            
            
{previous_context}
{context}

Please answer the below question:
{question}

"""  # fix 3: plain string (not f-string) so {context}/{question} are template variables
prompt=PromptTemplate(template=promptTemplate_,input_variables=["context","question","previous_context"])

llm = OllamaLLM(model="llama3.2")  # fix 4: model= keyword arg
previous_chat=""
while True:
    question = input("You: ")
    if question.lower() == "quit":
        break
    chain=( {"context":reteriver , "question":RunnablePassthrough(),"previous_context":lambda _: previous_chat} | prompt | llm | StrOutputParser() )  
    answer = chain.invoke(question)
    previous_chat += f"question: {question}\n"
    previous_chat += f"answer: {answer}\n"
    print(answer)
    # Correct - previous_chat is a fixed string, use lambda
