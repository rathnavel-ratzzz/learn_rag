
import chromadb
import  ollama
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")

file = open("sample.txt")
text=file.read()
file.close()

chunks=[]
# chunk_size = 200
# chunk_overlap =50

# start = 0

# while start < len(text):
#     end=start + chunk_size
#     chunk=text[start:end].strip()
#     if chunk:
#         chunks.append(chunk) 
#     start=end- chunk_overlap
        


### split by sentence
sentence =text.split(".")
i = 0
sentence_chunk=2
sentence_chunk_overlap=1
while i< len(sentence):
    end=i+sentence_chunk
    chunk=sentence[i:end]
    chunk_=".".join(chunk).strip()
    if chunk_:
        chunks.append(chunk_)
    i=end-sentence_chunk_overlap



    
client =chromadb.PersistentClient(path="chroma_db3")
collections=client.get_or_create_collection(name="mydoc")

if collections.count()==0:
    for i,chunk in enumerate(chunks):
        embeddings=embedder.encode(chunk).tolist()
        collections.add(ids=[str(i)],embeddings=[embeddings],documents=[chunk])
    print("stored succesfully")

def search_chroma(question):
    embedd_question=embedder.encode(question).tolist()
    answer=collections.query(query_embeddings=[embedd_question],n_results=3)
    return  answer["documents"][0]

def query_llm(question):
    chroma_search=search_chroma(question)
    prompt=f"""The user wants to query few details based on the given context.Kindly reply only to the relavant content
    the users question : {question}
    the context from semantic search for preloaded documents: {chroma_search}
    if you cannot answer. just reply "out of context"
    """
    chat =ollama.chat(model="llama3.2",messages=[{"role":"user","content":prompt}])
    return chat["message"]["content"]

while True:
    user=input("ask some question about Einstien")
    if user.lower()=="quit":
        break
    answer=query_llm(user)
    print(answer)
    
    