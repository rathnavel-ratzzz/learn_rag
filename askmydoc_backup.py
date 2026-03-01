
import chromadb
import  ollama
from sentence_transformers import SentenceTransformer
import pypdf

embedder = SentenceTransformer("all-MiniLM-L6-v2")
client =chromadb.PersistentClient(path="chroma_db4")
collections=client.get_or_create_collection(name="mydoc")

chat_history = []
def readTextFile(file):
    file = open(file)
    text=file.read()
    file.close()
    return text
def readPDFFile(file):
    text = ""
    reader = pypdf.PdfReader(file)
    print(f"PDF has {len(reader.pages)} pages")
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text += page_text
            print(f"  Read page {i+1}")
    return text

def readDocumnets(filename):
    splits=filename.split(".")
    if splits[1] == "pdf":
        return readPDFFile(filename)
    elif splits[1] == "txt":
        return readTextFile(filename)
    else:
        print(f"Unsupported file type")
        return None
    
    
def GenerateChunks(text):
    chunks=[]
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
    return chunks

def store_chunks(chunks):    
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
    # Build search query using previous question if available
    if len(chat_history) >= 2:
        search_query = chat_history[-2]["content"] + " " + question
    else:
        search_query = question

    chroma_search = search_chroma(search_query)

    if not chroma_search:
        return "I don't know, this is not in the document"

    context = "\n".join(chroma_search)

    system_message = f"""You are a document assistant. Answer ONLY from the context below.
STRICT RULES:
- If the answer is not clearly in the context, say "I don't know, this is not in the document"
- Do NOT guess or use outside knowledge
- Do NOT make up information

Context:
{context}"""

    chat_history.append({"role": "user", "content": question})

    messages = [{"role": "system", "content": system_message}] + chat_history

    chat = ollama.chat(model="llama3.2", messages=messages)
    reply = chat["message"]["content"]

    chat_history.append({"role": "assistant", "content": reply})

    return reply

def main():
    file="comgen.pdf"
    contents=readDocumnets(file)
    chunks=GenerateChunks(contents)
    store_chunks(chunks)
    while True:
        user=input(f"Ask some Questions about {file}")
        if user.lower() =="quit":
            return
       
        answer=query_llm(user)
        print(answer)
    
main()