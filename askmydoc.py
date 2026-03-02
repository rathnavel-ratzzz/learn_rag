
import chromadb
import  ollama
from sentence_transformers import SentenceTransformer
import pypdf
import os

embedder = SentenceTransformer("all-MiniLM-L6-v2")
client =chromadb.PersistentClient(path="chroma_db6")
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
    chunks = []
    
    # Split by paragraph first (double newline)
    paragraphs = text.split("\n\n")
    
    for para in paragraphs:
        para = para.strip()
        
        # Skip empty or very short paragraphs
        if len(para) < 30:
            continue
        
        # If paragraph is short enough, keep it as one chunk
        if len(para) < 500:
            chunks.append(para)
        
        # If paragraph is too long, split by sentence
        else:
            sentences = para.split(".")
            i = 0
            while i < len(sentences):
                end = i + 2
                chunk = ".".join(sentences[i:end]).strip()
                if len(chunk) > 30:
                    chunks.append(chunk)
                i = end - 1
    
    print(f"Created {len(chunks)} chunks")
    return chunks

def store_chunks(chunks):
    if collections.count() == 0:
        for i, chunk in enumerate(chunks):
            embeddings = embedder.encode(chunk["text"]).tolist()
            collections.add(
                ids=[str(i)],
                embeddings=[embeddings],
                documents=[chunk["text"]],
                metadatas=[{"source": chunk["source"]}]
            )
    else:
        print(f"DB loaded with {collections.count()} chunks")

def search_chroma(question):
    embedd_question = embedder.encode(question).tolist()
    answer = collections.query(
        query_embeddings=[embedd_question],
        n_results=5,
        include=["documents", "metadatas", "distances"]
    )

    # Filter out chunks that are too dissimilar (distance > threshold = irrelevant)
    DISTANCE_THRESHOLD = 1.2
    docs, sources = [], []
    for doc, meta, dist in zip(answer["documents"][0], answer["metadatas"][0], answer["distances"][0]):
        if dist <= DISTANCE_THRESHOLD:
            docs.append(doc)
            sources.append(meta["source"])

    print(f"\nChunks found: {len(docs)} relevant (filtered from 5)")
    return docs, sources

def query_llm(question):
    # No more chat history logic here - build_search_query handles it
    chroma_search, source = search_chroma(question)
    unique_source = list(set(source))
    
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

    return reply + f"\n\n📄 Source: {', '.join(unique_source)}"

def load_chunks(dir):
    all_chunks=[]
    supported =[".pdf",".txt"]
    files = os.listdir(dir)
    for file in files:
        ext=os.path.splitext(file)[1].lower()
        if ext not in supported:
            continue
        filepath=os.path.join(dir,file)
        print("laoding",filepath)    
        contents = readDocumnets(filepath)
        
        if not contents:
            print(f"  Skipping {file} - could not read")
            continue
        chunks = GenerateChunks(contents)
        tagged_chunks = []
        for chunk in chunks:
            tagged_chunks.append({
                "text": chunk,
                "source": file
            })
        all_chunks.extend(tagged_chunks)
    print(f"\nTotal chunks from all files: {len(all_chunks)}")
    return all_chunks
        
def build_search_query(question):
    if len(chat_history) == 0:
        return question

    # Only rewrite if the question references prior context via pronouns
    reference_words = {"it", "its", "they", "them", "their", "that", "this", "those", "these", "he", "she"}
    if not any(w in reference_words for w in question.lower().split()):
        return question

    # Build history, stripping source citations so they don't confuse the rewriter
    history_text = ""
    for msg in chat_history[-4:]:
        role = "User" if msg["role"] == "user" else "AI"
        content = msg["content"].split("\n\n")[0][:150]
        history_text += f"{role}: {content}\n"

    response = ollama.chat(
        model="llama3.2",
        messages=[{
            "role": "user",
            "content": f"""Given this conversation history:
{history_text}

New question: "{question}"

Rewrite the question to be fully self-contained and searchable.
RULES:
- ONLY replace pronouns like "it", "they", "that" with the actual topic from the history
- Fix spelling mistakes only
- Keep the meaning and topic EXACTLY the same — do NOT change the subject
- Do NOT rephrase or add new information
- Return ONLY the rewritten question, nothing else."""
        }]
    )

    rewritten = response["message"]["content"].strip()
    print(f"Rewritten query: '{rewritten}'")
    return rewritten
def main():
    if collections.count() == 0:
        contents=load_chunks("docs")
        store_chunks(contents)
        print("Chunks in database:") 
    else:
        print("DB loaded")
    while True:
        user = input("\nAsk a question (or 'quit'): ")
        if user.lower() == "quit":
            return
        question = build_search_query(user)
        answer = query_llm(question)
        print(answer)
    
main()