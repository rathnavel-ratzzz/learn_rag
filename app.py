"""
app.py — Flask backend for AskMyDoc UI

Run: python app.py
Then open: http://localhost:5000
"""

from flask import Flask, request, jsonify, send_from_directory
import chromadb
import ollama
import pypdf
import os
import shutil
from sentence_transformers import SentenceTransformer
import tiktoken

app = Flask(__name__, static_folder='.')
# ── Setup ──────────────────────────────────────────────────────────────────
embedder = SentenceTransformer("all-MiniLM-L6-v2")
DB_PATH = "./chroma_db_ui"
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(name="mydoc")
chat_history = []

UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
enc = tiktoken.get_encoding("cl100k_base")


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/upload', methods=['POST'])
def upload():
    """Receive files, chunk them, embed and store in ChromaDB."""
    try:
        files = request.files.getlist('files')
        if not files:
            return jsonify({'success': False, 'error': 'No files received'})

        # Clear old database
        global collection
        client.delete_collection("mydoc")
        collection = client.get_or_create_collection(name="mydoc")
        chat_history.clear()

        all_chunks = []

        for f in files:
            filename = f.filename
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            f.save(filepath)

            print(f"Processing: {filename}")
            text = extract_text(filepath, filename)

            if not text:
                continue

            chunks = generate_chunks(text)
            for chunk in chunks:
                all_chunks.append({'text': chunk, 'source': filename})

        if not all_chunks:
            return jsonify({'success': False, 'error': 'No text could be extracted'})

        # Filter junk
        junk_words = ["tutorialspoint", "copyright", "http", "www.", ".com", ".htm"]
        clean = [c for c in all_chunks if not any(w in c['text'].lower() for w in junk_words)]

        # Store in ChromaDB
        for i, chunk in enumerate(clean):
            vector = embedder.encode(chunk['text']).tolist()
            collection.add(
                ids=[str(i)],
                embeddings=[vector],
                documents=[chunk['text']],
                metadatas=[{'source': chunk['source']}]
            )

        print(f"Stored {len(clean)} chunks")
        return jsonify({'success': True, 'chunks': len(clean)})

    except Exception as e:
        print(f"Upload error: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/ask', methods=['POST'])
def ask():
    """Answer a question using RAG."""
    try:
        data = request.json
        question = data.get('question', '').strip()

        if not question:
            return jsonify({'answer': 'Please ask a question.'})

        # Rewrite query only for pronoun resolution (used for LLM, not search)
        resolved_question = build_search_query(question)

        # Search ChromaDB with the ORIGINAL question for consistent semantic search
        chunks, sources = search_chroma(question)

        if not chunks:
            return jsonify({'answer': "I couldn't find relevant information in your documents.", 'source': ''})

        context = "\n".join(chunks)
        unique_sources = list(set(sources))

        system_message = f"""You are a document assistant. Answer ONLY from the context below.
STRICT RULES:
- If the answer is not in the context at all, say "I don't know, this is not in the document"
- If partial information exists in the context, share what IS available
- Do NOT guess or use outside knowledge
- Do NOT make up information

Context:
{context}"""

        # Use the resolved question so the LLM understands pronoun references
        chat_history.append({'role': 'user', 'content': resolved_question})
        messages = [{'role': 'system', 'content': system_message}] + chat_history

        response = ollama.chat(model='llama3.2', messages=messages)
        reply = response['message']['content']

        chat_history.append({'role': 'assistant', 'content': reply})
        ctx_tokens    = count_tokens(context)
        q_tokens      = count_tokens(question)
        hist_tokens   = count_tokens(str(chat_history))
        total_tokens  = ctx_tokens + q_tokens + hist_tokens

        print(f"\n── Token Usage ──────────────────")
        print(f"  Context (chunks):  {ctx_tokens}")
        print(f"  Question:          {q_tokens}")
        print(f"  Chat history:      {hist_tokens}")
        print(f"  Total input:       {total_tokens}")
        print(f"  llama3.2 limit:    128000")
        print(f"  Used:              {round(total_tokens/128000*100, 1)}%")
        print(f"─────────────────────────────────\n")
        return jsonify({
            'answer': reply,
            'source': ', '.join(unique_sources)
        })

    except Exception as e:
        print(f"Ask error: {e}")
        return jsonify({'answer': f'Error: {str(e)}', 'source': ''})


@app.route('/clear', methods=['POST'])
def clear():
    """Clear the database and chat history."""
    global collection
    client.delete_collection("mydoc")
    collection = client.get_or_create_collection(name="mydoc")
    chat_history.clear()
    if os.path.exists(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)
        os.makedirs(UPLOAD_FOLDER)
    return jsonify({'success': True})


# ── Helper functions ───────────────────────────────────────────────────────

def extract_text(filepath, filename):
    ext = os.path.splitext(filename)[1].lower()
    try:
        if ext == '.pdf':
            text = ""
            reader = pypdf.PdfReader(filepath)
            for i, page in enumerate(reader.pages):
                if i == 0:
                    continue  # skip header page
                page_text = page.extract_text()
                if page_text:
                    page_text = page_text.replace("-\n", "").replace("\n", " ")
                    text += page_text + "\n\n"
            return text
        elif ext == '.txt':
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
    except Exception as e:
        print(f"Error reading {filename}: {e}")
    return None


def generate_chunks(text):
    chunks = []
    paragraphs = text.split("\n\n")
    for para in paragraphs:
        para = para.strip()
        if len(para) < 30:
            continue
        if len(para) < 500:
            chunks.append(para)
        else:
            sentences = para.split(".")
            i = 0
            while i < len(sentences):
                end = i + 2
                chunk = ".".join(sentences[i:end]).strip()
                if len(chunk) > 30:
                    chunks.append(chunk)
                i = end - 1
    return chunks


def search_chroma(question):
    vector = embedder.encode(question).tolist()
    results = collection.query(
        query_embeddings=[vector],
        n_results=5,
        include=["documents", "metadatas", "distances"]
    )
    THRESHOLD = 1.2
    docs, sources = [], []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        if dist <= THRESHOLD:
            docs.append(doc)
            sources.append(meta["source"])
    return docs, sources


def build_search_query(question):
    if len(chat_history) == 0:
        return question

    history_text = ""
    for msg in chat_history[-6:]:
        role = "User" if msg["role"] == "user" else "AI"
        history_text += f"{role}: {msg['content'][:100]}\n"

    response = ollama.chat(
        model="llama3.2",
        messages=[{
            "role": "user",
            "content": f"""Given this conversation history:
{history_text}

New question: "{question}"

Rewrite the question to be fully self-contained and searchable.
RULES:
- Replace pronouns like "it", "they", "that" with the actual topic
- Fix spelling mistakes
- Keep the meaning EXACTLY the same
- Do NOT rephrase or change vocabulary
- Return ONLY the rewritten question, nothing else."""
        }]
    )
    rewritten = response["message"]["content"].strip().strip('"').strip("'")
    print(f"Rewritten: '{rewritten}'")
    return rewritten

def count_tokens(text):
    """Count exact tokens using tiktoken"""
    return len(enc.encode(text))


# ── Run ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n🚀 AskMyDoc is running!")
    print("   Open: http://localhost:5000\n")
    app.run(debug=True, port=5000)