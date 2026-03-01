import ollama
import chromadb
from sentence_transformers import SentenceTransformer
# Our list of movies
embedder = SentenceTransformer("all-MiniLM-L6-v2")
movies = [
    "Inception - A mind-bending sci-fi thriller about dream infiltration",
    "Parasite - A dark Korean thriller about class inequality",
    "The Dark Knight - Batman faces the Joker in a psychological thriller",
    "Toy Story - Animated adventure about toys that come alive",
    "Gone Girl - A dark psychological mystery about a missing woman",
    "Tomb Raider - Action adventure following Lara Croft on a dangerous mission",
    "Mad Max Fury Road - Post apocalyptic action film with intense car chases",
]

# Open ChromaDB
print("Opening database...")
client = chromadb.PersistentClient(path="./chroma_db2")
collection = client.get_or_create_collection(name="movies")

# Store movies only if empty
if collection.count() == 0:
    print("Storing movies...")
    for i, movie in enumerate(movies):
        embeddings_ =embedder.encode(movie).tolist()
        collection.add(
            ids=[str(i)],
            embeddings=[embeddings_],
            documents=[movie]
        )
    print(f"Stored {collection.count()} movies!")
else:
    print(f"Database already has {collection.count()} movies!")

print()

# Function to search ChromaDB for relevant movies
def search_movies(question):
    results = collection.query(
        query_texts=[question],
        n_results=3
    )
    return results["documents"][0]

# Function to get recommendation from AI
def get_recommendation(question):
    # Step 1 — find relevant movies from database
    relevant_movies = search_movies(question)

    print("Most relevant movies found:")
    for movie in relevant_movies:
        print(f"  - {movie}")
    print()

    # Step 2 — send only relevant movies to AI
    movie_list = "\n".join(relevant_movies)

    prompt = f"""Here are some relevant movies:

                {movie_list}

                The user wants: {question}
                IMPORTANT: You MUST only recommend a movie from the list above.
                Do NOT suggest any other movies.    
                Recommend the best one and explain why in 2 sentences."""   

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]

# Loop
print("=== Movie Recommender ===")
print("Type 'quit' to exit\n")

while True:
    question = input("What kind of movie are you in the mood for? ")

    if question.lower() == "quit":
        print("Goodbye!")
        break

    print("\nSearching database...")
    answer = get_recommendation(question)
    print("AI recommends:")
    print(answer)
    print()