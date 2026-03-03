import ollama

# Same question, same context, different temperatures
context = """
AVL trees are self-balancing binary search trees.
They were invented by Georgy Adelson-Velsky and Evgenii Landis in 1962.
AVL trees use rotation to maintain balance after insertions and deletions.
The height difference between left and right subtrees cannot exceed 1.
"""

question = "Explain AVL trees"

temperatures = [0.0, 0.5, 1.0, 1.5]

for temp in temperatures:
    print(f"\n{'='*50}")
    print(f"Temperature: {temp}")
    print('='*50)
    
    response = ollama.chat(
        model="llama3.2",
        options={"temperature": temp},
        messages=[{
            "role": "user",
            "content": f"""Answer based on context below.
            
Context: {context}

Question: {question}"""
        }]
    )
    
    print(response["message"]["content"])