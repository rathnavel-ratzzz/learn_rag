from langchain_core.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2", temperature=0.3)
history = ChatMessageHistory()

# Add system message once
history.add_message(SystemMessage(content="""You are a document assistant.
Answer only from what you know. Be concise."""))

print("=== Chat with Message History ===")
print("Type 'quit' to exit")
print("Type 'history' to see all messages\n")

while True:
    question = input("You: ")
    
    if question.lower() == "quit":
        break
    
    if question.lower() == "history":
        print("\n── Message History ──")
        for msg in history.messages:
            print(f"{msg.type:10} → {msg.content[:80]}")
        print()
        continue
    
    # Add user message
    history.add_user_message(question)
    
    # Send full history to LLM
    response = llm.invoke(history.messages)
    
    # Add AI response to history
    history.add_ai_message(response.content)
    
    print(f"AI: {response.content}\n")
    print(f"[History size: {len(history.messages)} messages]")