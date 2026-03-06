from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

# Create history
history = ChatMessageHistory()

# Add conversations
history.c("what is first generation?")
history.add_ai_message("First gen used vacuum tubes")

history.add_user_message("what is second generation?")
history.add_ai_message("Second gen used transistors")

history.add_user_message("what languages did it use?")
history.add_ai_message("FORTRAN and COBOL")

# Check what's stored
print("All messages:")
for msg in history.messages:
    print(f"{msg.type}: {msg.content}")

print(f"\nTotal messages: {len(history.messages)}")

# Clear old messages (window behavior)
MAX_MESSAGES = 4  # keep last 4 (2 Q&A pairs)
if len(history.messages) > MAX_MESSAGES:
    history.messages = history.messages[-MAX_MESSAGES:]
    print(f"\nAfter trimming to last {MAX_MESSAGES}:")
    for msg in history.messages:
        print(f"{msg.type}: {msg.content}")