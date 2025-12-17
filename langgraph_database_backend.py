from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import sqlite3
import os
import google.generativeai as genai

load_dotenv()

# Verify API key is loaded
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

# Configure Gemini
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.5-flash')

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    
    # Convert messages to text for Gemini
    prompt_parts = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            prompt_parts.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            prompt_parts.append(f"Assistant: {msg.content}")
    
    prompt = "\n".join(prompt_parts)
    
    # Get response from Gemini (streaming handled in frontend)
    response = model.generate_content(prompt)
    
    # Return as AIMessage
    return {"messages": [AIMessage(content=response.text)]}

def get_streaming_response(messages):
    """Generate streaming response for given messages"""
    # Convert messages to text for Gemini
    prompt_parts = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            prompt_parts.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            prompt_parts.append(f"Assistant: {msg.content}")
    
    prompt = "\n".join(prompt_parts)
    
    # Stream response from Gemini
    response = model.generate_content(prompt, stream=True)
    
    for chunk in response:
        if chunk.text:
            yield chunk.text

conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)

# Create table for conversation names
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS conversation_names (
        thread_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
''')
conn.commit()

# Checkpointer
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

def generate_conversation_name(first_message):
    """Generate a short conversation name from the first user message"""
    # Use a simple approach: take first 40 characters or first sentence
    import re
    
    # Remove extra whitespace
    cleaned = ' '.join(first_message.split())
    
    # Try to get first sentence
    sentences = re.split(r'[.!?]', cleaned)
    if sentences and sentences[0]:
        name = sentences[0].strip()
    else:
        name = cleaned
    
    # Limit to 40 characters
    if len(name) > 40:
        name = name[:37] + '...'
    
    return name if name else 'New Conversation'

def save_conversation_name(thread_id, name):
    """Save or update conversation name for a thread"""
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO conversation_names (thread_id, name)
        VALUES (?, ?)
    ''', (str(thread_id), name))
    conn.commit()

def get_conversation_name(thread_id):
    """Get conversation name for a thread"""
    cursor = conn.cursor()
    cursor.execute('SELECT name FROM conversation_names WHERE thread_id = ?', (str(thread_id),))
    result = cursor.fetchone()
    return result[0] if result else None

def delete_conversation(thread_id):
    """Delete a conversation and its name from the database"""
    cursor = conn.cursor()
    
    # Delete from conversation_names table
    cursor.execute('DELETE FROM conversation_names WHERE thread_id = ?', (str(thread_id),))
    
    # Delete from checkpoints table (LangGraph storage)
    cursor.execute('DELETE FROM checkpoints WHERE thread_id = ?', (str(thread_id),))
    
    conn.commit()
    return True

def retrieve_all_threads():
    """Retrieve all threads with their names"""
    all_threads = {}
    for checkpoint in checkpointer.list(None):
        thread_id = checkpoint.config['configurable']['thread_id']
        name = get_conversation_name(thread_id)
        all_threads[thread_id] = name
    
    return all_threads

