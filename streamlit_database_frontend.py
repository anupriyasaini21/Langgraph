import streamlit as st
import sys
import os

# Add the current directory to Python path to import the backend module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langgraph_database_backend import (
    chatbot, 
    retrieve_all_threads, 
    generate_conversation_name, 
    save_conversation_name,
    get_conversation_name,
    get_streaming_response,
    delete_conversation
)
from langchain_core.messages import HumanMessage, AIMessage
import uuid

# **************************************** utility functions *************************

def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    st.session_state['message_history'] = []
    st.session_state['is_new_conversation'] = True

def add_thread(thread_id, name=None):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'][thread_id] = name

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    # Check if messages key exists in state values, return empty list if not
    return state.values.get('messages', [])


# **************************************** Session Setup ******************************
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

if 'is_new_conversation' not in st.session_state:
    st.session_state['is_new_conversation'] = True

if st.session_state['thread_id'] not in st.session_state['chat_threads']:
    add_thread(st.session_state['thread_id'])


# **************************************** Sidebar UI *********************************

st.sidebar.title('LangGraph Chatbot')

if st.sidebar.button('➕ New Chat', use_container_width=True):
    reset_chat()
    st.rerun()

st.sidebar.header('My Conversations')

# Sort threads by most recent (assuming reverse order)
threads_list = list(st.session_state['chat_threads'].items())
for thread_id, conv_name in reversed(threads_list):
    # Display name or 'New Chat' if no name yet
    display_name = conv_name if conv_name else "New Chat"
    
    # Create columns for chat name and delete button
    col1, col2 = st.sidebar.columns([5, 2])
    
    with col1:
        # Highlight current conversation
        if thread_id == st.session_state['thread_id']:
            button_label = f"▶ {display_name}"
            button_type = "primary"
        else:
            button_label = display_name
            button_type = "secondary"
        
        if st.button(button_label, key=str(thread_id), use_container_width=True, type=button_type):
            st.session_state['thread_id'] = thread_id
            st.session_state['is_new_conversation'] = False
            messages = load_conversation(thread_id)

            temp_messages = []

            for msg in messages:
                if isinstance(msg, HumanMessage):
                    role='user'
                else:
                    role='assistant'
                temp_messages.append({'role': role, 'content': msg.content})

            st.session_state['message_history'] = temp_messages
            st.rerun()
    
    with col2:
        # Delete button for each conversation
        if st.button("✕", key=f"del_{thread_id}", help="Delete this conversation"):
            # Delete the conversation
            delete_conversation(thread_id)
            del st.session_state['chat_threads'][thread_id]
            
            # If deleting current conversation, switch to a new one
            if thread_id == st.session_state['thread_id']:
                reset_chat()
            
            st.rerun()


# **************************************** Main UI ************************************

# loading the conversation history
if not st.session_state['message_history']:
    # Display welcome message for new chat
    st.markdown("""
    <div style='text-align: center; padding: 60px 20px;'>
        <h1 style='color: #1f1f1f; font-size: 2.5rem; margin-bottom: 20px;'>
            👋 Hello! How can I help you today?
        </h1>
        <p style='color: #666; font-size: 1.1rem; margin-bottom: 30px;'>
            I'm your AI assistant powered by Gemini. Ask me anything!
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    for message in st.session_state['message_history']:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

user_input = st.chat_input('Type here')

if user_input:

    # first add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)

    # Generate conversation name for new conversations after first message
    if st.session_state['is_new_conversation']:
        conv_name = generate_conversation_name(user_input)
        save_conversation_name(st.session_state['thread_id'], conv_name)
        st.session_state['chat_threads'][st.session_state['thread_id']] = conv_name
        st.session_state['is_new_conversation'] = False

    #CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {
            "thread_id": st.session_state["thread_id"]
        },
        "run_name": "chat_turn",
    }

    # Generate response with streaming
    with st.chat_message('assistant'):
        message_placeholder = st.empty()
        full_response = ""
        
        # Get all messages for context
        all_messages = []
        for msg in st.session_state['message_history']:
            if msg['role'] == 'user':
                all_messages.append(HumanMessage(content=msg['content']))
            else:
                all_messages.append(AIMessage(content=msg['content']))
        
        # Stream the response
        for chunk in get_streaming_response(all_messages):
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)
    
    # Save to message history
    st.session_state['message_history'].append({'role': 'assistant', 'content': full_response})
    
    # Save to database using chatbot
    chatbot.invoke(
        {'messages': [HumanMessage(content=user_input)]},
        config=CONFIG
    )