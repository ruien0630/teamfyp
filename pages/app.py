import streamlit as st
import os
import json
import shutil
from datetime import datetime

# Chat sessions directory
CHAT_SESSIONS_DIR = "chat_sessions"
os.makedirs(CHAT_SESSIONS_DIR, exist_ok=True)

# Function to load chat names
def load_chat_names():
    chat_names_file = os.path.join(CHAT_SESSIONS_DIR, "chat_names.json")
    if os.path.exists(chat_names_file):
        try:
            with open(chat_names_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

# Function to save chat names
def save_chat_names(chat_names):
    chat_names_file = os.path.join(CHAT_SESSIONS_DIR, "chat_names.json")
    with open(chat_names_file, 'w', encoding='utf-8') as f:
        json.dump(chat_names, f, indent=2, ensure_ascii=False)

# Function to delete a chat session
def delete_chat_session(session_id):
    """Delete all files and folders related to a chat session"""
    if session_id == "default":
        st.error("Cannot delete the default chat")
        return False
    
    try:
        # Delete session files
        chat_file = os.path.join(CHAT_SESSIONS_DIR, f"{session_id}_chat.json")
        files_file = os.path.join(CHAT_SESSIONS_DIR, f"{session_id}_files.json")
        
        if os.path.exists(chat_file):
            os.remove(chat_file)
        if os.path.exists(files_file):
            os.remove(files_file)
        
        # Delete input directory
        input_dir = os.path.join("Company Annual Report", session_id)
        if os.path.exists(input_dir):
            shutil.rmtree(input_dir)
        
        # Delete output directory
        output_dir = os.path.join("output_md", session_id)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        
        # Remove from chat names
        chat_names = load_chat_names()
        if session_id in chat_names:
            del chat_names[session_id]
            save_chat_names(chat_names)
        
        return True
    except Exception as e:
        st.error(f"Error deleting chat: {str(e)}")
        return False

# Function to get all chat sessions
def get_all_sessions():
    """Get list of all chat session IDs"""
    sessions = []
    if os.path.exists(CHAT_SESSIONS_DIR):
        for file in os.listdir(CHAT_SESSIONS_DIR):
            if file.endswith("_chat.json"):
                session_id = file.replace("_chat.json", "")
                sessions.append(session_id)
    
    # Always include default session
    if "default" not in sessions:
        sessions.insert(0, "default")
    
    return sorted(sessions, reverse=True)  # Most recent first

# Initialize current session if not set
if "current_session_id" not in st.session_state:
    st.session_state["current_session_id"] = "default"

# Handle session query param BEFORE setting up navigation
if "session" in st.query_params:
    new_session = st.query_params["session"]
    # Get all sessions to validate
    temp_sessions = []
    if os.path.exists(CHAT_SESSIONS_DIR):
        for file in os.listdir(CHAT_SESSIONS_DIR):
            if file.endswith("_chat.json"):
                temp_sessions.append(file.replace("_chat.json", ""))
    
    if new_session in temp_sessions:
        st.session_state["current_session_id"] = new_session
        del st.query_params["session"]
        st.rerun()

# Get all sessions and their names
all_sessions = get_all_sessions()
chat_names = load_chat_names()

st.set_page_config(page_title="Chat Manager", page_icon="💬")

# New chat page
new_chat_page = st.Page("create.py", title=" New Chat", icon=":material/add_circle:", url_path="new_chat")

# Delete page
delete_page = st.Page("delete.py", title=" Delete Chats", icon=":material/delete:", url_path="delete")

# Don't create separate pages for each chat - they all show the same main_page.py
# Instead, create one main page and let users switch sessions via sidebar
main_page = st.Page("main_page.py", title="Chat", icon="💬", default=True, url_path="chat")

# Set up navigation with sections - only utility pages in nav
pg = st.navigation({
    "": [main_page, new_chat_page, delete_page]
})

# Store all sessions for the sidebar selector
st.session_state["all_available_sessions"] = all_sessions
st.session_state["chat_names_map"] = chat_names

# Run the selected page
pg.run()


