import streamlit as st
import json
import os
from datetime import datetime

st.title("💬 Create New Chat")
st.write("Start a fresh conversation with a custom name.")

# Chat sessions directory
CHAT_SESSIONS_DIR = "chat_sessions"
CHAT_NAMES_FILE = os.path.join(CHAT_SESSIONS_DIR, "chat_names.json")
os.makedirs(CHAT_SESSIONS_DIR, exist_ok=True)

# Load existing chat names
def load_chat_names():
    if os.path.exists(CHAT_NAMES_FILE):
        try:
            with open(CHAT_NAMES_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_chat_names(names):
    with open(CHAT_NAMES_FILE, 'w', encoding='utf-8') as f:
        json.dump(names, f, ensure_ascii=False, indent=2)

# Form to create new chat
with st.form("new_chat_form"):
    chat_name = st.text_input(
        "Chat Name",
        placeholder="e.g., Financial Analysis, Q&A Session, etc.",
        help="Give your chat a descriptive name"
    )
    
    submitted = st.form_submit_button("✨ Create Chat", type="primary", use_container_width=True)
    
    if submitted:
        if not chat_name or not chat_name.strip():
            st.error("❌ Please enter a chat name!")
        else:
            # Generate unique session ID
            new_session_id = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Save chat name
            chat_names = load_chat_names()
            chat_names[new_session_id] = chat_name.strip()
            save_chat_names(chat_names)
            
            # Create empty chat history file so it appears in navigation
            chat_history_file = os.path.join(CHAT_SESSIONS_DIR, f"{new_session_id}_chat.json")
            with open(chat_history_file, 'w', encoding='utf-8') as f:
                json.dump([], f)  # Empty chat history
            
            # Create empty files list
            files_list_file = os.path.join(CHAT_SESSIONS_DIR, f"{new_session_id}_files.json")
            with open(files_list_file, 'w', encoding='utf-8') as f:
                json.dump([], f)  # Empty files list
            
            # Set as current session
            st.session_state["current_session_id"] = new_session_id
            st.session_state["session_for_this_page"] = new_session_id
            
            # Force reload of messages and files for new session by clearing cached data
            # This will make main_page.py load fresh data from the new session files
            if "messages" in st.session_state:
                del st.session_state["messages"]
            if "uploaded_files" in st.session_state:
                del st.session_state["uploaded_files"]
            if "uploader_key" in st.session_state:
                del st.session_state["uploader_key"]
            
            st.success(f"✓ Chat '{chat_name}' created successfully!")
            st.balloons()
            
            # Navigate to the new chat page using query params to ensure correct session
            st.query_params["session"] = new_session_id
            st.switch_page("main_page.py")

st.write("---")
st.caption("💡 Tip: Each chat is completely independent with its own files and conversation history.")

