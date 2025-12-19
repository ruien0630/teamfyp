import streamlit as st
import json
import os
import shutil

st.title("🗑️ Delete Chats")
st.write("Manage and delete your chat sessions.")

# Chat sessions directory
CHAT_SESSIONS_DIR = "chat_sessions"
CHAT_NAMES_FILE = os.path.join(CHAT_SESSIONS_DIR, "chat_names.json")

# Load chat names
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

# Get all chat sessions
def get_all_sessions():
    """Get list of all chat session IDs"""
    sessions = []
    if os.path.exists(CHAT_SESSIONS_DIR):
        for file in os.listdir(CHAT_SESSIONS_DIR):
            if file.endswith("_chat.json"):
                session_id = file.replace("_chat.json", "")
                sessions.append(session_id)
    return sorted(sessions, reverse=True)

# Delete a chat session
def delete_chat_session(session_id):
    """Delete all files and folders related to a chat session"""
    if session_id == "default":
        st.error("❌ Cannot delete the default chat")
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
        st.error(f"❌ Error deleting chat: {str(e)}")
        return False

# Get all sessions
all_sessions = get_all_sessions()
chat_names = load_chat_names()

# Filter out default session
deletable_sessions = [s for s in all_sessions if s != "default"]

if not deletable_sessions:
    st.info("📭 No chats to delete. Create a new chat first!")
else:
    st.write(f"**Total chats:** {len(deletable_sessions)}")
    st.markdown("---")
    
    # Display each chat with delete button
    for session_id in deletable_sessions:
        display_name = chat_names.get(session_id, f"Chat {session_id[-6:]}")
        
        # Get session info
        chat_file = os.path.join(CHAT_SESSIONS_DIR, f"{session_id}_chat.json")
        files_file = os.path.join(CHAT_SESSIONS_DIR, f"{session_id}_files.json")
        
        # Count messages and files
        message_count = 0
        file_count = 0
        
        try:
            if os.path.exists(chat_file):
                with open(chat_file, 'r', encoding='utf-8') as f:
                    messages = json.load(f)
                    message_count = len(messages)
            
            if os.path.exists(files_file):
                with open(files_file, 'r', encoding='utf-8') as f:
                    files = json.load(f)
                    file_count = len(files)
        except:
            pass
        
        # Display chat info in expander
        with st.expander(f"💬 {display_name}", expanded=False):
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                st.write(f"**Session ID:** `{session_id}`")
                st.write(f"**Messages:** {message_count}")
            
            with col2:
                st.write(f"**Files uploaded:** {file_count}")
                is_current = (session_id == st.session_state.get("current_session_id", "default"))
                if is_current:
                    st.write("**Status:** 🟢 Active")
                else:
                    st.write("**Status:** ⚪ Inactive")
            
            with col3:
                if st.button(f"🗑️ Delete", key=f"delete_{session_id}", type="secondary"):
                    if delete_chat_session(session_id):
                        st.success(f"✓ Deleted '{display_name}'")
                        
                        # If deleting current session, switch to default
                        if is_current:
                            st.session_state["current_session_id"] = "default"
                        
                        # Check if there are any remaining chats
                        remaining_sessions = get_all_sessions()
                        deletable_remaining = [s for s in remaining_sessions if s != "default"]
                        
                        # If no more chats, redirect to default chat
                        if not deletable_remaining:
                            st.session_state["current_session_id"] = "default"
                            st.switch_page("main_page.py")
                        else:
                            st.rerun()

st.write("---")
st.caption("⚠️ **Warning:** Deleting a chat will permanently remove all messages and uploaded files.")
