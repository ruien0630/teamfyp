# FINANCIAL RAG ASSISTANT - Streamlit Frontend
# Purpose: UI for uploading financial documents and querying them via RAG

import json
import os

import streamlit as st

# Create uploads folder if it doesn't exist
UPLOADS_DIR = "./data/uploads"
SELECTIONS_FILE = "./data/selections.json"
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs("./data", exist_ok=True)


def save_selections():
    """Save selected files to disk for persistence across sessions."""
    try:
        with open(SELECTIONS_FILE, "w") as f:
            json.dump(list(st.session_state.selected_files), f)
    except Exception:
        pass


def load_selections():
    """Load selected files from disk if available."""
    try:
        if os.path.exists(SELECTIONS_FILE):
            with open(SELECTIONS_FILE, "r") as f:
                return set(json.load(f))
    except Exception:
        pass
    return set()

# Page Configuration & Styling
st.set_page_config(page_title="Financial RAG Assistant", page_icon="💼", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
    /* Global font styling */
    * {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    
    p, div, span {
        font-weight: 400;
        line-height: 1.5;
    }
    
    /* Hide file uploader's built-in file list */
    div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] + div {
        display: none;
    }
    
    div[data-testid="stFileUploader"] section + div > button {
        display: none;
    }
    
    div[data-testid="stFileUploader"] ul {
        display: none;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] .block-container {
        padding-top: 0.5rem;
    }

    div[data-testid="stFileUploader"] {
        margin-bottom: 0.5rem;
    }

    .source-item {
        margin-top: 0.25rem;
        margin-bottom: 0.25rem;
    }

    section[data-testid="stSidebar"] .stCheckbox {
        margin-bottom: 0.25rem;
    }
    
    /* Conversation item styling */
    .conversation-item {
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 8px;
        background-color: rgba(240, 240, 240, 0.5);
        color: #1a1a1a;
        transition: background-color 0.2s;
    }
    
    .conversation-item:hover {
        background-color: rgba(224, 224, 224, 0.7);
    }
    
    /* Button styling */
    .stButton button {
        border-radius: 8px;
    }
    
    /* Info panel styling */
    .info-panel {
        background-color: #f8f9fa;
        border-left: 1px solid #e0e0e0;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Session State - persists across page reruns
# Without session state, variables reset every time the page reloads

def init_session_state():
    """Initialize all session state variables that persist across reruns"""
    # Chat messages: stores user and assistant messages
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # Selected files: set of filenames user has checked for querying
    if "selected_files" not in st.session_state:
        st.session_state.selected_files = load_selections()
    # Search query: text for searching through files
    if "search_query_value" not in st.session_state:
        st.session_state.search_query_value = ""
    # Viewed file: currently selected file being previewed
    if "viewed_file" not in st.session_state:
        st.session_state.viewed_file = None
    # Show uploader: toggle to show/hide file uploader widget
    if "show_uploader" not in st.session_state:
        st.session_state.show_uploader = False
    # Deleted files: keep track to avoid re-adding on reruns
    if "deleted_files" not in st.session_state:
        st.session_state.deleted_files = set()


def add_uploads(uploaded_files):
    """Save uploaded files to disk and track selection (frontend only)."""
    for uploaded_file in uploaded_files:
        # If user deleted it, don't re-add it during reruns
        if uploaded_file.name in st.session_state.deleted_files:
            continue

        file_path = os.path.join(UPLOADS_DIR, uploaded_file.name)

        # Only write if not already on disk
        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        # Auto-select newly uploaded
        st.session_state.selected_files.add(uploaded_file.name)
    
    # Persist selection to disk
    save_selections()

def format_file_size(size_bytes):
    """Format file size in human-readable format"""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f}MB"

def remove_file(filename):
    """Remove a single file from disk and update UI state (frontend only)."""
    file_path = os.path.join(UPLOADS_DIR, filename)

    if os.path.exists(file_path):
        os.remove(file_path)
    # Mark as deleted so uploader won't re-add it on rerun
    st.session_state.deleted_files.add(filename)
    # Clear any persisted checkbox widget state for this file
    st.session_state.pop(f"cb_{filename}", None)
    # Deselect the file if it was selected
    st.session_state.selected_files.discard(filename)
    # If this file was being previewed, clear the preview
    if st.session_state.viewed_file == filename:
        st.session_state.viewed_file = None
    # Persist selection to disk
    save_selections()


def clear_search():
    """Clear the search query value."""
    st.session_state.search_query_value = ""


def update_selection_from_checkbox(filename, is_checked):
    """Callback to sync checkbox state with selected_files."""
    if is_checked:
        st.session_state.selected_files.add(filename)
    else:
        st.session_state.selected_files.discard(filename)
    save_selections()

# Initialize session state on app start
init_session_state()

def get_files_from_disk():
    """Get list of files from ./data/uploads folder"""
    files = []
    if os.path.exists(UPLOADS_DIR):
        for filename in os.listdir(UPLOADS_DIR):
            file_path = os.path.join(UPLOADS_DIR, filename)
            if os.path.isfile(file_path):
                files.append(filename)
    return sorted(files)


def sync_selected_files_to_checkboxes(files):
    """Clean up selected_files to only contain existing files (don't touch widget keys)."""
    pass  # Removed: widget state is handled by checkbox value= parameter


def set_all(files, checked: bool):
    """Select all / clear selection (updates selected_files only, let widgets initialize from value=)."""
    if checked:
        st.session_state.selected_files = set(files)
    else:
        st.session_state.selected_files = set()

    save_selections()


def refresh_from_folder():
    """Refresh list and also clean selected_files so it only contains existing files."""
    files = get_files_from_disk()
    st.session_state.selected_files = set(st.session_state.selected_files).intersection(files)
    sync_selected_files_to_checkboxes(files)
    save_selections()

# Sidebar - Left Panel (File Management & Conversation Controls)
with st.sidebar:
    # Quick Upload - Users can drag-drop or click to upload files
    st.markdown("### ⬆️ Quick Upload")
    uploaded_files = st.file_uploader(
        "Drop File Here or Click to Upload",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        key="file_uploader",
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        add_uploads(uploaded_files)
        # Reset uploader by clearing its widget state so user can upload more
        st.session_state.pop("file_uploader", None)
        st.rerun()
    
    st.divider()
    
    # Conversations Section
    st.markdown("### 💬 Conversations")
    
    # Active conversation
    with st.container():
        st.markdown('<div class="conversation-item">📁 HybridRAG: Advanced Information Retrieval</div>', unsafe_allow_html=True)
    
    # Conversation actions
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("✏️", help="New conversation"):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("🗑️", help="Delete conversation"):
            st.session_state.messages = []
            st.rerun()
    with col3:
        st.button("📋", help="Copy")
    with col4:
        st.button("📌", help="Pin")
    
    if st.button("Shared", use_container_width=True):
        pass
    
    st.divider()
    
    # File Collection - shows all uploaded files with checkboxes, view, and remove buttons
    with st.expander("📁 File Collection", expanded=True):
        st.text_input("Search files", key="search_query_value", placeholder="Type to filter…")

        if st.button("Search in File(s)", use_container_width=True, type="primary"):
            pass
        
        # Display all uploaded files with UI controls
        disk_files = get_files_from_disk()
        all_files = disk_files  # Keep original list to check if folder is empty
        if st.session_state.search_query_value:
            q = st.session_state.search_query_value.lower()
            disk_files = [f for f in disk_files if q in f.lower()]
        if disk_files:
            for filename in disk_files:
                st.markdown('<div class="source-item">', unsafe_allow_html=True)
                
                # File row: checkbox | filename button | remove button
                col1, col2, col3 = st.columns([0.1, 0.7, 0.2])
                
                # Checkbox to select/deselect file for querying
                with col1:
                    is_selected = st.checkbox(
                        "✓",
                        value=st.session_state.get(f"cb_{filename}", filename in st.session_state.selected_files),
                        key=f"cb_{filename}",
                        label_visibility="collapsed",
                        help="Select file for querying"
                    )
                    # Sync checkbox state with selected_files immediately
                    if is_selected:
                        st.session_state.selected_files.add(filename)
                    else:
                        st.session_state.selected_files.discard(filename)
                    save_selections()
                    
                # Filename button to open/preview file
                with col2:
                    file_path = os.path.join(UPLOADS_DIR, filename)
                    file_size = os.path.getsize(file_path)
                    if st.button(f"📄 {filename[:20]}...", key=f"view_{filename}", use_container_width=True, type="secondary"):
                        st.session_state.viewed_file = filename
                        st.rerun()
                
                # Remove button (✕) to delete file
                with col3:
                    if st.button("✕", key=f"remove_{filename}", help="Remove file"):
                        remove_file(filename)
                        st.rerun()
                
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Show appropriate message: empty folder vs. no search results
            if not all_files:
                st.info("No files uploaded yet.")
            else:
                st.info("No files match your search.")
    
    st.divider()
    
    # GraphRAG Collection Section
    with st.expander("🕸️ GraphRAG Collection", expanded=False):
        col1, col2 = st.columns(2)
        with col1: 
            st.button("Search All", use_container_width=True, key="graph_search_all")
        with col2:
            st.button("Search in File(s)", use_container_width=True, type="primary", key="graph_search_files")
    
    st.divider()
    
    # File Preview - shows details and options for the currently viewed file
    if st.session_state.viewed_file:
        file_path = os.path.join(UPLOADS_DIR, st.session_state.viewed_file)
        if os.path.exists(file_path):
            with st.expander("📄 File Preview", expanded=True):
                viewed_filename = st.session_state.viewed_file
                file_size = os.path.getsize(file_path)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✕ Close Preview", use_container_width=True, key="close_preview"):
                        st.session_state.viewed_file = None
                        st.rerun()
                with col2:
                    with open(file_path, "rb") as f:
                        st.download_button(
                            label="⬇️ Download",
                            data=f.read(),
                            file_name=viewed_filename,
                            mime="application/octet-stream",
                            use_container_width=True
                        )
                
                st.markdown(f"**File:** {viewed_filename}")
                st.metric("Size", format_file_size(file_size))
                
                st.divider()
                
                st.info("📝 File Preview")
                st.markdown("""
                **Preview area** - Backend integration needed.
                
                Will display:
                - PDF pages
                - Text content
                - Document structure
                """)
        
        st.divider()

# Main Interface - Chat Area
# Users interact with the RAG system through this chat interface

with st.container(border=True):
    st.markdown("### 💬 Chat")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Chat input"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        # Assistant response
        with st.chat_message("assistant"):
            if not st.session_state.selected_files:
                answer = "Please select at least one file."
                st.markdown(answer)
            else:
                with st.spinner("Searching through documents..."):
                    answer = f"UI Response: I am now querying the following files: {', '.join(st.session_state.selected_files)}"
                    st.markdown(answer)
            
            st.session_state.messages.append({"role": "assistant", "content": answer})
