# FINANCIAL RAG ASSISTANT - Streamlit Frontend
# Purpose: UI for uploading financial documents and querying them via RAG

import json
import os

import streamlit as st
import Experiments.query_util_exp5 as query_util  # Using your best performing prompt logic
import docling_util_baseline as docling_util # Import your processing utility

# Initialize the chain once globally to save memory and time
# Avoid creating a ghost chroma_db by loading only if it exists
if "qa_chain" not in st.session_state:
    with st.sidebar:  # Show loader in sidebar to keep chat clean
        if os.path.exists("./chroma_db"):
            try:
                with st.spinner("🔄 Loading Existing Knowledge..."):
                    st.session_state.qa_chain = query_util.setup_qa_chain(
                        local_vector_store_path="./chroma_db"
                    )
            except Exception as e:
                st.error(f"Failed to initialize QA chain: {e}")
                # Keep app usable; initialize to None until upload occurs
                st.session_state.qa_chain = None
        else:
            # No DB yet; wait for first upload before initializing
            st.session_state.qa_chain = None
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
    """Save files, convert to Markdown, and update ChromaDB with progress tracking."""
    for uploaded_file in uploaded_files:
        if uploaded_file.name in st.session_state.deleted_files:
            continue

        file_path = os.path.join(UPLOADS_DIR, uploaded_file.name)

        if not os.path.exists(file_path):
            # 1. STEP: PHYSICAL UPLOAD (Saving to /data/uploads)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # 2. STEP: BACKEND PROCESSING (Transition Status)
            with st.status(f"🚀 Ingesting {uploaded_file.name}...", expanded=True) as status:
                # Provide a visual progress bar
                progress_bar = st.progress(0)
                
                st.write("📂 Converting PDF to structured Markdown...")
                # Call Docling utility
                docling_util.process_documents_to_md(UPLOADS_DIR, "no image annotation")
                progress_bar.progress(40)
                
                st.write("🧠 Chunking text and generating embeddings...")
                # Target the specific markdown file created
                md_path = f"output_md/{uploaded_file.name.replace('.pdf', '.md')}"
                # Call Chroma utility
                docling_util.create_chroma_vectordb(file_paths=[md_path])
                progress_bar.progress(90)
                
                # Signal completion
                progress_bar.progress(100)
                status.update(label=f"✅ {uploaded_file.name} is now searchable!", state="complete")

            # 3. STEP: REFRESH AI ENGINE
            # Clear the old qa_chain so the next query sees the new data
            st.session_state.pop("qa_chain", None)
            st.session_state.selected_files.add(uploaded_file.name)
            save_selections()
            st.rerun()

def format_file_size(size_bytes):
    """Format file size in human-readable format"""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f}MB"

def remove_file(filename):
    """Fast Delete: Removes from disk and specifically wipes vectors from Chroma."""
    try:
        file_path = os.path.join(UPLOADS_DIR, filename)
        if os.path.exists(file_path):
            os.remove(file_path)

        if st.session_state.get("qa_chain") is not None:
            # Try multiple possible path formats
            base_name = filename.replace('.pdf', '').replace('.docx', '').replace('.txt', '')
            possible_paths = [
                f"output_md/{base_name}.md",
                f"output_md\\{base_name}.md",
                f"{base_name}.md",
                filename
            ]
            
            # Try to get all documents and find the ones matching this file
            vectorstore = st.session_state.qa_chain.vectorstore
            deleted_count = 0
            
            for path in possible_paths:
                try:
                    # Get all documents with this source path
                    results = vectorstore.get(where={"source": path})
                    if results and results.get('ids'):
                        # Delete by IDs
                        vectorstore.delete(ids=results['ids'])
                        deleted_count += len(results['ids'])
                        st.info(f"Deleted {len(results['ids'])} vectors from path: {path}")
                except Exception as e:
                    continue
            
            if deleted_count == 0:
                st.warning(f"No vectors found for {filename}. It may have already been deleted.")
            
            # Refresh the engine
            st.session_state.pop("qa_chain", None)

        st.session_state.deleted_files.add(filename)
        st.session_state.pop(f"cb_{filename}", None)
        st.session_state.selected_files.discard(filename)
        if st.session_state.viewed_file == filename:
            st.session_state.viewed_file = None
        save_selections()

        st.success(f"✅ Successfully cleared {filename} from disk and AI memory.")
        st.rerun()

    except Exception as e:
        st.error(f"Delete failed: {str(e)}")


def clear_search():
    """Clear the search query value."""
    st.session_state.search_query_value = ""


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
                
                # File row: filename button | remove button
                col1, col2 = st.columns([0.75, 0.25])
                
                # Filename button to open/preview file
                with col1:
                    file_path = os.path.join(UPLOADS_DIR, filename)
                    file_size = os.path.getsize(file_path)
                    if st.button(f"📄 {filename[:25]}...", key=f"view_{filename}", use_container_width=True, type="secondary"):
                        st.session_state.viewed_file = filename
                        st.rerun()
                # Remove button (✕) to delete file
                with col2:
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

# --- MAIN CHAT INTERFACE ---
with st.container(border=True):
    st.markdown("### 💬 Chat")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# --- INPUT LOGIC (The Fix) ---
if prompt := st.chat_input("Ask a financial question..."):
    user_query = prompt.strip()
    
    if user_query:
        # 1. Store and display user message immediately
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # 2. Assistant Response Logic
        with st.chat_message("assistant"):
            if st.session_state.get("qa_chain") is None:
                st.warning("Please upload a document first to start the AI.")
            else:
                with st.spinner("Analyzing documents..."):
                    try:
                        result = query_util.ask_question(st.session_state.qa_chain, user_query)
                        answer = result.get("answer", "I couldn't find an answer.")

                        # Display immediately
                        st.markdown(answer)

                        # Display sources immediately (before rerun)
                        sources = result.get("sources", [])
                        if sources and len(sources) > 0:
                            with st.expander("🔍 View Specific Document Chunks", expanded=True):
                                for i, doc in enumerate(sources):
                                    st.markdown(f"**Chunk {i+1}** | Source: {doc.get('source', 'Unknown')}")
                                    st.info(doc.get("content", "No content"))
                                    st.divider()
                        else:
                            st.info("No source documents retrieved.")

                        # Save to history AFTER displaying
                        st.session_state.messages.append({"role": "assistant", "content": answer})

                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})