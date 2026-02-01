# FINANCIAL RAG ASSISTANT - Streamlit Frontend
# Purpose: UI for uploading financial documents and querying them via RAG

import json
import os

import streamlit as st
import query_util_integrated as query_util  # Using your best performing prompt logic
import docling_util_integrated as docling_util # Import your processing utility


# Page Configuration & Styling
st.set_page_config(page_title="Financial RAG Assistant", page_icon="💼", layout="wide")

# Initialize the chain once globally to save memory and time
# Avoid creating a ghost chroma_db by loading only if it exists
if "qa_chain" not in st.session_state:
    with st.sidebar:  # Show loader in sidebar to keep chat clean
        if os.path.exists("./chroma_db"):
            try:
                with st.spinner("🔄 Loading Existing Knowledge..."):
                    st.session_state.qa_chain = query_util.setup_qa_chain(
                        local_vector_store_path="./chroma_db",
                        k=5,
                        fetch_k=40,
                        lambda_mult=0.5
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
    
    /* Fixed chat input at bottom */
    .stChatFloatingInputContainer {
        position: fixed !important;
        bottom: 20px !important;
        left: 50% !important;
        transform: translateX(-50%) !important;
        width: calc(100% - 400px) !important;
        max-width: 900px !important;
        z-index: 999 !important;
    }
    
    /* Add padding to bottom of main area to prevent content hiding under fixed input */
    .main .block-container {
        padding-bottom: 100px !important;
    }
    
    /* Make chat messages scrollable */
    section[data-testid="stVerticalBlock"] {
        max-height: calc(100vh - 200px);
        overflow-y: auto;
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
    # Track last selected files for filtered retrieval
    if "qa_filter" not in st.session_state:
        st.session_state.qa_filter = tuple()



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
            with st.status(f"Uploading {uploaded_file.name}…", expanded=False) as status:
                progress = st.progress(0)
                st.caption("Uploading…")

                # Save complete (you already saved above)
                progress.progress(25)

                # Processing (hide technical details)
                status.update(label=f"Processing {uploaded_file.name}…", state="running")
                st.caption("Processing… this may take a moment.")
                progress.progress(50)

                # Create Chroma vectordb directly from the uploaded Docling file using ContentAwareSplitting
                docling_util.create_chroma_vectordb(
                    file_paths=[file_path],
                    text_splitter_choice="ContentAwareSplitting",
                    splitter_para={'max_tokens': 800}
                )
                progress.progress(100)

                status.update(label=f"Upload done ✅", state="complete")

            # 3. STEP: REFRESH AI ENGINE
            # Clear the old qa_chain and reinitialize with updated ChromaDB
            st.session_state.pop("qa_chain", None)
            
            # Reinitialize qa_chain with the newly created/updated ChromaDB
            try:
                st.session_state.qa_chain = query_util.setup_qa_chain(
                    local_vector_store_path="./chroma_db",
                    k=5,
                    fetch_k=40,
                    lambda_mult=0.5
                )
            except Exception as e:
                st.error(f"Failed to reload QA chain: {e}")
            
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
        # 1. Remove uploaded PDF/DOCX file
        file_path = os.path.join(UPLOADS_DIR, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            st.info(f"✓ Removed {filename} from uploads folder")
#only delete the uploaded file from the input folder and CN handle remove from the rest
        # 2. Remove markdown file if it exists
        base_name = filename.replace('.pdf', '').replace('.docx', '').replace('.txt', '')
        md_path = f"./output_md/{base_name}.md"
        if os.path.exists(md_path):
            os.remove(md_path)
            st.info(f"✓ Removed {base_name}.md from output folder")

        # 3. Remove vectors from Chroma DB
        if st.session_state.get("qa_chain") is not None:
            vectorstore = st.session_state.qa_chain.vectorstore
            
            try:
                # Get all vectors to find matching source paths
                all_data = vectorstore.get(include=['metadatas'])
                ids_to_delete = []
                
                if all_data and all_data.get('ids'):
                    metadatas = all_data.get('metadatas', [])
                    for i, metadata in enumerate(metadatas):
                        source = metadata.get('source', '') if isinstance(metadata, dict) else ''
                        # Match if source contains base_name or filename
                        if base_name.lower() in source.lower() or filename.lower() in source.lower():
                            ids_to_delete.append(all_data['ids'][i])
                
                if ids_to_delete:
                    vectorstore.delete(ids=ids_to_delete)
                    st.info(f"✓ Deleted {len(ids_to_delete)} vectors from ChromaDB")
                else:
                    st.warning(f"⚠ No vectors found for {filename} in ChromaDB")
            except Exception as e:
                st.warning(f"Could not delete vectors: {e}")
            
            # Refresh the QA chain to reload from fresh DB
            st.session_state.pop("qa_chain", None)

        # 4. Update session state
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
        
        # Select/Deselect All buttons

        
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
                    size_str = format_file_size(file_size)
                    if st.button(f"📄 {filename[:20]}... ({size_str})", key=f"view_{filename}", use_container_width=True, type="secondary"):
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
    
    # View Markdown Files section
    st.divider()
    with st.expander("📄 View Markdown Files", expanded=False):
        md_dir = "./output_md"
        if os.path.exists(md_dir):
            md_files = [f for f in os.listdir(md_dir) if f.endswith(".md")]
            if md_files:
                selected_md = st.selectbox("Select markdown file to view:", sorted(md_files), key="md_selector")
                if selected_md:
                    md_path = os.path.join(md_dir, selected_md)
                    try:
                        with open(md_path, "r", encoding="utf-8") as f:
                            md_content = f.read()
                        st.markdown(md_content, unsafe_allow_html=True)
                        
                        # Download button
                        st.download_button(
                            label=f"⬇️ Download {selected_md}",
                            data=md_content,
                            file_name=selected_md,
                            mime="text/markdown"
                        )
                    except Exception as e:
                        st.error(f"Could not read file: {e}")
            else:
                st.info("No markdown files generated yet. Upload a document to create one.")
        else:
            st.info("Markdown folder not found.")
    
  
    

#Using CN chat interface   
# --- MAIN CHAT INTERFACE ---
with st.container(border=True):
    st.markdown("### 💬 Chat")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# --- INPUT LOGIC (The Fix) --- Move outside container for fixed positioning
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
                # Rebuild QA chain if selected files changed (to filter retrieval)
                selected_files = tuple(sorted(st.session_state.selected_files))
                if st.session_state.qa_filter != selected_files:
                    try:
                        filter_dict = None
                        if selected_files:
                            selected_paths = [os.path.join(UPLOADS_DIR, name) for name in selected_files]
                            filter_dict = {"source": {"$in": selected_paths}}
                        st.session_state.qa_chain = query_util.setup_qa_chain(
                            local_vector_store_path="./chroma_db",
                            k=5,
                            fetch_k=40,
                            lambda_mult=0.5,
                            filter_dict=filter_dict
                        )
                        st.session_state.qa_filter = selected_files
                    except Exception as e:
                        st.error(f"Failed to update QA chain filter: {e}")

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