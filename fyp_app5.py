from ast import With
import json
import streamlit as st
import os
os.environ["HF_HUB_ENABLE_SYMLINKS"] = "0"

import tempfile
from dotenv import load_dotenv
from docling_util import (
                            convert_pdf_with_image_annotation, 
                            process_documents_to_md,
                            process_markdown_folder
)
from query_util import (
                            setup_qa_chain,
                            ask_question
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from langchain_classic.memory import ConversationBufferMemory

from langchain_ibm import WatsonxLLM
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

UPLOADS_DIR = "./input"
OUTPUTS_DIR = "./output_md"
SELECTIONS_FILE = "./input/selections.json"
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

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

def get_files_from_disk():
    """Get list of files from ./input folder"""
    files = []
    if os.path.exists(UPLOADS_DIR):
        for filename in os.listdir(UPLOADS_DIR):
            file_path = os.path.join(UPLOADS_DIR, filename)
            if os.path.isfile(file_path):
                files.append(filename)
    return sorted(files)

# ---------------------------------------
# CUSTOM UI THEME & STYLING (UI ONLY)
# ---------------------------------------
st.markdown("""
<style>

/* --------------------------
   Sidebar Styling
--------------------------- */
section[data-testid="stSidebar"] {
    background-color: #111827;
    color: white;
    padding: 20px;
}
section[data-testid="stSidebar"] * {
    color: #d1d5db !important;
}

/* --------------------------
   Main Content Adjustments
--------------------------- */
.block-container {
    padding-top: 1rem;
}

/* --------------------------
   Chat Bubble Styling
--------------------------- */
.stChatMessage {
    padding: 12px 18px !important;
    border-radius: 14px !important;
    margin-bottom: 14px !important;
}
.stChatMessage.user {
    background: #2563eb20 !important;
    border: 1px solid #2563eb30 !important;
}
.stChatMessage.assistant {
    background: #f3f4f6 !important;
    border: 1px solid #e5e7eb !important;
}

/* Chat window scroll container */
.chat-scroll {
    max-height: 65vh;
    overflow-y: auto;
    padding-right: 10px;
}

/* Input bar spacing */
[data-testid="chat-input"] {
    margin-top: 20px !important;
}

/* --------------------------
   Buttons
--------------------------- */
button {
    border-radius: 10px !important;
}

/* --------------------------
   Nice Expander Design
--------------------------- */
.streamlit-expanderHeader {
    font-size: 14px !important;
    font-weight: 600 !important;
    color: #d1d5db !important;
}

/* --------------------------
   Custom Loading Animation
--------------------------- */
.loading-dots {
    display: inline-block;
    font-size: 18px;
    font-weight: bold;
    color: #2563eb;
    letter-spacing: 2px;
}

.loading-dots::after {
    content: '';
    display: inline-block;
    animation: dots 1.4s infinite steps(3);
}

@keyframes dots {
    0% { content: ''; } 
    33% { content: '.'; }
    66% { content: '..'; }
    100% { content: '...'; }
}

div[data-testid="stChatInput"] {
    position: fixed;
    bottom: 20px;
    left: 50%;
    transform: translateX(-50%);
    width: calc(100% - 520px);
    max-width: 900px;
    z-index: 100;
}

/* Prevent chat content from being hidden behind input */
.chat-scroll {
    padding-bottom: 120px;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# LLM & Embeddings
# ------------------------------

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

CHROMA_PERSIST_DIR = "chroma_db"

# ---------------------------------------
# NEW: Load existing vector DB if available
# ---------------------------------------

@st.cache_resource
def load_existing_vector_store():
    """Loads the local Chroma DB if it exists."""
    if os.path.exists(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR):
        try:
            return Chroma(
                persist_directory=CHROMA_PERSIST_DIR,
                embedding_function=embeddings
            )
        except Exception as e:
            st.error(f"⚠ Failed to load existing vector DB: {e}")
    return None

# ---------------------------------------
# Existing file processing
# ---------------------------------------

@st.cache_resource
def build_vector_store_from_disk():
    docs = process_markdown_folder(OUTPUTS_DIR)

    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=200
    )

    split_docs = splitter.split_documents(docs)

    return Chroma.from_documents(
        split_docs,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR
    )

def get_conversation_chain(vector_store):
    if vector_store is None:
        return None

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    return RetrievalQA.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )

# ---------------------------------------
# Chat Handling With Loading Animation
# ---------------------------------------
def handle_user_input(user_question):
    if st.session_state.conversation is None:
        st.error("⚠ No database loaded. Please process files or load existing DB.")
        return
    
    current_chat = st.session_state.chat_sessions[st.session_state.current_chat]
    # Save user question to prompt history
    st.session_state.prompt_history.append(user_question)
    
    # Temporary placeholder "Thinking…" animation
    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("<span class='loading-dots'>Thinking</span>", unsafe_allow_html=True)
    
    # Get model response
    result = ask_question(
        qa_chain=st.session_state.conversation,
        question=user_question
    )

    answer = result["answer"]
    sources = result["sources"]
    confidence = result["confidence"]

    # Replace loading animation with final answer
    thinking_placeholder.empty()
    
    # Save messages
    current_chat["messages"].append({"role": "user", "content": user_question})
    current_chat["messages"].append({"role": "assistant", "content": answer})
    
    # Display user & assistant messages
    with st.chat_message("user"):
        st.markdown(user_question)
    with st.chat_message("assistant"):
        st.markdown(answer)

        if sources:
            st.markdown("**Sources:**")
            for src in set(sources):
                st.markdown(f"- {src}")

# ---------------------------------------
# Main UI
# ---------------------------------------
def main():
    st.set_page_config(
        page_title="RAG Chatbot",
        page_icon="🤖",
        layout="wide"
    )

    # Initialize session state
    """Initialize all session state variables that persist across reruns"""
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "prompt_history" not in st.session_state:
        st.session_state.prompt_history = []
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = {}
    if "current_chat" not in st.session_state:
        st.session_state.current_chat = "Default"
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
    # Pending uploads (files staged but not yet processed)
    if "pending_uploads" not in st.session_state:
        st.session_state.pending_uploads = []

    # 🔥 AUTO LOAD LOCAL CHROMA DB ON STARTUP
    if st.session_state.conversation is None:
        existing_db = load_existing_vector_store()

        if existing_db:
            st.session_state.conversation = setup_qa_chain(
                local_vector_store_path=CHROMA_PERSIST_DIR,
                use_local_path=True
            )
            if not st.session_state.messages:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Loaded local database successfully! Ask me anything."
                })
        else:
            if not st.session_state.messages:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "No local database found. Upload files to create one."
                })

    if "Default" not in st.session_state.chat_sessions:
        st.session_state.chat_sessions["Default"] = {
            "messages": st.session_state.messages,
            "conversation": st.session_state.conversation
        }

    # Sidebar
    with st.sidebar:
        st.title("Conversations")
        
        chat_option = st.selectbox(
            " ",
            ["Default", "New Chat"],
            label_visibility="collapsed"
        )

        if chat_option == "New Chat":
            new_chat_name = f"Chat {len(st.session_state.chat_sessions) + 1}"
            st.session_state.current_chat = new_chat_name

            st.session_state.chat_sessions[new_chat_name] = {
                "messages": [
                    {"role": "assistant", "content": "New chat started. Ask me anything."}
                ],
                "conversation": st.session_state.conversation
            }
        else:
            st.session_state.current_chat = "Default"

        st.divider()
        
        st.markdown("### ⬆️ Quick Upload")

        uploaded_files = st.file_uploader(
            "Drop File Here or Click to Upload",
            type=["pdf", "md"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )

        if uploaded_files:
            st.session_state.pending_uploads = uploaded_files
            st.info("🕓 Files uploaded. Click 'Process Documents' to index them.")

        if st.button("Process Documents", use_container_width=True):
            # 1️⃣ Check staged files
            if not st.session_state.pending_uploads:
                st.warning("Please upload files before processing.")
            else:
                with st.spinner("Processing documents..."):
                    # 1️⃣ Save uploaded files
                    add_uploads(st.session_state.pending_uploads)

                    # 2️⃣ Convert PDFs using Docling
                    for uploaded_file in st.session_state.pending_uploads:
                        file_path = os.path.join(UPLOADS_DIR, uploaded_file.name)

                        if uploaded_file.name.lower().endswith(".pdf"):
                            file_path = os.path.join(UPLOADS_DIR, uploaded_file.name)
                            if not os.path.exists(file_path):
                                st.warning(f"File {uploaded_file.name} not found, skipping Docling processing.")
                                continue
                            convert_pdf_with_image_annotation(file_path)

                    # 3️⃣ Convert all documents to Markdown
                    process_documents_to_md(UPLOADS_DIR,OUTPUTS_DIR)

                    # 4️⃣ Build vector database from processed markdown
                    build_vector_store_from_disk()

                    # 5️⃣ Create conversation chain
                    st.session_state.conversation = setup_qa_chain(
                        local_vector_store_path=CHROMA_PERSIST_DIR,
                        use_local_path=True
                    )
                    # 6️⃣ Reset chat messages
                    st.session_state.messages = [
                        {
                            "role": "assistant",
                            "content": "Documents processed! Ask me anything."
                        }
                    ]
                    # 7️⃣ Clear staged uploads
                    st.session_state.pending_uploads = []

                st.success("Completed!")

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
        
        with st.expander("🕘 User Prompt History", expanded=False):
            if st.session_state.prompt_history:
                for i, q in enumerate(st.session_state.prompt_history, 1):
                    st.markdown(f"**{i}.** {q}")
            else:
                st.write("No prompts yet.")

        with st.expander("📊 GraphRAG Collection", expanded=False):
            st.write("Coming soon...")
            
        with st.expander("📝 Feedback"):
            st.text_area("Your feedback...", placeholder="Enter your thoughts here")

    # ---------------------------------------
    # Layout: Left Sidebar | Main | Right Sidebar
    # ---------------------------------------
    left_sidebar, main_col, right_sidebar = st.columns([0.2, 6, 1.5])

    with main_col:
        # Main Chat Area
        st.markdown("<hr style='border: 1px solid #e5e7eb;'>", unsafe_allow_html=True)
        st.title("📄 RAG Chatbot")
        st.markdown(
            "<p style='color:#6b7280; font-size:16px; margin-top:-10px;'>AI Assistant powered by Granite LLM & Chroma</p>",
            unsafe_allow_html=True
        )

        # Scrollable chat history container
        st.markdown('<div class="chat-scroll">', unsafe_allow_html=True)

        current_chat = st.session_state.chat_sessions[st.session_state.current_chat]

        for msg in current_chat["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        st.markdown('</div>', unsafe_allow_html=True)

        # Chat input
        user_prompt = st.chat_input("Ask something about your documents...")
        if user_prompt:
            handle_user_input(user_prompt)
    
    with right_sidebar:
        st.divider()
        st.markdown("#### 📚 Source Selector")

        source_option = st.radio(
            "Search Source",
            [
                "All Documents",
                "Uploaded Files Only",
                "Current File",
                "GraphRAG (Coming Soon)"
            ],
            index=0
        )

        st.divider()

        st.markdown("#### ⚙️ Search Options")

        top_k = st.slider("Top K Results", 1, 10, 4)
        similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.7)

        # Save user selection (NO logic change yet)
        st.session_state.selected_source = source_option
        st.session_state.top_k = top_k
        st.session_state.similarity_threshold = similarity_threshold

if __name__ == "__main__":

    main()
