import streamlit as st
import os
import sys
import shutil
import json
from pathlib import Path
from dotenv import load_dotenv
import tempfile
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import local utilities
from docling_util_Final import create_chroma_vectordb
from query_util_Final import setup_qa_chain, ask_question
from langchain_community.vectorstores import Chroma
from langchain_docling import DoclingLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata
from docling.document_converter import DocumentConverter

# Load environment variables
load_dotenv()

# Constants
UPLOAD_DIR = "./uploaded_documents"
CHROMA_DB_DIR = "./chroma_db"
CHAT_HISTORY_FILE = "./chat_history.json"
SESSION_UPLOAD_DIR = None

def ensure_directories():
    """Ensure necessary directories exist"""
    Path(UPLOAD_DIR).mkdir(exist_ok=True)
    Path(CHROMA_DB_DIR).mkdir(exist_ok=True)

def save_chat_history():
    """Save chat history to JSON file"""
    try:
        with open(CHAT_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(st.session_state.chat_history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving chat history: {e}")

def load_chat_history():
    """Load chat history from JSON file"""
    try:
        if os.path.exists(CHAT_HISTORY_FILE):
            with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as f:
                history = json.load(f)
                # Validate format: should be list of dicts or tuples
                if isinstance(history, list):
                    valid_history = []
                    for item in history:
                        # New format: dict with question, answer, sources
                        if isinstance(item, dict) and 'question' in item and 'answer' in item:
                            valid_history.append(item)
                        # Old format: tuple/list with 2 elements - convert to new format
                        elif isinstance(item, (list, tuple)) and len(item) == 2:
                            valid_history.append({
                                'question': item[0],
                                'answer': item[1],
                                'sources': []
                            })
                    return valid_history
    except Exception as e:
        print(f"Error loading chat history: {e}")
    return []

def get_uploaded_files():
    """Get list of uploaded documents"""
    if not os.path.exists(UPLOAD_DIR):
        return []
    return [f for f in os.listdir(UPLOAD_DIR) if f.endswith(('.pdf', '.md', '.txt', '.docx'))]

def upload_document(uploaded_file):
    """Handle document upload and add directly to vector database"""
    try:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Add document directly to database with progress tracking
        success, message = add_to_vectordb([file_path], show_progress=True)
        if success:
            return True, f"Successfully uploaded and indexed {uploaded_file.name}"
        else:
            return True, f"Uploaded {uploaded_file.name} but indexing failed: {message}. You can rebuild the database manually."
    except Exception as e:
        return False, f"Error uploading file: {str(e)}"

def add_to_vectordb(file_paths: list, model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", show_progress=False):
    """Add documents directly to existing vector database or create new one"""
    try:
        if show_progress:
            progress_text = st.sidebar.empty()
            progress_text.info("🔄 Step 1/4: Converting document to markdown...")
        
        # Step 1: Load and convert documents using Docling
        with st.spinner("Converting document...") if show_progress else st.spinner():
            converter = DocumentConverter()
            loader = DoclingLoader(file_paths, converter=converter)
            documents = loader.load()
        
        if not documents:
            if show_progress:
                progress_text.error("❌ No documents loaded from files")
            return False, "No documents loaded from files"
        
        if show_progress:
            progress_text.success("✓ Step 1/4: Document conversion complete")
            progress_text.info("🔄 Step 2/4: Analyzing document structure...")
        
        # Step 2: Analyze and split documents using content-aware splitting
        with st.spinner("Analyzing and chunking document...") if show_progress else st.spinner():
            chunk_size = 1000
            chunk_overlap = 200
            text_splitter = CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len
            )
            chunks = text_splitter.split_documents(documents)
        
        if show_progress:
            progress_text.success(f"✓ Step 2/4: Created {len(chunks)} searchable chunks")
            progress_text.info("🔄 Step 3/4: Generating embeddings...")
        
        # Step 3: Initialize embeddings
        with st.spinner("Generating embeddings...") if show_progress else st.spinner():
            embeddings = HuggingFaceEmbeddings(model_name=model_name)
        
        
        if show_progress:
            progress_text.success("✓ Step 3/4: Embeddings generated")
            progress_text.info("🔄 Step 4/4: Building search index...")
        
        # Step 4: Add to database
        with st.spinner("Building search index...") if show_progress else st.spinner():
            if os.path.exists(CHROMA_DB_DIR) and os.listdir(CHROMA_DB_DIR):
                # Database exists, add to it
                vectorstore = Chroma(
                    persist_directory=CHROMA_DB_DIR,
                    embedding_function=embeddings
                )
                vectorstore.add_documents(filter_complex_metadata(chunks))
            else:
                # Create new database using create_chroma_vectordb from docling_util_Final
                vectorstore = create_chroma_vectordb(
                    file_paths=file_paths,
                    chroma_db_folder=CHROMA_DB_DIR,
                    text_splitter_choice="ContentAwareSplitting",
                    splitter_para={'max_tokens': 800},
                    model_name=model_name
                )
                if vectorstore is None:
                    if show_progress:
                        progress_text.error("❌ Failed to create vector database")
                    return False, "Failed to create vector database"
                if show_progress:
                    progress_text.success("✓ Step 4/4: Search index ready!")
                return True, f"Successfully created database and indexed {len(chunks)} chunks from {len(documents)} document(s)"
        
        if show_progress:
            progress_text.success("✓ Step 4/4: Search index ready!")
        
        return True, f"Successfully indexed {len(chunks)} chunks from {len(documents)} document(s)"
    
    except Exception as e:
        if show_progress:
            st.sidebar.error(f"❌ Error: {str(e)}")
        return False, f"Error adding to vector database: {str(e)}"

def remove_document_from_vectordb(filename: str, model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
    """Remove a document's embeddings from the vector database"""
    try:
        if not os.path.exists(CHROMA_DB_DIR) or not os.listdir(CHROMA_DB_DIR):
            return True, f"Database is empty, {filename} not found in database"
        
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=embeddings
        )
        
        # Get all documents from the database
        collection = vectorstore._collection
        
        # Find and delete documents with matching source filename
        documents = collection.get()
        
        ids_to_delete = []
        if documents and documents.get('metadatas'):
            for idx, metadata in enumerate(documents['metadatas']):
                if metadata and metadata.get('source') and filename in metadata.get('source', ''):
                    ids_to_delete.append(documents['ids'][idx])
        
        # Delete the documents from the collection
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
            return True, f"Removed {len(ids_to_delete)} embeddings for {filename}"
        else:
            return True, f"No embeddings found for {filename} in database"
    
    except Exception as e:
        return False, f"Error removing from database: {str(e)}"

def remove_document(filename):
    """Remove a document from uploaded files and from vector database"""
    try:
        file_path = os.path.join(UPLOAD_DIR, filename)
        
        # Remove from vector database first
        success, db_message = remove_document_from_vectordb(filename)
        
        # Then remove the file
        if os.path.exists(file_path):
            os.remove(file_path)
            return True, f"Successfully removed {filename} from both storage and database"
        return False, "File not found"
    except Exception as e:
        return False, f"Error removing file: {str(e)}"

def rebuild_vectordb():
    """Rebuild the vector database from uploaded documents"""
    try:
        uploaded_files = get_uploaded_files()
        if not uploaded_files:
            return False, "No documents to process"
        
        file_paths = [os.path.join(UPLOAD_DIR, f) for f in uploaded_files]
        
        # Clear existing database
        if os.path.exists(CHROMA_DB_DIR):
            shutil.rmtree(CHROMA_DB_DIR)
        
        # Create new database using ContentAwareSplitting
        vectorstore = create_chroma_vectordb(
            file_paths=file_paths,
            chroma_db_folder=CHROMA_DB_DIR,
            text_splitter_choice="ContentAwareSplitting",
            splitter_para={'max_tokens': 800}
        )
        
        if vectorstore is None:
            return False, "Failed to create vector database"
        
        return True, f"Successfully indexed {len(uploaded_files)} document(s)"
    except Exception as e:
        return False, f"Error building vector database: {str(e)}"

def initialize_session_state():
    """Initialize Streamlit session state"""
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = load_chat_history()
    
    if "db_initialized" not in st.session_state:
        st.session_state.db_initialized = False
    
    if "last_upload_time" not in st.session_state:
        st.session_state.last_upload_time = None
    
    if "selected_files" not in st.session_state:
        st.session_state.selected_files = set()
    
    if "input_counter" not in st.session_state:
        st.session_state.input_counter = 0
    
    if "ratings" not in st.session_state:
        st.session_state.ratings = {}

def load_qa_chain(filter_dict=None):
    """Load the QA chain from saved vector database"""
    try:
        if not os.path.exists(CHROMA_DB_DIR):
            print(f"Chroma DB directory does not exist: {CHROMA_DB_DIR}")
            return None
        
        if not os.listdir(CHROMA_DB_DIR):
            print(f"Chroma DB directory is empty: {CHROMA_DB_DIR}")
            return None
        
        print(f"Loading QA chain from: {CHROMA_DB_DIR}")
        if filter_dict:
            print(f"Applying filter: {filter_dict}")
        
        qa_chain_dict = setup_qa_chain(
            local_vector_store_path=CHROMA_DB_DIR,
            use_local_path=True,
            model_name="llama3.1",
            temperature=0.1,
            embbedings_model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            search_type="mmr",
            k=4,
            fetch_k=20,
            lambda_mult=0.5,
            filter_dict=filter_dict
        )
        qa_chain = qa_chain_dict
        
        if qa_chain_dict is None:
            st.error("Failed to create QA chain - chain is None")
            return None
        
        print("QA chain loaded successfully")
        return qa_chain
    except Exception as e:
        st.error(f"Error loading QA chain: {str(e)}")
        import traceback
        print(f"Full error: {traceback.format_exc()}")
        return None

def main():
    st.set_page_config(
        page_title="Document RAG Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📚 Document RAG Chatbot")
    st.markdown("Upload documents and ask questions about their content")
    
    # Add custom CSS for square rating buttons only
    st.markdown("""
        <style>
        /* Target only buttons in rating-button class - more aggressive */
        .rating-button button,
        .rating-button button[kind="primary"],
        .rating-button button[kind="secondary"],
        .rating-button div[data-testid="stButton"] > button {
            width: 2.5rem !important;
            min-width: 2.5rem !important;
            max-width: 2.5rem !important;
            height: 2.5rem !important;
            min-height: 2.5rem !important;
            max-height: 2.5rem !important;
            padding: 0.2rem !important;
            aspect-ratio: 1 / 1 !important;
        }
        .rating-button button p {
            margin: 0 !important;
            padding: 0 !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    ensure_directories()
    initialize_session_state()
    
    # Sidebar for document management
    with st.sidebar:
        st.header("📋 Document Manager")
        
        # Document upload section
        st.subheader("📤 Upload Documents")
        st.info("ℹ️ Documents are automatically indexed. You can ask questions immediately after upload.")
        uploaded_file = st.file_uploader(
            "Choose a document to upload",
            type=["pdf", "md", "txt", "docx"],
            help="Supported formats: PDF, Markdown, Text, Word",
            key="file_uploader"
        )
        
        if uploaded_file is not None:
            # Check if this file was already processed
            file_already_exists = uploaded_file.name in get_uploaded_files()
            
            if not file_already_exists:
                success, message = upload_document(uploaded_file)
                if success:
                    st.success(message)
                    st.session_state.last_upload_time = datetime.now()
                    # Force reload QA chain with new document
                    st.session_state.qa_chain = None  # Clear existing chain
                    new_chain = load_qa_chain()
                    if new_chain is not None:
                        st.session_state.qa_chain = new_chain
                        st.session_state.db_initialized = True
                        st.info("✅ You may now ask questions about the document using the chatbot on the right.")
                    else:
                        st.warning("Document uploaded but QA chain failed to load. Try refreshing the page.")
                        st.session_state.db_initialized = False
                else:
                    st.error(message)
            else:
                st.info(f"📄 {uploaded_file.name} is already uploaded")
        
        # Display uploaded documents with checkboxes
        st.subheader("📂 Uploaded Documents")
        uploaded_files = get_uploaded_files()
        
        if uploaded_files:
            st.write(f"**Total documents:** {len(uploaded_files)}")
            
            # Select/Deselect all buttons
            col_select1, col_select2 = st.columns(2)
            with col_select1:
                if st.button("✓ Select All", use_container_width=True):
                    st.session_state.selected_files = set(uploaded_files)
                    st.rerun()
            with col_select2:
                if st.button("✗ Deselect All", use_container_width=True):
                    st.session_state.selected_files = set()
                    st.rerun()
            
            st.caption("Select files to use as sources for chat:")
            
            for filename in uploaded_files:
                col1, col2, col3 = st.columns([0.5, 2.5, 1])
                with col1:
                    is_selected = st.checkbox(
                        "",
                        value=filename in st.session_state.selected_files,
                        key=f"select_{filename}",
                        label_visibility="collapsed"
                    )
                    if is_selected:
                        st.session_state.selected_files.add(filename)
                    else:
                        st.session_state.selected_files.discard(filename)
                
                with col2:
                    st.write(f"📄 {filename}")
                
                with col3:
                    if st.button("❌", key=f"remove_{filename}", help="Remove this document"):
                        success, message = remove_document(filename)
                        if success:
                            st.success(message)
                            st.session_state.selected_files.discard(filename)
                            # Reload QA chain to exclude removed document
                            st.session_state.qa_chain = load_qa_chain()
                            st.rerun()
                        else:
                            st.error(message)
            
            if st.session_state.selected_files:
                st.info(f"✓ {len(st.session_state.selected_files)} file(s) selected as sources")
            else:
                st.warning("⚠️ No files selected. Queries will search all documents.")
        else:
            st.info("No documents uploaded yet")
        
        # Database status
        st.divider()
        st.subheader("📊 Database Status")
        if os.path.exists(CHROMA_DB_DIR) and os.listdir(CHROMA_DB_DIR):
            st.success("✅ Vector database ready")
            st.caption("Database is active and ready for queries")
        else:
            st.warning("⚠️ Vector database empty")
            st.caption("Upload documents to populate the database")
        
        # Debug section
        with st.expander("🔧 Debug Info"):
            st.write(f"DB Initialized: {st.session_state.db_initialized}")
            st.write(f"QA Chain exists: {st.session_state.qa_chain is not None}")
            st.write(f"DB Path exists: {os.path.exists(CHROMA_DB_DIR)}")
            if st.button("🔄 Force Reload QA Chain"):
                st.session_state.qa_chain = None
                # Create filter if files are selected
                filter_dict = None
                if st.session_state.selected_files:
                    # Create OR filter for selected files
                    filter_dict = {"source": {"$in": [os.path.join(UPLOAD_DIR, f) for f in st.session_state.selected_files]}}
                st.session_state.qa_chain = load_qa_chain(filter_dict=filter_dict)
                if st.session_state.qa_chain:
                    st.session_state.db_initialized = True
                    st.success("QA Chain reloaded!")
                else:
                    st.error("Failed to reload QA chain")
                st.rerun()
    
    # Main chat area
    main_col = st.container()
    
    # Load QA chain if not already loaded and database exists
    db_exists = os.path.exists(CHROMA_DB_DIR) and os.listdir(CHROMA_DB_DIR)
    
    # Always try to load QA chain if database exists but chain is not loaded
    if st.session_state.qa_chain is None and db_exists:
        with st.spinner("Loading document database..."):
            # Create filter if files are selected
            filter_dict = None
            if st.session_state.selected_files:
                filter_dict = {"source": {"$in": [os.path.join(UPLOAD_DIR, f) for f in st.session_state.selected_files]}}
            
            st.session_state.qa_chain = load_qa_chain(filter_dict=filter_dict)
            if st.session_state.qa_chain is not None:
                st.session_state.db_initialized = True
                print("QA chain loaded and initialized")
            else:
                st.error("Failed to load QA chain. Check if Ollama is running with 'llama3.1' model.")
                st.session_state.db_initialized = False
    
    # If database exists and we have files, we should be ready
    if db_exists and len(get_uploaded_files()) > 0:
        if st.session_state.qa_chain is not None:
            st.session_state.db_initialized = True
    
    # Chat interface
    st.subheader("💬 Ask Questions")
    
    # Show input interface if we have database and QA chain
    can_ask_questions = st.session_state.db_initialized and st.session_state.qa_chain is not None
    
    if not can_ask_questions:
        st.warning("📤 Please upload documents to start asking questions.")
        st.info("💡 Tip: After uploading, your documents will be automatically indexed and you can start asking questions immediately!")
        
        # Show debug info if database exists but QA chain failed
        if db_exists and st.session_state.qa_chain is None:
            st.error("⚠️ Database exists but QA chain failed to load. Click 'Force Reload' in the sidebar debug section.")
    else:
        # Display chat history
        st.write("---")
        
        if st.session_state.chat_history:
            for i, chat_item in enumerate(st.session_state.chat_history):
                # Handle both old format (tuple) and new format (dict)
                if isinstance(chat_item, dict):
                    question = chat_item.get('question', '')
                    answer = chat_item.get('answer', '')
                    sources = chat_item.get('sources', [])
                    rating = chat_item.get('rating', None)
                else:
                    # Old format - just question and answer
                    question, answer = chat_item
                    sources = []
                    rating = None
                
                with st.container():
                    st.markdown(f"**Q:** {question}")
                    st.markdown(f"**A:** {answer}")
                    
                    # Rating system - use columns for inline buttons
                    col_rate = st.columns([0.5, 0.5, 10])
                    
                    # Thumbs up button
                    with col_rate[0]:
                        st.markdown('<div class="rating-button">', unsafe_allow_html=True)
                        # Use type parameter to change button style when selected
                        if rating == 'up':
                            st.button("👍", key=f"thumbs_up_{i}", disabled=True, type="primary")
                        elif st.button("👍", key=f"thumbs_up_{i}", disabled=rating is not None):
                            # Update rating in chat history
                            if isinstance(st.session_state.chat_history[i], dict):
                                st.session_state.chat_history[i]['rating'] = 'up'
                            else:
                                # Convert old format to new format
                                st.session_state.chat_history[i] = {
                                    'question': question,
                                    'answer': answer,
                                    'sources': sources,
                                    'rating': 'up'
                                }
                            save_chat_history()
                            st.session_state['show_feedback'] = ('up', i)
                            st.rerun()
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Thumbs down button
                    with col_rate[1]:
                        st.markdown('<div class="rating-button">', unsafe_allow_html=True)
                        # Use type parameter to change button style when selected
                        if rating == 'down':
                            st.button("👎", key=f"thumbs_down_{i}", disabled=True, type="primary")
                        elif st.button("👎", key=f"thumbs_down_{i}", disabled=rating is not None):
                            # Update rating in chat history
                            if isinstance(st.session_state.chat_history[i], dict):
                                st.session_state.chat_history[i]['rating'] = 'down'
                            else:
                                # Convert old format to new format
                                st.session_state.chat_history[i] = {
                                    'question': question,
                                    'answer': answer,
                                    'sources': sources,
                                    'rating': 'down'
                                }
                            save_chat_history()
                            st.session_state['show_feedback'] = ('down', i)
                            st.rerun()
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Show feedback message or rating status
                    if 'show_feedback' in st.session_state and st.session_state['show_feedback'][1] == i:
                        feedback_type = st.session_state['show_feedback'][0]
                        if feedback_type == 'up':
                            st.success("✅ Thank you for using our chatbot! We're glad we could help.")
                        else:
                            st.warning("📝 Thank you for your feedback. We will continue improving our responses.")
                        # Clear the feedback message after showing
                        del st.session_state['show_feedback']
                    elif rating == 'up':
                        st.success("✅ Thank you for using our chatbot! We're glad we could help.")
                    elif rating == 'down':
                        st.warning("📝 Thank you for your feedback. We will continue improving our responses.")
                    
                    # Display sources if available
                    if sources:
                        with st.expander("📚 View Sources", expanded=False):
                            st.caption("This answer was generated from the following documents:")
                            for idx, source in enumerate(set(sources), 1):
                                # Extract just the filename from the path
                                source_name = os.path.basename(source)
                                
                                # Check if file still exists
                                file_exists = os.path.exists(source)
                                if file_exists:
                                    st.markdown(f"**{idx}.** {source_name}")
                                else:
                                    st.markdown(f"**{idx}.** {source_name} _(removed)_")
                    else:
                        st.caption("_No sources available for this answer_")
                    
                    st.divider()
        
        # Input area
        question = st.text_input(
            "Enter your question:",
            placeholder="What is the main topic of the documents?",
            key=f"question_input_{st.session_state.input_counter}"
        )
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            pass
        
        with col2:
            submit_button = st.button("Ask", use_container_width=True, type="primary")
        
        with col3:
            if st.button("Clear History", use_container_width=True):
                st.session_state.chat_history = []
                # Delete chat history file
                if os.path.exists(CHAT_HISTORY_FILE):
                    os.remove(CHAT_HISTORY_FILE)
                st.rerun()
        
        if submit_button and question:
            # Check if files are selected
            if not st.session_state.selected_files:
                st.warning("⚠️ Please select at least one document from the sidebar to use as a source for your question.")
            else:
                with st.spinner("Searching documents..."):
                    try:
                        # Reload chain with current filter for selected files
                        filter_dict = {"source": {"$in": [os.path.join(UPLOAD_DIR, f) for f in st.session_state.selected_files]}}
                        st.session_state.qa_chain = load_qa_chain(filter_dict=filter_dict)
                        
                        response = ask_question(st.session_state.qa_chain, question)
                        answer = response.get("answer", "No answer found")
                        sources = response.get("sources", [])
                        
                        # Add to chat history with sources (append to end for newest at bottom)
                        st.session_state.chat_history.append({
                            'question': question,
                            'answer': answer,
                            'sources': sources,
                            'rating': None
                        })
                        
                        # Save chat history to file
                        save_chat_history()
                        
                        # Limit chat history to last 20 messages
                        if len(st.session_state.chat_history) > 20:
                            st.session_state.chat_history = st.session_state.chat_history[-20:]
                            save_chat_history()
                        
                        # Force clear the input by using a counter to change the key
                        if "input_counter" not in st.session_state:
                            st.session_state.input_counter = 0
                        st.session_state.input_counter += 1
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error processing question: {str(e)}")
        
        elif submit_button and not question:
            st.warning("Please enter a question")
    
    # Footer
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        total_docs = len(get_uploaded_files())
        selected_docs = len(st.session_state.selected_files)
        st.caption(f"Documents: {total_docs} | Selected: {selected_docs}")
    with col2:
        st.caption(f"Chat history: {len(st.session_state.chat_history)} messages")
    with col3:
        st.caption("Powered by LangChain + Chroma + Llama 3.1")

if __name__ == "__main__":
    main()
