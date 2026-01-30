import streamlit as st
import os
import shutil
from pathlib import Path
from dotenv import load_dotenv
import tempfile
from datetime import datetime

# Import local utilities
from docling_util import create_chroma_vectordb
from query_util import setup_qa_chain, ask_question
from langchain_chroma import Chroma
from langchain_docling import DoclingLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata
from docling.document_converter import DocumentConverter

# Load environment variables
load_dotenv()

# Constants
UPLOAD_DIR = "./uploaded_documents"
CHROMA_DB_DIR = "./chroma_db"
SESSION_UPLOAD_DIR = None

def ensure_directories():
    """Ensure necessary directories exist"""
    Path(UPLOAD_DIR).mkdir(exist_ok=True)
    Path(CHROMA_DB_DIR).mkdir(exist_ok=True)

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
        
        # Add document directly to database
        success, message = add_to_vectordb([file_path])
        if success:
            return True, f"Successfully uploaded and indexed {uploaded_file.name}"
        else:
            return True, f"Uploaded {uploaded_file.name} but indexing failed: {message}. You can rebuild the database manually."
    except Exception as e:
        return False, f"Error uploading file: {str(e)}"

def add_to_vectordb(file_paths: list, model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
    """Add documents directly to existing vector database or create new one"""
    try:
        # Step 1: Load documents using Docling
        converter = DocumentConverter()
        loader = DoclingLoader(file_paths, converter=converter)
        documents = loader.load()
        
        if not documents:
            return False, "No documents loaded from files"
        
        # Step 2: Split documents
        chunk_size = 1000
        chunk_overlap = 200
        text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        
        # Step 3: Initialize embeddings
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        
        # Ensure the directory for Chroma DB exists
        if not os.path.exists(CHROMA_DB_DIR):
            os.makedirs(CHROMA_DB_DIR)
        
        # Step 4: Add to database
        if os.path.exists(CHROMA_DB_DIR) and os.listdir(CHROMA_DB_DIR):
            # Database exists, add to it
            vectorstore = Chroma(
                persist_directory=CHROMA_DB_DIR,
                embedding_function=embeddings
            )
            vectorstore.add_documents(filter_complex_metadata(chunks))
        else:
            # Create new database using create_chroma_vectordb from docling_util
            vectorstore = create_chroma_vectordb(
                file_paths=file_paths,
                chroma_db_folder=CHROMA_DB_DIR,
                text_splitter_choice="CharacterTextSplitter",
                splitter_para={'chunk_size': chunk_size, 'chunk_overlap': chunk_overlap},
                model_name=model_name
            )
            if vectorstore is None:
                return False, "Failed to create vector database"
            return True, f"Successfully created database and indexed {len(chunks)} chunks from {len(documents)} document(s)"
        
        return True, f"Successfully indexed {len(chunks)} chunks from {len(documents)} document(s)"
    
    except Exception as e:
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
        
        # Create new database
        vectorstore = create_chroma_vectordb(
            file_paths=file_paths,
            chroma_db_folder=CHROMA_DB_DIR,
            text_splitter_choice="CharacterTextSplitter",
            splitter_para={'chunk_size': 1000, 'chunk_overlap': 200}
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
        st.session_state.chat_history = []
    
    if "db_initialized" not in st.session_state:
        st.session_state.db_initialized = False
    
    if "last_upload_time" not in st.session_state:
        st.session_state.last_upload_time = None

def load_qa_chain():
    """Load the QA chain from saved vector database"""
    try:
        if not os.path.exists(CHROMA_DB_DIR):
            print(f"Chroma DB directory does not exist: {CHROMA_DB_DIR}")
            return None
        
        if not os.listdir(CHROMA_DB_DIR):
            print(f"Chroma DB directory is empty: {CHROMA_DB_DIR}")
            return None
        
        print(f"Loading QA chain from: {CHROMA_DB_DIR}")
        qa_chain = setup_qa_chain(
            local_vector_store_path=CHROMA_DB_DIR,
            use_local_path=True,
            model_id="llama3.1",
            embbedings_model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
        
        if qa_chain is None:
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
                with st.spinner(f"Uploading and indexing {uploaded_file.name}..."):
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
                            st.info("✅ You can now ask questions about your documents below!")
                        else:
                            st.warning("Document uploaded but QA chain failed to load. Try refreshing the page.")
                            st.session_state.db_initialized = False
                    else:
                        st.error(message)
            else:
                st.info(f"📄 {uploaded_file.name} is already uploaded")
        
        # Display uploaded documents
        st.subheader("📂 Uploaded Documents")
        uploaded_files = get_uploaded_files()
        
        if uploaded_files:
            st.write(f"**Total documents:** {len(uploaded_files)}")
            
            for filename in uploaded_files:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"📄 {filename}")
                with col2:
                    if st.button("❌", key=f"remove_{filename}", help="Remove this document"):
                        success, message = remove_document(filename)
                        if success:
                            st.success(message)
                            # Reload QA chain to exclude removed document
                            st.session_state.qa_chain = load_qa_chain()
                            st.rerun()
                        else:
                            st.error(message)
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
                st.session_state.qa_chain = load_qa_chain()
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
            st.session_state.qa_chain = load_qa_chain()
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
            for i, (question, answer) in enumerate(st.session_state.chat_history):
                with st.container():
                    st.markdown(f"**Q:** {question}")
                    st.markdown(f"**A:** {answer}")
                    st.divider()
        
        # Input area
        st.write("---")
        question = st.text_input(
            "Enter your question:",
            placeholder="What is the main topic of the documents?",
            key="question_input"
        )
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            pass
        
        with col2:
            submit_button = st.button("Ask", use_container_width=True, type="primary")
        
        with col3:
            if st.button("Clear History", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        
        if submit_button and question:
            with st.spinner("Searching documents..."):
                try:
                    response = ask_question(st.session_state.qa_chain, question)
                    answer = response.get("answer", "No answer found")
                    
                    # Add to chat history
                    st.session_state.chat_history.insert(0, (question, answer))
                    
                    # Limit chat history to last 20 messages
                    if len(st.session_state.chat_history) > 20:
                        st.session_state.chat_history = st.session_state.chat_history[:20]
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error processing question: {str(e)}")
        
        elif submit_button and not question:
            st.warning("Please enter a question")
    
    # Footer
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption(f"Documents: {len(get_uploaded_files())}")
    with col2:
        st.caption(f"Chat history: {len(st.session_state.chat_history)} messages")
    with col3:
        st.caption("Powered by LangChain + Chroma + IBM Granite")

if __name__ == "__main__":
    main()
