import streamlit as st
import pandas as pd
import time 
import random
import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import docling_util
sys.path.append(str(Path(__file__).parent.parent))

# Chat sessions directory
CHAT_SESSIONS_DIR = "chat_sessions"
os.makedirs(CHAT_SESSIONS_DIR, exist_ok=True)

# Lazy import docling_util only when needed
def get_docling_util():
    """Lazy import docling_util to avoid startup errors"""
    import docling_util
    return docling_util

def get_current_session_id():
    """Get or create current session ID"""
    
    # WORKAROUND: Since Streamlit navigation doesn't tell us which page is active,
    # we'll use a manual session selector at the top of the page
    # or detect from a "switch session" action
    
    # Check if explicitly set (from create/delete pages)
    if "session_for_this_page" in st.session_state:
        session_id = st.session_state["session_for_this_page"]
        st.session_state["current_session_id"] = session_id
        # Don't delete it - keep it set
        return session_id
    
    # Fallback to current_session_id
    if "current_session_id" not in st.session_state:
        st.session_state["current_session_id"] = "default"
    
    return st.session_state["current_session_id"]

def get_session_files():
    """Get file paths for current session"""
    session_id = get_current_session_id()
    return {
        "chat_history": os.path.join(CHAT_SESSIONS_DIR, f"{session_id}_chat.json"),
        "uploaded_files": os.path.join(CHAT_SESSIONS_DIR, f"{session_id}_files.json"),
        "input_dir": os.path.join("Company Annual Report", session_id, "input"),
        "output_dir": os.path.join("output_md", session_id)
    }

# --- 1. Sidebar Content (File Upload Feature) ---
def render_sidebar():
    """Renders the file upload feature in the sidebar."""
    
    # Session selector at the top
    st.sidebar.markdown("### 💬 Switch Chat")
    
    all_sessions = st.session_state.get("all_available_sessions", ["default"])
    chat_names_map = st.session_state.get("chat_names_map", {})
    current_session = get_current_session_id()
    
    # Create display names for selector
    session_options = []
    session_display_map = {}
    for session_id in all_sessions:
        if session_id == "default":
            display_name = "Default Chat"
        else:
            display_name = chat_names_map.get(session_id, f"Chat {session_id[-6:]}")
        session_options.append(display_name)
        session_display_map[display_name] = session_id
    
    # Find current selection
    current_display = "Default Chat"
    for display, sid in session_display_map.items():
        if sid == current_session:
            current_display = display
            break
    
    # Session selector
    selected_display = st.sidebar.selectbox(
        "Select Chat",
        options=session_options,
        index=session_options.index(current_display) if current_display in session_options else 0,
        key="session_selector"
    )
    
    # Update session if changed
    selected_session = session_display_map[selected_display]
    if selected_session != current_session:
        st.session_state["current_session_id"] = selected_session
        st.session_state["session_for_this_page"] = selected_session
        st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📁 Upload Files")
    
    session_files = get_session_files()
    
    # Initialize uploaded files list in session state
    if "uploaded_files" not in st.session_state:
        # Try to reload files from saved list
        if os.path.exists(session_files["uploaded_files"]):
            try:
                with open(session_files["uploaded_files"], 'r', encoding='utf-8') as f:
                    saved_files = json.load(f)
                
                # Recreate file objects from saved files in session input directory
                uploaded_files = []
                for file_info in saved_files:
                    file_path = os.path.join(session_files["input_dir"], file_info['name'])
                    if os.path.exists(file_path):
                        # Read the file and create a file-like object
                        with open(file_path, 'rb') as f:
                            file_data = f.read()
                        
                        # Create a mock uploaded file object
                        from io import BytesIO
                        mock_file = BytesIO(file_data)
                        mock_file.name = file_info['name']
                        mock_file.size = file_info['size']
                        uploaded_files.append(mock_file)
                
                st.session_state["uploaded_files"] = uploaded_files
            except Exception as e:
                st.session_state["uploaded_files"] = []
        else:
            st.session_state["uploaded_files"] = []
    
    # Initialize file uploader counter for unique keys
    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 0
    
    st.sidebar.title("Input File Upload")
    st.sidebar.markdown("<small style='color: gray;'>Upload files one by one for analysis.</small>", unsafe_allow_html=True)
    
    uploaded_file = st.sidebar.file_uploader(
        "Choose a file",
        type=['csv', 'txt', 'xlsx', 'mb', 'pdf', 'docx', 'pptx', 'jpg', 'jpeg', 'png', 'tiff', 'bmp'], 
        accept_multiple_files=False,
        key=f"file_uploader_{st.session_state['uploader_key']}"
    )
    
    if uploaded_file is not None:
        # Check if this file is already in the list (by name)
        existing_names = [f.name for f in st.session_state["uploaded_files"]]
        
        if uploaded_file.name not in existing_names:
            # Save file to session-specific directory
            os.makedirs(session_files["input_dir"], exist_ok=True)
            os.makedirs(session_files["output_dir"], exist_ok=True)
            
            # Save uploaded file
            file_path = os.path.join(session_files["input_dir"], uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Check file extension to determine processing method
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            # Check if file has already been processed (caching)
            md_output_path = os.path.join(session_files["output_dir"], f"{Path(uploaded_file.name).stem}.md")
            already_processed = os.path.exists(md_output_path)
            
            # Process file based on type
            if file_extension in ['txt', 'csv', 'mb']:
                # Text files don't need conversion - they're already readable
                st.sidebar.success(f"✓ Uploaded text file: **{uploaded_file.name}**")
                # Add text file directly to ChromaDB if it exists
                try:
                    with st.spinner("Adding to database..."):
                        get_docling_util().add_document_to_chroma(file_path)
                except Exception as e:
                    st.sidebar.warning(f"Could not add to database: {str(e)}")
            elif already_processed:
                # File was already processed - use cached version
                st.sidebar.info(f"⚡ Using cached version of **{uploaded_file.name}**")
                try:
                    docling = get_docling_util()
                    progress_text = st.sidebar.empty()
                    
                    progress_text.info("🔄 Adding cached file to database...")
                    with st.spinner("Loading from cache..."):
                        docling.add_document_to_chroma(md_output_path)
                    progress_text.success("✓ Cached file loaded successfully")
                    
                    st.sidebar.success(f"✅ **{uploaded_file.name}** ready for chat! (from cache)")
                except Exception as e:
                    st.sidebar.error(f"Error loading cached file: {str(e)}")
                    # If cache fails, we could reprocess, but for now just show error
            else:
                # Process documents (PDF, DOCX, PPTX, images) to markdown
                try:
                    docling = get_docling_util()
                    
                    # Step 1: Convert to markdown
                    progress_text = st.sidebar.empty()
                    progress_text.info("🔄 Step 1/3: Converting to markdown...")
                    with st.spinner("Converting document..."):
                        md_file = docling.process_single_document_to_md(file_path, "no image annotation")
                    progress_text.success("✓ Step 1/3: Markdown conversion complete")
                    
                    # Step 2: Analyze images with AI
                    progress_text.info("🔄 Step 2/3: Analyzing images with AI...")
                    with st.spinner("Analyzing images..."):
                        docling.describe_images_and_update(md_file)
                    progress_text.success("✓ Step 2/3: Image analysis complete")
                    
                    # Step 3: Add to ChromaDB
                    progress_text.info("🔄 Step 3/3: Adding to database...")
                    with st.spinner("Building search index..."):
                        docling.add_document_to_chroma(md_file)
                    progress_text.success("✓ Step 3/3: Database updated")
                    
                    st.sidebar.success(f"✅ **{uploaded_file.name}** ready for chat!")
                except Exception as e:
                    st.sidebar.error(f"Error processing file: {str(e)}")
            
            # Add the new file to the list
            st.session_state["uploaded_files"].append(uploaded_file)
            
            # Save uploaded files list to persist across refreshes
            try:
                files_info = [{"name": f.name, "size": f.size} for f in st.session_state["uploaded_files"]]
                with open(session_files["uploaded_files"], 'w', encoding='utf-8') as f:
                    json.dump(files_info, f, ensure_ascii=False, indent=2)
            except Exception as e:
                pass
            
            st.sidebar.success(f"Added: **{uploaded_file.name}**")
        else:
            st.sidebar.info(f"**{uploaded_file.name}** already uploaded")
    
    # Display list of uploaded files
    if st.session_state["uploaded_files"]:
        st.sidebar.markdown("---")
        st.sidebar.subheader(f"📁 Uploaded Files ({len(st.session_state['uploaded_files'])})")
        
        files_to_delete = []
        
        for idx, file in enumerate(st.session_state["uploaded_files"]):
            file_size_mb = file.size / (1024 * 1024)
            col1, col2 = st.sidebar.columns([4, 2])
            
            with col1:
                st.sidebar.text(f"{idx + 1}. {file.name} ({file_size_mb:.2f} MB)")
            
            with col2:
                if st.sidebar.button("Delete", key=f"delete_{idx}_{file.name}", type="secondary"):
                    files_to_delete.append((idx, file.name))
        
                # Delete files after iteration to avoid modification during iteration
                if files_to_delete:
                    for idx, filename in sorted(files_to_delete, reverse=True):
                        # 1. Delete from ChromaDB
                        try:
                            docling = get_docling_util()
                            # Delete markdown version if it exists
                            base_name = filename.rsplit('.', 1)[0]
                            md_filename = f"{base_name}.md"
                            docling.delete_document_from_chroma(md_filename)
                            # Also try to delete original if it was added directly
                            docling.delete_document_from_chroma(filename)
                        except Exception as e:
                            st.sidebar.warning(f"Could not remove from database: {str(e)}")
                        
                        # 2. Delete physical files
                        try:
                            # Delete from input folder
                            input_file = os.path.join("Company Annual Report/input", filename)
                            if os.path.exists(input_file):
                                os.remove(input_file)
                            
                            # Delete markdown file
                            md_file = os.path.join("output_md", f"{base_name}.md")
                            if os.path.exists(md_file):
                                os.remove(md_file)
                            
                            # Delete artifacts folder if it exists
                            artifacts_folder = os.path.join("output_md", "output_md", f"{base_name}_artifacts")
                            if os.path.exists(artifacts_folder):
                                import shutil
                                shutil.rmtree(artifacts_folder)
                        except Exception as e:
                            st.sidebar.warning(f"Could not delete physical files: {str(e)}")
                        
                        # 3. Remove from session state
                        st.session_state["uploaded_files"].pop(idx)
                    
                    # Save updated files list
                    try:
                        session_files = get_session_files()
                        files_info = [{"name": f.name, "size": f.size} for f in st.session_state["uploaded_files"]]
                        with open(session_files["uploaded_files"], 'w', encoding='utf-8') as f:
                            json.dump(files_info, f, ensure_ascii=False, indent=2)
                    except Exception as e:
                        pass
                    
                    # Change uploader key to reset the file uploader
                    st.session_state["uploader_key"] += 1
                    st.rerun()
    else:
        st.sidebar.info("No files uploaded yet")


# --- 2. Main Page Content (Chat Application) ---

# Response generator that processes uploaded files
def response_generator(user_prompt):
    """Generates a response based on the uploaded files content."""
    
    uploaded_files = st.session_state.get("uploaded_files", [])
    
    # Keywords that trigger file analysis
    analysis_keywords = ["analyze", "analyse", "read", "check", "examine", "inspect", 
                        "columns", "rows", "show", "display", "content", "data",
                        "preview", "summary", "what's in", "tell me about", "describe"]
    
    # Check if user wants file analysis
    wants_analysis = any(keyword in user_prompt.lower() for keyword in analysis_keywords)
    
    if not uploaded_files:
        # Friendly greeting/chat response when no files
        greetings = ["hello", "hi", "hey", "greetings"]
        if any(greeting in user_prompt.lower() for greeting in greetings):
            response = "Hello! I'm here to help you analyze files. Upload a file using the sidebar and I can help you understand its contents!"
        else:
            response = "I'm ready to help! Upload a file using the sidebar, and I can analyze it for you when you're ready."
    elif not wants_analysis:
        # Engage in friendly conversation when files are uploaded but user doesn't request analysis
        num_files = len(uploaded_files)
        greetings = ["hello", "hi", "hey", "greetings"]
        
        if any(greeting in user_prompt.lower() for greeting in greetings):
            response = f"Hello! I see you've uploaded {num_files} file(s). When you're ready, I can analyze them for you. Just ask me to 'analyze', 'show', or 'check' the files!"
        elif "help" in user_prompt.lower():
            response = f"I can help you analyze your {num_files} uploaded file(s)! Try asking: 'analyze the file', 'show columns', 'how many rows?', or 'display the content'."
        else:
            response = f"I see you have {num_files} file(s) uploaded. I can analyze them whenever you're ready! Just let me know what you'd like to know about them."
    else:
        # User requested file analysis - proceed with analysis
        target_file = None
        for file in uploaded_files:
            if file.name.lower() in user_prompt.lower():
                target_file = file
                break
        
        # If no specific file mentioned and multiple files, handle accordingly
        if target_file is None and len(uploaded_files) > 1:
            if "list" in user_prompt.lower() or "files" in user_prompt.lower() or "all" in user_prompt.lower():
                response = f"You have {len(uploaded_files)} files uploaded:\n\n"
                for idx, file in enumerate(uploaded_files, 1):
                    file_type = file.name.split('.')[-1].upper()
                    response += f"{idx}. {file.name} ({file_type}, {file.size / (1024*1024):.2f} MB)\n"
                response += "\nMention a specific filename to analyze it!"
            else:
                # Use the first file by default
                target_file = uploaded_files[0]
                response = f"Analyzing '{target_file.name}' (you have {len(uploaded_files)} files):\n\n"
        elif target_file is None:
            target_file = uploaded_files[0]
            response = ""
        else:
            response = ""
        
        # Process the target file if one is selected
        if target_file:
            file_extension = target_file.name.split('.')[-1].lower()
            
            # Check if markdown version exists (for processed files)
            base_name = target_file.name.rsplit('.', 1)[0]
            md_file_path = f"output_md/{base_name}.md"
            
            try:
                # Check if processed markdown file exists
                if os.path.exists(md_file_path):
                    with open(md_file_path, 'r', encoding='utf-8') as f:
                        md_content = f.read()
                    
                    # Extract key information from markdown
                    lines = md_content.split('\n')
                    word_count = len(md_content.split())
                    char_count = len(md_content)
                    
                    # Get first few non-empty lines as preview
                    preview_lines = [line for line in lines[:20] if line.strip()]
                    preview = '\n'.join(preview_lines[:10])
                    
                    if "content" in user_prompt.lower() or "show" in user_prompt.lower() or "display" in user_prompt.lower():
                        response += f"**{target_file.name}** (processed to markdown):\n\n{preview}\n\n... (showing first 10 lines)"
                    elif "summary" in user_prompt.lower():
                        response += f"**{target_file.name}** summary:\n- Format: Converted to Markdown\n- Words: {word_count:,}\n- Characters: {char_count:,}\n- Available in: output_md/{base_name}.md"
                    else:
                        response += f"**{target_file.name}** has been processed:\n- Converted to markdown format\n- Words: {word_count:,}\n- Characters: {char_count:,}\n- Location: output_md/{base_name}.md\n\nAsk me to 'show content' or 'display' to see the text!"
                
                elif file_extension == 'csv':
                    target_file.seek(0)
                    df = pd.read_csv(target_file)
                    
                    if "columns" in user_prompt.lower() or "headers" in user_prompt.lower():
                        response += f"**{target_file.name}** has {len(df.columns)} columns: {', '.join(df.columns.tolist())}"
                    elif "rows" in user_prompt.lower() or "many" in user_prompt.lower():
                        response += f"**{target_file.name}** has {len(df)} rows"
                    elif "show" in user_prompt.lower() or "display" in user_prompt.lower() or "preview" in user_prompt.lower():
                        response += f"**{target_file.name}** preview:\n\n{df.head().to_string()}"
                    else:
                        response += f"**{target_file.name}**:\n- Rows: {len(df)}\n- Columns: {len(df.columns)}\n- Column names: {', '.join(df.columns.tolist())}"
                        
                elif file_extension in ['txt', 'mb']:
                    target_file.seek(0)
                    content = target_file.getvalue().decode("utf-8")
                    
                    if "content" in user_prompt.lower() or "show" in user_prompt.lower() or "display" in user_prompt.lower():
                        preview = content[:500] + "..." if len(content) > 500 else content
                        response += f"**{target_file.name}** content:\n\n{preview}"
                    elif "length" in user_prompt.lower() or "size" in user_prompt.lower():
                        response += f"**{target_file.name}** has {len(content)} characters"
                    else:
                        char_count = len(content)
                        word_count = len(content.split())
                        response += f"**{target_file.name}**:\n- Characters: {char_count}\n- Words: {word_count}\n- File type: Text"
                else:
                    response += f"**{target_file.name}**: {file_extension.upper()} file ({target_file.size / (1024*1024):.2f} MB). Processing in progress..."
                    
            except Exception as e:
                response = f"Error processing {target_file.name}: {str(e)}"
    
    # Stream the response
    for word in response.split():
        yield word + " "
        time.sleep(0.02)


def render_chat_app():
    """Renders the main chat interface."""
    
    st.title("Chatbot.com")
    
    # Detect if session has changed and reload data
    current_session = get_current_session_id()
    

    
    if "last_loaded_session" not in st.session_state or st.session_state["last_loaded_session"] != current_session:
        # Session changed - clear cached data to force reload
        if "messages" in st.session_state:
            del st.session_state["messages"]
        if "uploaded_files" in st.session_state:
            del st.session_state["uploaded_files"]
        st.session_state["last_loaded_session"] = current_session
    
    # Show file count
    num_files = len(st.session_state.get("uploaded_files", []))
    if num_files > 0:
        st.caption(f"📂 {num_files} file(s) uploaded and ready for analysis")
    else:
        st.caption("💬 Upload files using the sidebar to get started")

    # Initialize chat history
    session_files = get_session_files()
    if "messages" not in st.session_state:
        # Try to load chat history from session-specific file
        if os.path.exists(session_files["chat_history"]):
            try:
                with open(session_files["chat_history"], 'r', encoding='utf-8') as f:
                    st.session_state.messages = json.load(f)
            except:
                st.session_state.messages = []
        else:
            st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Type here to start chatting..."):
        
        # 1. Add user message to chat history and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Display assistant response in chat message container
        with st.chat_message("assistant"):
            response = st.write_stream(response_generator(prompt))
            
        # 3. Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # 4. Save chat history to session-specific file
        try:
            session_files = get_session_files()
            with open(session_files["chat_history"], 'w', encoding='utf-8') as f:
                json.dump(st.session_state.messages, f, ensure_ascii=False, indent=2)
        except Exception as e:
            pass  # Silently fail if can't save


# --- 3. Run the Application ---

if __name__ == "__main__":
    # Render the sidebar first
    render_sidebar()
    # Then render the main chat interface
    render_chat_app()