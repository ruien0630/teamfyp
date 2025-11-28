import streamlit as st
import pandas as pd
import time 
import random

# --- 1. Sidebar Content (File Upload Feature) ---
def render_sidebar():
    """Renders the file upload feature in the sidebar."""
    
    # Initialize uploaded files list in session state
    if "uploaded_files" not in st.session_state:
        st.session_state["uploaded_files"] = []
    
    # Initialize file uploader counter for unique keys
    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 0
    
    st.sidebar.title("Input File Upload")
    st.sidebar.markdown("Upload files one by one for analysis.")
    
    uploaded_file = st.sidebar.file_uploader(
        "Choose a file",
        type=['csv', 'txt', 'xlsx', 'mb', 'pdf'], 
        accept_multiple_files=False,
        key=f"file_uploader_{st.session_state['uploader_key']}"
    )
    
    if uploaded_file is not None:
        # Check if this file is already in the list (by name)
        existing_names = [f.name for f in st.session_state["uploaded_files"]]
        
        if uploaded_file.name not in existing_names:
            # Add the new file to the list
            st.session_state["uploaded_files"].append(uploaded_file)
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
                    files_to_delete.append(idx)
        
                # Delete files after iteration to avoid modification during iteration
                if files_to_delete:
                    for idx in sorted(files_to_delete, reverse=True):
                        st.session_state["uploaded_files"].pop(idx)
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
            
            try:
                if file_extension == 'csv':
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
                    response += f"**{target_file.name}**: {file_extension.upper()} file ({target_file.size / (1024*1024):.2f} MB). Format not fully supported yet."
                    
            except Exception as e:
                response = f"Error processing {target_file.name}: {str(e)}"
    
    # Stream the response
    for word in response.split():
        yield word + " "
        time.sleep(0.02)


def render_chat_app():
    """Renders the main chat interface."""
    
    st.title("Chatbot.com")
    
    # Show file count
    num_files = len(st.session_state.get("uploaded_files", []))
    if num_files > 0:
        st.caption(f"📂 {num_files} file(s) uploaded and ready for analysis")
    else:
        st.caption("💬 Upload files using the sidebar to get started")

    # Initialize chat history
    if "messages" not in st.session_state:
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


# --- 3. Run the Application ---

if __name__ == "__main__":
    # Render the sidebar first
    render_sidebar()
    # Then render the main chat interface
    render_chat_app()