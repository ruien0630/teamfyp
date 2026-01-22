import streamlit as st
import query_util_baseline as query_util
import os

# Set page configuration
st.set_page_config(
    page_title="Financial RAG Chatbot",
    page_icon="💼",
    layout="wide"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "qa_chain" not in st.session_state:
    # Auto-initialize with default credentials
    os.environ["WATSONX_APIKEY"] = "wmYPwxoE3VrEUxHjPoBfQGc9_bWoq2tAlBmwtyfD4byr"
    os.environ["IBM_PROJECT_ID"] = "fdeaf360-1c36-44a4-8cd1-1b5183b6ace4"
    try:
        st.session_state.qa_chain = query_util.setup_qa_chain("./chroma_db")
        st.session_state.initialized = True
    except:
        st.session_state.qa_chain = None
        st.session_state.initialized = False
if "initialized" not in st.session_state:
    st.session_state.initialized = False

# Sidebar for quick actions
with st.sidebar:
    st.title("Quick Actions")
    
    # Status display
    if st.session_state.initialized:
        st.success("✅ RAG System Ready")
    else:
        st.error("❌ RAG System Not Initialized")
        if st.button("🔄 Retry Initialization"):
            try:
                st.session_state.qa_chain = query_util.setup_qa_chain("./chroma_db")
                st.session_state.initialized = True
                st.rerun()
            except Exception as e:
                st.error(f"Failed: {e}")
    
    st.divider()
    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    # Status
    st.divider()
    st.caption("Status")
    if st.session_state.qa_chain:
        st.success("🟢 RAG System Ready")
    else:
        st.warning("🟡 RAG System Not Initialized")

# Main chat interface
st.title("💼 Financial RAG Chatbot")
st.caption("Ask questions about your financial annual reports")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display sources if available
        if "sources" in message and message["sources"]:
            with st.expander("📚 Sources"):
                for idx, source in enumerate(message["sources"], 1):
                    st.caption(f"{idx}. {source}")

# Chat input
if prompt := st.chat_input("Ask a question about the financial reports..."):
    if not st.session_state.qa_chain:
        st.error("❌ Please initialize the RAG system first using the sidebar.")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = query_util.ask_question(st.session_state.qa_chain, prompt)
                    
                    # Display answer
                    st.markdown(response["answer"])
                    
                    # Display sources
                    if response["sources"]:
                        with st.expander("📚 Sources"):
                            for idx, source in enumerate(response["sources"], 1):
                                st.caption(f"{idx}. {source}")
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response["sources"]
                    })
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

# Footer
st.divider()
st.caption("Powered by IBM Watsonx and ChromaDB")
