
import streamlit as st
import sys
from pathlib import Path

# Set the session ID for this specific chat page
st.session_state["current_session_id"] = "chat_20251204_210347"
st.session_state["session_for_this_page"] = "chat_20251204_210347"

# Import and run the main page logic
sys.path.insert(0, str(Path(__file__).parent.parent))
from pages.main_page import render_sidebar, render_chat_app

# Render the chat interface
render_sidebar()
render_chat_app()
