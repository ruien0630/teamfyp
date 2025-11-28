import streamlit as st

# Define the pages
create_page = st.Page("create.py", title="Create entry", icon=":material/add_circle:")
delete_page = st.Page("delete.py", title="Delete entry", icon=":material/delete:")
main_page = st.Page("main_page.py", title="Chatbot.com")

# Set up navigation with sections
pg = st.navigation({
    "": [create_page, delete_page],
    "Pages": [main_page]
})

st.set_page_config(page_title="Data manager", page_icon=":material/edit:")

# Run the selected page
pg.run()


