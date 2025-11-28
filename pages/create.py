import streamlit as st

st.title("➕ Create Entry")
st.write("Create a new entry in the system.")

# Example form
with st.form("create_form"):
    name = st.text_input("Name")
    description = st.text_area("Description")
    submitted = st.form_submit_button("Create")
    
    if submitted:
        st.success(f"✓ Created entry: {name}")
        st.write(f"Description: {description}")

