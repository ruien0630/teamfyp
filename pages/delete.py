import streamlit as st

st.title("🗑️ Delete Entry")
st.write("Select an entry to delete from the system.")

# Example deletion interface
entry_name = st.text_input("Entry name to delete")

if st.button("Delete", type="primary"):
    if entry_name:
        st.warning(f"⚠️ Deleted entry: {entry_name}")
    else:
        st.error("Please enter an entry name")

