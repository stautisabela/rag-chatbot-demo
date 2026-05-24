import streamlit as st
from document_processor import process_file, reset_store
from rag_pipeline import set_api_key, retrieve_and_answer

# App Title
st.markdown("<h1 style='text-align: center;'>📄 Local Document QA System 🧠</h1>", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("### 🛠️ Setup")

# Enter OpenAI API Key
api_key = st.sidebar.text_input("🔑 Enter your OpenAI API Key", type="password")
if api_key:
    set_api_key(api_key)

# Reset Button
if st.sidebar.button("🗑️ Clear Existing Documents"):
    reset_store()
    st.sidebar.success("✅ All documents cleared. You can upload new ones.")

# Document Upload Section
st.sidebar.markdown("### 📂 Upload Documents")
uploaded_files = st.sidebar.file_uploader("📤 Upload PDFs", accept_multiple_files=True, type=["pdf"])

if uploaded_files:
    st.sidebar.write("🔄 Processing files...")
    for uploaded_file in uploaded_files:
        process_file(uploaded_file)
    st.sidebar.success("✅ Documents processed successfully!")

# Question-Answering Section
st.markdown("### 💬 Ask a Question")
query = st.text_input("💡 Type your question here:")

if query and api_key:
    st.markdown("### 🧠 Answer")
    answer, sources = retrieve_and_answer(query)
    st.write(answer)
    st.markdown("### 🔗 Relevant Sources")
    for i, source in enumerate(sources):
        st.write(f"📘 **Source {i+1}:** {source[:300]}...")  # max 300 char
elif query:
    st.warning("⚠️ Please provide your OpenAI API Key in the sidebar.")
