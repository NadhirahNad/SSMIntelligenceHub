import streamlit as st
from backend import ask_ssm_bot
from huggingface_hub import login
from dotenv import load_dotenv
import os
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


st.set_page_config(page_title="SSM Intelligence Hub", layout="wide")

st.image("LOGO_SSM.png", width=150)

st.title("SSM Intelligence Hub")
st.markdown("Ask questions about the *Registration of Businesses Act 1956* and the *Registration of Companies Act 2016*.")

with st.form("ask_form"):
    query = st.text_input("🔍 Enter your legal question:", placeholder="e.g. Is ROB 1956 applied to whole Malaysia?")
    submit = st.form_submit_button("Ask")

if submit:
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("🔎 Retrieving legal information..."):
            answer, confidence, retrieved = ask_ssm_bot(query)

        st.success("✅ Answer Generated")
        st.markdown(f"**Answer:** {answer}")
        st.markdown(f"**Confidence Score:** {confidence}")

        st.divider()
        st.subheader("📂 Retrieved Contexts")
        for item in retrieved:
            tag = "🗑️ **This section is deleted.**" if item['is_deleted'] else ""
            st.markdown(
                f"""
                **Act**: {item['act']}  
                **Section**: {item['section']}  
                **Title**: {item['title']}  
                **Similarity Score**: {item['similarity']:.4f}  
                {tag}
                """
            )
            st.code(item["content"], language="markdown")
