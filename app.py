import os
import streamlit as st

from rag_utility import process_multiple_pdfs, answer_question_with_sources

working_dir=os.getcwd()

st.set_page_config(
    page_title="Multi_PDF chatbot",
      layout="centered")

st.title("Multi_PDF Chatbot")

uploaded_files=st.file_uploader(
    "Upload your PDF Files",
    type=["pdf"],
    accept_multiple_files=True

)
file_names=[]

if uploaded_files:
    for file in uploaded_files:
        save_path=os.path.join(working_dir,file.name)

        with open(save_path, "wb") as f:
            f.write(file.getbuffer())
        
        file_names.append(file.name)

    process_multiple_pdfs(file_names)
    st.info("Documents processed successfully")


    #text widget to get the user input
user_question=st.text_area("ask your question about the document")

if st.button("Answer"):
    answer,sources=answer_question_with_sources(user_question)

    st.markdown("### LLama-3.1-8b")
    st.markdown(answer)
    st.markdown("sources: \n")
    st.markdown(sources)
