import os
from dotenv import load_dotenv


from langchain_community.document_loaders import PyPDFLoader

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA


##load environment variables from .env file
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

working_dir= os.path.dirname(os.path.abspath((__file__)))

embedding=HuggingFaceEmbeddings()

llm=ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0,
    api_key=groq_api_key 
)

#Ingestion: 
def process_multiple_pdfs(file_list):
    all_docs=[]
    
    for file_name in file_list:
        # loader= UnstructuredPDFLoader(f"{working_dir}/{file_name}")
        # documents=loader.load()

        loader = PyPDFLoader(os.path.join(working_dir, file_name))
        documents = loader.load()

        #add metadata 
        for doc in documents: 
            doc.metadata["source"] = file_name
        
        all_docs.extend(documents)

        #split: 
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
    texts=text_splitter.split_documents(all_docs)

    #store in Chromadb: 
    vectordb=Chroma.from_documents(
            documents=texts,
            embedding=embedding,
            persist_directory=f"{working_dir}/doc_vectorstore"
        )
    return 0

def answer_question_with_sources(user_question):
    vectordb=Chroma(
        persist_directory=f"{working_dir}/doc_vectorstore",
        embedding_function=embedding
    )
    retriever= vectordb.as_retriever()

    qa_chain=RetrievalQA.from_chain_type(
        llm=llm,
        chain_type= "stuff",
        retriever=retriever,
        return_source_documents=True 
    )
    response=qa_chain.invoke({"query":user_question})
    answer=response["result"]

    #extract sources
    source_docs= response["source_documents"]

    sources=list(set([
        doc.metadata.get("source", "Unknown")
        for doc in source_docs
    ]))
    
    return answer, sources
