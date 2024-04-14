import streamlit as st

from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

from pypdf import PdfReader
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter

import os
from dotenv import load_dotenv

from typing import Tuple, List
from utils.scrap_pubmed_papers import scrap_pubmed
import pandas as pd

load_dotenv()

st.set_page_config(
    page_title="Chatbot",
    layout="centered"
)

st.title("Document Chatbot")
st.subheader("Chat with your own files.. or scrap papers from Pubmed")
st.markdown("---")

use_openai = False
    
def get_pdf_text(pdf_docs) -> Tuple[str, Document]:
    # from: https://stackoverflow.com/a/76816979/13283654
    # process pdf_docs to langchain's Document object
    docs = []
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text()
            text += page_text
            docs.append(Document(page_content=page_text, metadata={'page': page.page_number}))
    return text, docs

def get_text_chunks(text: str) -> List[str]:
    text_splitter = CharacterTextSplitter(
        separator="/n",
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks: List[str]):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
    if use_openai:
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro", google_api_key=os.getenv("GOOGLE_API_KEY"), 
        # ValueError: SystemMessages are not yet supported! To automatically convert the leading SystemMessage to a HumanMessage, set `convert_system_message_to_human` to True. Example: llm = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)
        convert_system_message_to_human=True 
    )
    if use_openai:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        # input_key="question", 
        # output_key="answer", 
        return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def compose_keywords(query: str) -> str:
    # use llm to create keywords to answer question related
    keywords = ["sugar", "treatment", "young adults"]
    return keywords

def load_abstracts_from_pubmed(keywords: str, doc_num: int = 50) -> pd.DataFrame:
    df = scrap_pubmed(query=keywords)
    # pubmed_abstracts = df["Abstract"]
    
    bar = st.progress(0, text="Performing similarity search based on the query...")
    # embed documents
    def embed_fn(title, text):
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            task_type="retrieval_document",
            google_api_key=os.getenv('GOOGLE_API_KEY')
        ).embed_query(text=text)
    df["Abstract Embeddings"] = df.apply(lambda row: embed_fn(row["Title"], row["Abstract"]), axis=1)
    
    query_embedding = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", task_type="retrieval_query", google_api_key=os.getenv('GOOGLE_API_KEY')
    )
    qe = query_embedding.embed_query(keywords)
    import numpy as np
    dot_products = np.dot(np.stack(df['Abstract Embeddings']), qe)
    idx = np.argsort(-dot_products)[:doc_num] # sort indexes from index with the highest value to the lowest
    bar.empty()
    return df.iloc[idx][['Title', 'Abstract']] # Return text with the associated indexes

def main():
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "pubmed_papers_keywords" not in st.session_state:
        st.session_state.pubmed_papers_keywords = []
        st.session_state.pubmed_papers_scrap_results = {}
    
    # user input
    user_query = st.chat_input("What's on your mind?")
    if user_query is not None and user_query != "":
        response = st.session_state.conversation({"question": user_query})
        st.session_state.chat_history = response["chat_history"]
            
    # display conversation
    for idx, message in enumerate(st.session_state.chat_history):
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.markdown(message.content)
        
    with st.sidebar:
        st.header("Upload Your Documents\nor Use Search through Pubmed Papers")
        document_choice = st.radio(
            "**Document choice**",
            ["My Own Files", "Pubmed Papers"],
            captions = ["Upload your own documents", "Search through pubmed papers"]
        )
        st.markdown("---")
        if document_choice == "My Own Files":
            st.subheader("File Uploader")
            pdf_docs = st.file_uploader("**Upload your PDFs here and click on Process**", accept_multiple_files=True)
            if pdf_docs:
                if st.button("Process"):
                    with st.spinner("Processing"):
                        raw_text, documents = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        vectorstore = get_vectorstore(text_chunks)
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                        
                        # # print relevant page
                        # page_num = 5
                        # for doc in documents:
                        #     if doc.metadata["page"] == page_num:
                        #         page_doc = f"<h4>Page {page_num}</h4>{doc.page_content}"
                        #         break
                        #     else:
                        #         page_doc = "This document don't have that page number"
                        # st.write(page_doc, unsafe_allow_html=True)
                        # st.markdown("---")
                    
        elif document_choice == "Pubmed Papers":
            st.subheader("Search Relevant Pubmed Papers")
            search_query = st.text_input("**Enter your keywords:**", placeholder="high sugar intake for young adult")
            if search_query:
                st.subheader("Search query entered")
                with st.spinner("Searching Relevant Papers"):
                    if search_query not in st.session_state.pubmed_papers_keywords:
                        df_title_abstracts = load_abstracts_from_pubmed(search_query, doc_num=10)
                        paper_titles = df_title_abstracts["Title"]
                        abstracts = df_title_abstracts["Abstract"]
                        st.session_state.pubmed_papers_keywords.append(search_query)
                        st.session_state.pubmed_papers_scrap_results[search_query] = {
                            "paper_titles": paper_titles,
                            "abstracts": abstracts
                        } 
                    else:
                        paper_titles = st.session_state.pubmed_papers_scrap_results[search_query]["paper_titles"]
                        abstracts = st.session_state.pubmed_papers_scrap_results[search_query]["abstracts"]
  
                    for idx, (title, abs) in enumerate(zip(paper_titles, abstracts)):
                        st.write(f"Paper Title {idx}: <h3>{title}</h3><p>{abs}</p><hr>", unsafe_allow_html=True)
                    
                    raw_text = ""
                    for idx, (title, abs) in enumerate(zip(paper_titles, abstracts)):
                        raw_text += f"Title {idx}: {title}\nAbstract: {abs}\n\n"
                    
                    # somehow this part is reloaded whenever doing chat so implement checking to prevent resetting session_state
                    if st.session_state.conversation is None:
                        text_chunks = get_text_chunks(raw_text)
                        vectorstore = get_vectorstore(text_chunks)
                        st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == "__main__":
    main()