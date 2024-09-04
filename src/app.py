import streamlit as st

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

from pypdf import PdfReader
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.retrievers import EnsembleRetriever

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
st.subheader("Chat with your own files.. or scrap papers from PubMed")
st.markdown("---")

use_openai = True

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

def get_text_chunks(text: str):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n\n", "\n\n", "\n"],
        chunk_size=4000,
        chunk_overlap=400,
        length_function=len
    )
    docs = text_splitter.create_documents([text])
    docs = text_splitter.split_documents(documents=docs)
    return docs

def get_retriever(docs, k_docs_for_rag):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
    if use_openai and os.getenv("OPENAI_API_KEY") is not None:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
    
    # from: https://medium.com/@nadikapoudel16/advanced-rag-implementation-using-hybrid-search-reranking-with-zephyr-alpha-llm-4340b55fef22
    retriever_vectordb = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": k_docs_for_rag})
    keyword_retriever = BM25Retriever.from_documents(docs)
    keyword_retriever.k =  k_docs_for_rag
    retriever = EnsembleRetriever(
        retrievers = [retriever_vectordb, keyword_retriever],
        weights = [0.5, 0.5]
    )
    return retriever

def get_conversation_chain(retriever, openai_api_key=None):
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro", google_api_key=os.getenv("GOOGLE_API_KEY"), 
        # ValueError: SystemMessages are not yet supported! To automatically convert the leading SystemMessage to a HumanMessage, set `convert_system_message_to_human` to True. Example: llm = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)
        convert_system_message_to_human=True 
    )
    if use_openai and os.getenv("OPENAI_API_KEY") is not None:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        input_key="question",
        output_key="answer", # because issue: https://github.com/langchain-ai/langchain/issues/2303#issuecomment-1508973646
        return_messages=True
    )
    
    ### SYSTEM PROMPT
    # from: https://github.com/langchain-ai/langchain/issues/5462#issuecomment-1569923207
    system_template = """Use the following pieces of context to answer the users question. 
    If you cannot find the answer from the pieces of context, just say that you don't know, don't try to make up an answer.
    The context given is a scrapped paper abstract and its title, given a number format. Always cite the source of your answer by using [number of paper].
    ----------------
    {context}"""
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)
    ### CONDENSE QUESTION PROMPT
    # from: https://github.com/langchain-ai/langchain/issues/4076#issuecomment-1563138403
    condense_question_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. Preserve the original question in the answer sentiment during rephrasing.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    condense_question_prompt = PromptTemplate.from_template(condense_question_template)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = retriever,
        memory = memory,
        return_source_documents = True,
        combine_docs_chain_kwargs = {"prompt": qa_prompt},
        condense_question_prompt = condense_question_prompt
        
    )
    return conversation_chain

def load_abstracts_from_pubmed(
    keywords: str, 
    retmax: int = 10
) -> pd.DataFrame:
    df = scrap_pubmed(query=keywords, retmax=retmax)
    
    # bar = st.progress(0, text="Performing similarity search based on the query...")
    # # embed documents
    # def embed_fn(title, text):
    #     return GoogleGenerativeAIEmbeddings(
    #         model="models/embedding-001",
    #         task_type="retrieval_document",
    #         google_api_key=os.getenv('GOOGLE_API_KEY')
    #     ).embed_query(text=text)
    # df["Abstract Embeddings"] = df.apply(lambda row: embed_fn(row["Title"], row["Abstract"]), axis=1)
    
    # query_embedding = GoogleGenerativeAIEmbeddings(
    #     model="models/embedding-001", task_type="retrieval_query", google_api_key=os.getenv('GOOGLE_API_KEY')
    # )
    # qe = query_embedding.embed_query(keywords)
    # import numpy as np
    # dot_products = np.dot(np.stack(df['Abstract Embeddings']), qe)
    # idx = np.argsort(-dot_products)[:doc_num] # sort indexes from index with the highest value to the lowest
    # bar.empty()
    # # df.columns: ['Title', 'Abstract', 'Journal', 'Language', 'Year', 'Month', 'PMID', 'DOI']
    # return df.iloc[idx][['Title', 'Abstract', 'Journal', 'Year', 'Month', 'PMID', 'DOI']] # Return text with the associated indexes
    return df[['Title', 'Abstract', 'Journal', 'Year', 'Month', 'PMID', 'DOI']]

def main():
    if "conversation" not in st.session_state:
        # print("Initialize conversation")
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        # print("Initialize chat_history")
        st.session_state.chat_history = []
    if "pubmed_papers_keywords" not in st.session_state:
        # print("Initialize pubmed_papers_keywords and pubmed_papers_scrap_results")
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
        # st.markdown("---")
        # st.subheader("Model Option")
        # openai_api_key = st.text_input("**Decide your OpenAI model by providing your API:**")
        st.markdown("---")
        if document_choice == "My Own Files":
            st.subheader("File Uploader")
            pdf_docs = st.file_uploader("**Upload your PDFs here and click on Process**", accept_multiple_files=True)
            st.write("**WARNING**: The citation prompt is still bad")
            if pdf_docs:
                if st.button("Process"):
                    with st.spinner("Processing"):
                        if st.session_state.conversation is None:
                            raw_text, documents = get_pdf_text(pdf_docs)
                            text_chunks = get_text_chunks(raw_text)
                            retriever = get_retriever(text_chunks, k_docs_for_rag = 5)
                            st.session_state.conversation = get_conversation_chain(retriever)
                        
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
            col1_sidebar, col2_sidebar = st.columns(2)
            with col1_sidebar:
                pubmed_retmax = st.radio("Retmax", key="pubmed_retmax", options=[10, 20, 50, 100])
            with col2_sidebar:
                k_docs_for_rag = st.radio("RAG contexts", key="k_docs_for_rag", options=[5, 10, 20, 50])
            if search_query:
                st.subheader("Search query entered")
                st.write(
                    "Parameter config:<br>Entrez's retmax: {}<br>Num. of docs for RAG contexts: {}".format(pubmed_retmax, k_docs_for_rag),
                    unsafe_allow_html=True
                )
                with st.spinner("Searching Relevant Papers"):
                    if search_query not in st.session_state.pubmed_papers_keywords:
                        df_title_abstracts = load_abstracts_from_pubmed(
                            search_query, 
                            retmax=pubmed_retmax,
                        )
                        st.session_state.pubmed_papers_keywords.append(search_query)
                        st.session_state.pubmed_papers_scrap_results[search_query] = {
                            "df_title_abstracts": df_title_abstracts
                        } 
                    else:
                        df_title_abstracts = st.session_state.pubmed_papers_scrap_results[search_query]["df_title_abstracts"]
                    paper_titles = df_title_abstracts["Title"]
                    abstracts = df_title_abstracts["Abstract"]
                    journals = df_title_abstracts['Journal']
                    years = df_title_abstracts['Year']
                    months = df_title_abstracts['Month']
                    pmids = df_title_abstracts['PMID']
                    dois = df_title_abstracts['DOI']
                    for idx, (title, abs, journal, year, month, pmid, doi) in enumerate(zip(paper_titles, abstracts, journals, years, months, pmids, dois)):
                        st.write(
                            f"""<h3>[{idx}] {title}</h3>
                            <p>{abs}</p>
                            <p>
                                Journal : {journal} <br>
                                Date &emsp;&nbsp;: {month}, {year} <br>
                                PMID &emsp;: <a href="https://pubmed.ncbi.nlm.nih.gov/{pmid}/">{pmid}</a><br>
                                doi&emsp;&emsp;: <a href="https://www.doi.org/{doi}">{doi}</a>
                            </p>
                            <hr>""", 
                            unsafe_allow_html=True
                        )
                    
                    # constructing raw text for contexts to create vectorstore
                    # NOTE: may change this method into create a Document object for each paper and its metadata
                    raw_text = ""
                    for idx, (title, abs) in enumerate(zip(paper_titles, abstracts)):
                        raw_text += f"[{idx}] Title: {title}\nAbstract:\n{abs}\n\n"
                    # somehow this part is reloaded whenever doing chat so implement checking to prevent resetting session_state
                    if st.session_state.conversation is None:
                        docs = get_text_chunks(raw_text)
                        retriever = get_retriever(docs, k_docs_for_rag = k_docs_for_rag)
                        st.session_state.conversation = get_conversation_chain(retriever)

if __name__ == "__main__":
    main()