import streamlit as st

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

from pypdf import PdfReader
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

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
    # text_splitter = CharacterTextSplitter(
    #     separator="/n",
    #     chunk_size=2000,
    #     chunk_overlap=200,
    #     length_function=len
    # )
    # chunks = text_splitter.split_text(text)
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n\n", "\n\n", "\n"],
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len
    )
    docs = text_splitter.create_documents([text])
    docs = text_splitter.split_documents(documents=docs)
    # print(type(docs))
    # print(len(docs))
    # print(type(docs[0]))
    # print(docs[0])
    return docs

def get_vectorstore(docs):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
    if use_openai and os.getenv("OPENAI_API_KEY") is not None:
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    # vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
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
    condense_question_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. Preserve the original question in the answer setiment during rephrasing.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    condense_question_prompt = PromptTemplate.from_template(condense_question_template)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        condense_question_prompt=condense_question_prompt
        
    )
    return conversation_chain
    
# def compose_keywords(query: str) -> str:
#     # use llm to create keywords to answer question related
#     keywords = ["sugar", "treatment", "young adults"]
#     return keywords

# def retrieve_relevant_paper():
#     # get relevant paper inside database based on query
#     return ""

# def retrieve_relevant_texts_from_paper(text: str, query: str):
#     # given a text/paper, retrieve k texts that are most likely to answer given query
#     return ""

def load_abstracts_from_pubmed(
    keywords: str, 
    doc_num: int = 10,
    retmax: int = 10
) -> pd.DataFrame:
    df = scrap_pubmed(query=keywords, retmax=retmax)
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
    # df.columns: ['Title', 'Abstract', 'Journal', 'Language', 'Year', 'Month', 'PMID', 'DOI']
    return df.iloc[idx][['Title', 'Abstract', 'Journal', 'Year', 'Month', 'PMID', 'DOI']] # Return text with the associated indexes

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
            col1_sidebar, col2_sidebar = st.columns(2)
            with col1_sidebar:
                pubmed_retmax = st.radio("Retmax", key="pubmed_retmax", options=[10, 20, 100, 200, 500, 1000])
            with col2_sidebar:
                pubmed_num_docs_similarity = st.radio("Filter by Similarity", key="pubmed_num_docs_similarity", options=[10, 20, 50, 100])
            if search_query:
                st.subheader("Search query entered")
                st.write(
                    "Parameter config:<br>Entrez's retmax: {}<br>Num. of docs to filter by similarity: {}".format(pubmed_retmax, pubmed_num_docs_similarity),
                    unsafe_allow_html=True
                )
                with st.spinner("Searching Relevant Papers"):
                    if search_query not in st.session_state.pubmed_papers_keywords:
                        df_title_abstracts = load_abstracts_from_pubmed(
                            search_query, 
                            doc_num=pubmed_num_docs_similarity,
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
                        vectorstore = get_vectorstore(docs)
                        st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == "__main__":
    main()