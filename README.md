# Pubmed-Chatbot-App

PubMed paper abstracts chatbot application utilizing Google Gemini's API

## Example

```text
System:
[cari2 prompt medical expert]
[cari2 prompt medical researcher expert]
[cari2 prompt comprehensive explanation]
[cari2 prompt bullet points explanation]

User:
saya mengalami gejala batuk dan pilek. bagaimana cara mengatasinya?

AI:
berikut beberapa metode
```

## Chatbot Features

- medical-specialized prompt
- pubmed scraper for RAG
- pdf-option for context

## RAG Diagram

1. User gives keywords for the backend to search through relevant papers
2. The backend scraps X number of papers, preprocessed into pandas dataframe. The dataframe contains the following columns/criteria:

    - pubmed id
    - title
    - abstract
    - contents
    - background
    - related works
    - methodology
    - experiment result and analysis
    - conclusion
    - other details like journal, authors, etc.

3. The dataframe is filtered based on the keywords using cosine similarity on the embeddings of abstract/contents with regards to the embedding of the keywords.

    - the number of filtered papers can be chosen arbitrarily, but let's say it is X//10

4. Now build the vectorstore. For each row, take its corresponding paper title and abstract/content. And then concatenate it into one single giant string.

    - raw text splitter: `CharacterTextSplitter(separator="/n", chunk_size=2000, chunk_overlap=200, length_function=len)`
    - vectorstore: `FAISS.from_texts(texts=text_chunks, embedding=...)`, where the embedding is either `GoogleGenerativeAIEmbeddings` or `OpenAIEmbeddings`
    - build conversation chain using `ConversationBufferMemory(memory_key="chat_history", return_messages=True)` and `ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)` where LLM is either `ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)` or `ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)`
    - the conversation chain is stored within streamlit's `session_state`

5. The backend then can receive chat message using `st.session_state.conversation({"question": user_query})`
