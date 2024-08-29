# Pubmed-Chatbot-App

PubMed paper abstracts chatbot application utilizing Google Gemini's API for document filtering

## Installation

1. First create a project folder (example project folder name: PROJECT_NAME) and go inside the folder
2. Clone the repository

```
    git clone https://github.com/rinogrego/Pubmed-Chatbot-App/
```

3. Initiate python virtual environment

```
    python -m venv venv
```

4. Install the necessary packages

```
    pip install -r requirements.txt
```

5. Create a new `.env` file to store the environment keys and fill the following necessary keys:

```
    GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY
    OPENAI_API_KEY=YOUR_OPENAI_API_KEY

    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
    LANGCHAIN_API_KEY=YOUR_LANGCHAIN_API_KEY
    LANGCHAIN_PROJECT=YOUR_LANGCHAIN_PROJECT_NAME
```

6. Run the web server using the following code

```
    streamlit run src/app.py
```

## Chatbot Features

- Pubmed Scraper for RAG
  - `retmax`: parameter for number of abstracts scrapped from PubMed
- Filter scrapped abstracts using Cosine Similarity of Google's Embedding
  - `pubmed_num_docs_similarity`: parameter for Top K number of similar documents according to query
- Retrieval-Augmented Generation on abstracts that have been filtered
  - currently the retriever is only set with `k=3` and use `mmr` search type
- PDF-option for RAG [NOT IMPLEMENTED]
- Medical-specialized prompt [NOT IMPLEMENTED]

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

## Problems

- Given a question, retrieving an entire paper can be noisy. The relevant answer might be only in result section or background section.
  - idea:
    - for each chunk after splitting, also provide the title from its source. might be good to also provide the corresponding page
    - design a prompt template that can provide context, title, and page and have the LLM answer based on that context -> reduce token usage
- How to make the LLM give answer while providing the source?
  - idea:
    - design a SYSTEM_PROMPT using few-shot prompt to provide answer and citation