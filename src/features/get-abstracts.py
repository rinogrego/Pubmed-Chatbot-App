from langchain_google_genai import GoogleGenerativeAIEmbeddings
import pandas as pd
import numpy as np

from dotenv import load_dotenv
import os

load_dotenv()
google_api_key=os.getenv("GOOGLE_API_KEY")

def load_abstracts_from_pubmed(keywords):
    # search through relevant
    df = pd.DataFrame({
        "Title": ["A", "B", "very high sugar", "Kid", "Coronavirus", "Low Sugar", "High Sugar"],
        "Abstract": ["AAA", "BBBB", "Really High Sugar", "Play with me", "Covid 19", "Low sugar causes", "High sugar causes"]
    })
    
    # embed documents
    def embed_fn(title, text):
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            task_type="retrieval_document",
            google_api_key=google_api_key
        ).embed_query(text=text)
    df["Abstract Embeddings"] = df.apply(lambda row: embed_fn(row["Title"], row["Abstract"]), axis=1)
    
    query_embedding = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", task_type="retrieval_query", google_api_key=google_api_key
    )
    qe = query_embedding.embed_query(keywords)
    dot_products = np.dot(np.stack(df['Abstract Embeddings']), qe)
    idx = np.argsort(-dot_products) # sort indexes from index with the highest value to the lowest
    return df.iloc[idx]['Abstract'].to_list() # Return text with the associated indexes

query_text = "effect of low sugar"
print(query_text)
print(load_abstracts_from_pubmed(query_text))