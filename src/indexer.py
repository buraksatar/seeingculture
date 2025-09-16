# Import required libraries
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
import tiktoken
from tqdm import tqdm

def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..",))

load_dotenv(dotenv_path=os.path.join(get_project_root(), '.env'))

# Check for API key
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY environment variable not found. "
        "Please set this variable in your environment or .env file."
    )

def count_tokens_and_cost(text, model="text-embedding-ada-002", encoding="text-davinci-003"):
    encoding = tiktoken.encoding_for_model(encoding)
    num_tokens = len(encoding.encode(text))
    if model == "text-embedding-ada-002":
        cost = (num_tokens / 1000) * 0.00001  # $0.00001 per 1K tokens
    return num_tokens, cost

def load_data_from_csv(data_path):
    df = pd.read_csv(data_path)
    
    required_columns = ["Question", "Rationale", "Concept", "Category", "Country", "Image Path", "Object"]
    for col in required_columns:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in CSV. This may cause issues.")
    
    if "Status" in df.columns:
        df = df.loc[(df['Status'].isin(['ACC']))]
        print(len(df))
    
    
    df = df[required_columns].dropna()
    return df

def create_faiss_index(data_path, index_path="faiss_index"):
    df = load_data_from_csv(data_path)

    df["text"] = df.apply(
        lambda x: str(x["Question"])
        + " "
        + str(x["Rationale"] if pd.notna(x["Rationale"]) else ""),
        axis=1,
    )

    embeddings = OpenAIEmbeddings()
    texts = df["text"].tolist()
    metadatas = df[["Concept", "Category", "Country", "Image Path", "Rationale"]].to_dict("records")

    print("Embedding documents...")
    text_embeddings = []
    token_counts = 0
    costs = 0
    for text, metadata in tqdm(zip(texts, metadatas), total=len(texts), desc="Creating embeddings"):
        try:
            tokens, cost = count_tokens_and_cost(str(text))
            token_counts += tokens
            costs += cost

            embedding = embeddings.embed_query(str(text))
            text_embeddings.append((text, embedding))
        except Exception as e:
            print(f"Error embedding text: {e}")
            continue

    # Create FAISS index and save locally
    vectorstore = FAISS.from_embeddings(
        text_embeddings, 
        embeddings, 
        metadatas=metadatas[: len(text_embeddings)]
    )
    vectorstore.save_local(index_path)
    
    print(f"FAISS index created and saved to {index_path}")
    print(f"Total tokens used: {token_counts}")
    print(f"Total cost: ${costs:.4f}")
    
    return df, vectorstore

def initialize_vectorstore(data_path, index_path="faiss_index"):
    """Initialize or load the FAISS index"""
    embeddings = OpenAIEmbeddings()
    
    if os.path.exists(index_path):
        print(f"Loading existing FAISS index from {index_path}...")
        vectorstore = FAISS.load_local(
            index_path, 
            embeddings,
            allow_dangerous_deserialization=True
        )
        df = load_data_from_csv(data_path)
        return df, vectorstore
    else:
        print(f"No existing index found at {index_path}. Creating new index...")
        return create_faiss_index(data_path, index_path)
