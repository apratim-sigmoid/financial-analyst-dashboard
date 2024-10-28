import io
import pdfplumber
import pathlib
import openai
import numpy as np
import tempfile
import os
import re
import pandas as pd

# Libraries for Web Apps
import tiktoken
import streamlit as st

# Libraries for Text Data Extraction
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings
from pymilvus import MilvusClient
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from rank_bm25 import BM25Okapi

@st.cache_data(persist=True)
def process_document_with_file_search(api_key=None, model=None, file_streams=None):
    """
    Description: Create a vector DB using Openai's File Search
    """

    file_streams = np.array([file_streams]).flatten().tolist()
    file_streams = [io.BufferedReader(file) for file in file_streams]

    # Initialize Openai Client
    client = openai.OpenAI(
        api_key=api_key,
        timeout=None
    )

    # Create an assistant
    assistant = client.beta.assistants.create(
        name="Financial Analyst Assistant",
        instructions="You are an expert financial analyst. Use you knowledge base to answer questions about financial reports/data.",
        model=model,
        tools=[{"type": "file_search"}],
    )

    # Create a vector store caled "Financial Statements"
    vector_store = client.beta.vector_stores.create(name="Financial Statements")

    file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vector_store.id, files=file_streams
    )

    # Print the status and the file counts of the batch to see the result of this operation.
    print(file_batch.status)
    print(file_batch.file_counts)

    # Update the assistant to to use the new Vector Store
    assistant = client.beta.assistants.update(
        assistant_id=assistant.id,
        tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
    )

    # Storing Client, Assistant, File-Batch
    st.session_state["client"] = client
    st.session_state["file_batch"] = file_batch
    st.session_state["assistant"] = assistant

    return


# @st.cache(persist=True)
def process_document(input_choice, file, api_key):
    """Extracts text and store the documents into vector DB"""
    
    if input_choice=="Document":        
        return process_and_store_data(file, api_key)

    elif input_choice=="File Directory":
        words, string_data, tokens = extract_directory_files(file)
        return None, None, None, None


@st.cache_data(persist=True)
def extract_data_with_pypdf(feed):
    if pathlib.Path(feed.name).suffix.lower() == '.pdf':
        # Save the uploaded file content to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(feed.getvalue())
            tmp_file_path = tmp_file.name
        
        loader = PyPDFLoader(tmp_file_path)
        pages = loader.load_and_split()

    return pages

@st.cache_data(persist=True)
def extract_data_with_csvloader(feed):
    if pathlib.Path(feed.name).suffix.lower() == '.csv':
        # Save the uploaded file content to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            tmp_file.write(feed.getvalue())
            tmp_file_path = tmp_file.name
        
        loader = CSVLoader(tmp_file_path)
        csv_data = loader.load_and_split()

    return csv_data

# @st.cache_data(persist=True)
def process_and_store_data(feed, api_key):
    """Function to extract text from uploaded file"""
    
    if pathlib.Path(feed.name).suffix.lower() == '.pdf':
        pages = extract_data_with_pypdf(feed)

        if "vector_db" not in os.listdir():
            os.makedirs("vector_db")

        # Create vector database
        db = MilvusClient("./vector_db/temp_.db")

        # Collection name
        collection_name = "demo_collection"

        # Drop the existing collection if it exists
        if collection_name in db.list_collections():
            db.drop_collection(collection_name)

        # Initialize a new collection
        db.create_collection(collection_name=collection_name, dimension=1536)

        embedding_fn = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small")
        vectors = embedding_fn.embed_documents([d.page_content for d in pages])
        data = [{
                "id": i, "vector": vectors[i], "text": pages[i].page_content, 
                "source": pages[i].metadata["source"], "page": pages[i].metadata["page"]
            } for i in range(len(vectors))
        ]

        res = db.insert(
            collection_name="demo_collection",
            data=data
        )

        return "pdf", pages, db, embedding_fn
    if pathlib.Path(feed.name).suffix.lower() == '.csv':
        # Load data as dataframe
        df = pd.read_csv(feed)

        # Update column names to include underscore instead of whitespace
        for col in df.columns:
            df.rename({col: col.replace(" ", "_")}, axis=1, inplace=True)
        
        # Create folder if it is not present in directory
        if "sql_db" not in os.listdir():
            os.makedirs("sql_db")
        
        # Delete old file
        if "temp_.db" in os.listdir("sql_db/"):
            os.remove("sql_db/temp_.db")
        
        # Initialize DB
        engine = create_engine("sqlite:///sql_db/temp_.db")
        df.to_sql("temp_", engine, index=False)

        db = SQLDatabase(engine=engine)
        
        # Create bm25 object
        csv_data = extract_data_with_csvloader(feed)
        bm25 = create_bm25_object(csv_data)
        
        return "csv", csv_data, (db, bm25), None        

@st.cache_data(persist=True)
def extract_directory_files(feed):
    """Function to extract data when multiple files from a directory is provided. Allowed extensions are: .pdf"""
    text = ""

    for file in feed:
        if file.name.endswith(".pdf"):
            with pdfplumber.open(file) as pdf: 
                pages = pdf.pages 
                for p in pages: 
                    text += p.extract_text() 

    words = len(text.split())
    tokens = num_tokens_from_string(text, encoding_name="cl100k_base")

    return words, text, tokens

def num_tokens_from_string(string: str, encoding_name="cl100k_base") -> int:
    """Function to count number of tokens in a text string"""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string))


def preprocess_text_for_bm25(text):
    # This pattern matches any character that is not a letter, number, or space
    pattern = r'[^a-zA-Z0-9\s]'
    formatted_text = re.sub(pattern, ' ', text).lower()
    return formatted_text.split(' ')

def create_bm25_object(data):
    """Creates a bm25 object to store the document corpus."""
    # Sample documents
    corpus = [doc_.page_content.replace("\n", " ") for doc_ in data]

    # Tokenize the documents
    tokenized_corpus = [preprocess_text_for_bm25(doc) for doc in corpus]

    # Create a BM25 object
    bm25 = BM25Okapi(tokenized_corpus)

    return bm25