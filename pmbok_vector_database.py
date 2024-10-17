# RETRIEVAL-AUGMENTED GENERATION (RAG) BBUILDER

######
# - Langchain Document Loaders
# - Text Embeddings
# - Vector Databases

# LIBRARIES 

from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings


import pandas as pd
import yaml
from pprint import pprint

from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from huggingface_hub import login

import chromadb

from langchain_community.document_loaders import DirectoryLoader
from transformers import AutoTokenizer, AutoModel
import os
import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter

#from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# pip install PyMuPDF solves fitz problem
# OPENAI API SETUP

#OPENAI_API_KEY = yaml.safe_load(open('credentials.yml'))['openai']
HUGGING_FACE_api= yaml.safe_load(open('credentials.yml'))['huggingface_api']
#OPENAI_API_KEY = yaml.safe_load(open('credentials.yml'))['gemini_api']




# Use your Hugging Face token here
token = HUGGING_FACE_api
login(token)



# 1.0 DATA PREPARATION ----

pdf_directory = "data/pdf_folder" 
chroma_db_path = "data/chroma_store"

# Ensure the ChromaDB path exists and has write permissions
if not os.path.exists(chroma_db_path):
    os.makedirs(chroma_db_path)
    os.chmod(chroma_db_path, 0o775)

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text("text")
    return text

# Load PDFs and extract text
documents = []
for filename in os.listdir(pdf_directory):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_directory, filename)
        text = extract_text_from_pdf(pdf_path)
        documents.append({"text": text, "source": pdf_path})

# Define the HuggingFace model for embeddings
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)

model = AutoModel.from_pretrained(embedding_model_name)

# # Define a function to create embeddings
def create_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.detach().numpy()

# Initialize the HuggingFaceEmbeddings
#hf_embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
hf_embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

#embedding = OpenAIEmbeddings(model='text-embedding-ada-002',api_key=HUGGING_FACE_api)

# Use a text splitter to handle large documents
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

# Split texts and create embeddings
#split_texts = [chunk for doc in documents for chunk in splitter.split_text(doc["text"])]
#print(type(split_texts))

#embeddings = hf_embeddings.embed_documents(split_texts)
#embeddings = hf_embeddings.embed_query(split_texts)

##############################################
# Initialize ChromaDB with the correct embedding function
   #chroma = Chroma(embedding_function=hf_embeddings.embed_documents, persist_directory=chroma_db_path)

# Split texts first if needed, then generate embeddings
#split_texts = [chunk for doc in documents for chunk in splitter.split_text(doc["text"])]
#embeddings = hf_embeddings.embed_documents(split_texts)  # Create embeddings for the split texts

# Add texts and their embeddings along with metadata
# Ensure the metadata aligns with the correct documents and split texts
#metadatas = [{'source': doc['source']} for doc in documents]

# Store texts, embeddings, and metadata in ChromaDB (Aj version)
#chroma.add_texts(texts=split_texts, embeddings=embeddings, metadatas=metadatas)
#chroma.persist()  # Persist the database to the specified directory

#################################################Another approach
# Initialize ChromaDB with a lambda function that calls the embedding method
chroma = Chroma(embedding_function=lambda texts: hf_embeddings.embed_documents(texts), persist_directory=chroma_db_path)

# Generate embeddings (manually or using Chroma)
split_texts = [chunk for doc in documents for chunk in splitter.split_text(doc["text"])]
embeddings = hf_embeddings.embed_documents(split_texts)

# Prepare metadata for the texts
metadatas = [{'source': doc['source']} for doc in documents for _ in splitter.split_text(doc["text"])]

# Add the texts and embeddings to ChromaDB
chroma.add_texts(texts=split_texts, embeddings=embeddings, metadatas=metadatas)

# Persist the database to disk
chroma.persist()

# mnmn     

################################alternatively to set chroma.add.text
from langchain.vectorstores import Chroma

# Initialize ChromaDB with the HuggingFaceEmbeddings object
chroma = Chroma(embedding_function=hf_embeddings, persist_directory=chroma_db_path)

# Prepare texts to be split
split_texts = [chunk for doc in documents for chunk in splitter.split_text(doc["text"])]

# Add texts directly to Chroma, it will handle embeddings internally
chroma.add_texts(texts=split_texts, metadatas=[{'source': doc['source']} for doc in documents for _ in splitter.split_text(doc["text"])])

# Persist the database
chroma.persist()



##############################################
# Create and store embeddings in ChromaDB
#chroma = Chroma(embedding_function = hf_embeddings.embed_documents, persist_directory=chroma_db_path)
#chroma.add_texts(texts=split_texts, embeddings= embeddings, metadatas=[{'source': doc['source']} for doc in documents])
#chroma.persist()

result = chroma.similarity_search("discuss briefly on item based recommender system?", k = 4)

pprint(result[0].page_content)


# # youtube_df = pd.read_csv('data/youtube_videos.csv')

# # youtube_df.head()

# # # * Text Preprocessing

# # youtube_df['page_content'] = youtube_df['page_content'].str.replace('\n\n', '\n', regex=False)

# # * Document Loaders
# #   https://python.langchain.com/docs/integrations/document_loaders/pandas_dataframe 

# loader = DataFrameLoader(youtube_df, page_content_column='page_content')

# documents = loader.load()

# documents[0].metadata
# documents[0].page_content

# pprint(documents[0].page_content)

# len(documents)

# # * Text Splitting
# #   https://python.langchain.com/docs/modules/data_connection/document_transformers

# CHUNK_SIZE = 1000

# # Character Splitter: Splits on simple default of 
# text_splitter = CharacterTextSplitter(
#     chunk_size=CHUNK_SIZE, 
#     # chunk_overlap=100,
#     separator="\n"
# )

# docs = text_splitter.split_documents(documents)

# docs[0].metadata

# len(docs)

# # Recursive Character Splitter: Uses "smart" splitting, and recursively tries to split until text is small enough
# text_splitter_recursive = RecursiveCharacterTextSplitter(
#     chunk_size = CHUNK_SIZE,
#     chunk_overlap=100,
# )

# docs_recursive = text_splitter_recursive.split_documents(documents)

# len(docs_recursive)

# # * Text Embeddings

# # OpenAI Embeddings
# # - See Account Limits for models: https://platform.openai.com/account/limits
# # - See billing to add to your credit balance: https://platform.openai.com/account/billing/overview

# embedding_function = OpenAIEmbeddings(
#     model='text-embedding-ada-002',
#     api_key=OPENAI_API_KEY
# )

# # Open Source Alternative:
# # Requires Torch and SentenceTransformer packages:
# # embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


# # * Langchain Vector Store: Chroma DB
# # https://python.langchain.com/docs/integrations/vectorstores/chroma

# # Creates a sqlite database called vector_store.db
# vectorstore = Chroma.from_documents(
#     docs, 
#     embedding=embedding_function, 
#     persist_directory="data/chroma_2.db"
# )

# vectorstore


# # * Similarity Search: The whole reason we did this

# result = vectorstore.similarity_search("How to create a social media strategy", k = 4)

# pprint(result[0].page_content)


