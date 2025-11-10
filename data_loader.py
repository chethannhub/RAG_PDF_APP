# data_loader.py

import os
import requests
from openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter

from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 3072

HF_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
HF_HEADERS = {"Authorization": f"Bearer {os.getenv('HF_API_KEY')}"}

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, 'text', None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks

# def embed_texts(texts: list[str]) -> list[list[float]]:
#     response = client.embeddings.create(
#         model=EMBEDDING_MODEL,
#         input=texts
#     )
#     return [item.embedding for item in response.data]

model = SentenceTransformer('intfloat/e5-base-v2')

def embed_texts(texts: list[str]) -> list[list[float]]:
    
    prefixed_texts = [f"passage: {t}" for t in texts]


    embeddings = model.encode(
        prefixed_texts, 
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    return embeddings.tolist()


# def embed_texts(texts: list[str]) -> list[list[float]]:
#     prefixed_texts = [f"passage: {t}" for t in texts]
#     response = requests.post(
#         HF_API_URL,
#         headers=HF_HEADERS,
#         json={"inputs": prefixed_texts}
#     )
#     response.raise_for_status()
#     return response.json()


