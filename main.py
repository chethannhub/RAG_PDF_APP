# main.py

import logging
import os
import datetime
from urllib import response
import uuid
import requests

import inngest
import inngest.fast_api
from inngest.experimental import ai
from fastapi import FastAPI
from dotenv import load_dotenv

from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_types import RAGChunkAndSrc, RAGUpsertResult, RAGSearchResult, RAGQueryResult

from transformers import pipeline

load_dotenv()

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)


API_URL = "https://api.groq.com/openai/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {os.getenv('GROK_API_KEY')}",
    "Content-Type": "application/json"
}

# API_URL = "https://router.huggingface.co/hf-inference/models/meta-llama/Llama-2-7b-hf"
# headers = {"Authorization": f"Bearer {os.getenv('HF_API_KEY')}"}

# try:
#     pipe = pipeline(
#         "text-generation", model="meta-llama/Llama-2-7b-chat-hf"
#     )
#     logging.info("HF pipeline initialized successfully.")

# except Exception as e:
#     logging.exception("Failed to create HF pipeline - make sure model is available and transformers is configured.")
#     raise

@inngest_client.create_function(
    fn_id="RAG: Inngest PDF",
    trigger=inngest.TriggerEvent(event="rag/inngest_pdf")
)
async def rag_inngest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data['pdf_path']
        source_id = ctx.event.data.get('source_id', pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunk=chunks, source_id=source_id)
    
    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunk
        source_id = chunks_and_src.source_id
        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}: {i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        
        QdrantStorage().upsert(ids=ids, vectors=vecs, payloads=payloads)
        return RAGUpsertResult(ingested=len(chunks))
        

    chunks_and_src = await ctx.step.run("load_and_chunk_pdf", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run("embed_and_upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)

    return ingested.model_dump()

@inngest_client.create_function(
    fn_id="RAG: Query pdf",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        query_vector = embed_texts([question])[0]
        store = QdrantStorage()
        found = store.search(query_vector=query_vector, top_k=top_k)
        return RAGSearchResult(contexts=found['contexts'], sources=found['sources'])

    question = ctx.event.data['question']
    top_k = int(ctx.event.data.get('top_k', 5))

    found = await ctx.step.run("embed_and_search", lambda: _search(question, top_k), output_type=RAGSearchResult)

    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    
    user_content = (
        "Use the following context to answer the question. \n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer concisely based on the context above."
    )
    
    # adapter = ai.openai.Adapter(
    #     auth_key=os.getenv("OPENAI_API_KEY"),
    #     model_name="gpt-4o"
    # )
    
    # res = await ctx.step.ai.infer(
    #     "llm_answer",
    #     adapter=adapter,
    #     body={
    #         "temperature": 0.2,
    #         "max_tokens": 1000,
    #         "message": [        
    #             {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
    #             {"role": "user", "content": user_content}
    #         ]
    #     }
    #     }
    # )
    
    # def query(payload):
    #     response = requests.post(API_URL, headers=headers, json=payload)
        
    #     if response.status_code != 200:
    #         print("HF Error:", response.status_code, response.text)
    #         return {"error": f"HTTP {response.status_code}: {response.text}"}
    
    #     try:
    #         return response.json()
    #     except Exception as e:
    #         print("HF returned non-JSON response:", response.text)
    #         return {"error": f"Invalid JSON: {response.text}"}

    # payload = {
    #     "inputs": [
    #         {
    #             "role": "system",
    #             "content": "You are a helpful assistant that answers questions based on provided context."
    #         },
    #         {
    #             "role": "user",
    #             "content": user_content
    #         }
    #     ],
    #     "parameters": {
    #         "max_new_tokens": 1000,
    #         "temperature": 0.2
    #     }
    # }
    
    # res = query(payload)

    # print("Generated response:", res)

    # if isinstance(res, list) and len(res) > 0 and "generated_text" in res[0]:
    #     answer = res[0]["generated_text"].strip()
    # elif isinstance(res, dict) and "generated_text" in res:
    #     answer = res["generated_text"].strip()
    # elif isinstance(res, dict) and "error" in res:
    #     answer = f"(HF API Error) {res['error']}"
    # else:
    #     answer = "(No valid response received from model)"
    
    
    payload = {
        "model": "openai/gpt-oss-20b",  
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that answers based on provided context."},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.3,
        "max_tokens": 1000
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    res = response.json()
    print("Generated response:", res)

    if "choices" in res and len(res["choices"]) > 0:
        answer = res["choices"][0]["message"]["content"]
    else:
        answer = "(No response from Grok API)"

    return {"answer": answer, "sources": found.sources, "num_contexts": len(found.contexts)}

app = FastAPI()

inngest.fast_api.serve(app, inngest_client, functions=[rag_inngest_pdf, rag_query_pdf_ai])