# vector_db.py

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# docker run -d --name qdrantRagDb -p 6333:6333 -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant

class QdrantStorage:
    def __init__(self, url="http://localhost:6333", collection="docs_e5_768", dim=768):
        self.client = QdrantClient(url=url, timeout=30)
        self.collection = collection
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection, 
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )

    def upsert(self, ids: list, vectors: list, payloads: list):
        points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query_vector, top_k: int = 5):
        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            with_payload=True,
            limit=top_k
        )
        contexts = []
        sources = set()
        
        for result in results:
            payload = getattr(result, 'payload', None) or {}
            text = payload.get('text', '')
            source = payload.get('source', '')
            if text:
                contexts.append(text)
                sources.add(source)

        return {"contexts": contexts, "sources": list(sources)}
    
    