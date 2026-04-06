from flask import Flask, request, jsonify
from flask_cors import CORS
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import os

app = Flask(__name__)
CORS(app)

# Initialize models and client
from qdrant_client.http import models

embedding_model = SentenceTransformer("moka-ai/m3e-small")
qdrant_client = QdrantClient(os.getenv("QDRANT_URL", "http://localhost:6333"))
COLLECTION_NAME = "test_novel2"

# Ensure Qdrant is healthy before attempting collection operations
import time
retries = 5
while retries > 0:
    try:
        # Check if Qdrant is responsive
        qdrant_client.get_collections()
        
        # Check if collection exists
        try:
            qdrant_client.get_collection(COLLECTION_NAME)
            app.logger.info(f"Collection {COLLECTION_NAME} already exists")
            break
        except Exception:
            app.logger.info(f"Creating collection {COLLECTION_NAME}")
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=768,
                    distance=models.Distance.COSINE
                )
            )
            app.logger.info(f"Collection {COLLECTION_NAME} created successfully")
            time.sleep(1)  # Brief pause after creation
            break
            
    except Exception as e:
        app.logger.warning(f"Qdrant not ready, retrying... ({retries} attempts left)")
        retries -= 1
        time.sleep(2)
else:
    app.logger.error("Failed to connect to Qdrant after multiple attempts")

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query')
    top_k = data.get('top_k', 5)
    
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400
    
    # Generate embedding
    embedding = embedding_model.encode(query, convert_to_tensor=False).tolist()
    
    # Search Qdrant
    search_results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=embedding,
        limit=top_k
    )
    
    # Format results
    results = []
    for result in search_results:
        results.append({
            "score": result.score,
            "chunk": result.payload.get("chunk", ""),
            "coordinate": result.payload.get("coordinate", [])
        })
    
    return jsonify(results)

if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)
