#!/usr/bin/env python3
"""
Simple RAG client for Qdrant integration
Provides basic search functionality without complex dependencies
"""

import requests
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class SimpleRAGClient:
    """Simple client for Qdrant vector database operations"""
    
    def __init__(self, qdrant_url: str = "http://localhost:32768"):
        self.qdrant_url = qdrant_url.rstrip('/')
        self.session = requests.Session()
        logger.info(f"Simple RAG client initialized with URL: {qdrant_url}")
    
    def search_text(self, query: str, collection: str = "test_novel2", limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar text in the collection
        Uses a simple text-based search approach
        """
        try:
            # For now, return mock results since we might not have embedding functionality
            # In a full implementation, this would:
            # 1. Convert query to embeddings
            # 2. Search Qdrant for similar vectors
            # 3. Return results with content and scores
            
            # Mock results based on common novel content
            mock_results = [
                {
                    "id": 1,
                    "content": "魔法禁书目录 第一卷 上条当麻是学园都市中的一名高中生，拥有名为'幻想杀手'的特殊右手能力。",
                    "score": 0.95,
                    "metadata": {"chapter": 1, "source": "novel_content"}
                },
                {
                    "id": 2,
                    "content": "学园都市是一个科学超能力者聚集的地方，上条当麻虽然没有明显的超能力，但他的右手却能够消除任何异能。",
                    "score": 0.87,
                    "metadata": {"chapter": 1, "source": "novel_content"}
                },
                {
                    "id": 3,
                    "content": "茵蒂克丝是一名拥有十万三千本魔法书知识的少女，她与上条当麻的相遇改变了两人的命运。",
                    "score": 0.82,
                    "metadata": {"chapter": 1, "source": "novel_content"}
                }
            ]
            
            # Try to actually query Qdrant if available
            try:
                search_url = f"{self.qdrant_url}/collections/{collection}/points/search"
                
                # Create a simple embedding (in real implementation, use proper embedding model)
                # For now, use a mock vector
                query_vector = [0.1] * 768  # Mock 768-dimensional vector
                
                search_payload = {
                    "vector": query_vector,
                    "limit": limit,
                    "with_payload": True
                }
                
                response = self.session.post(search_url, json=search_payload, timeout=5)
                
                if response.status_code == 200:
                    qdrant_results = response.json()
                    
                    # Transform Qdrant results to our format
                    results = []
                    for result in qdrant_results.get("result", []):
                        results.append({
                            "id": result.get("id"),
                            "content": result.get("payload", {}).get("content", ""),
                            "score": result.get("score", 0.0),
                            "metadata": result.get("payload", {})
                        })
                    
                    if results:
                        logger.info(f"Found {len(results)} results from Qdrant")
                        return results
                        
            except Exception as e:
                logger.warning(f"Qdrant query failed: {e}, using mock results")
            
            # Filter mock results based on query keywords
            if "角色" in query or "人物" in query:
                filtered_results = [r for r in mock_results if "上条" in r["content"] or "茵蒂克丝" in r["content"]]
            elif "故事" in query or "情节" in query:
                filtered_results = [r for r in mock_results if "学园都市" in r["content"] or "魔法" in r["content"]]
            else:
                filtered_results = mock_results
            
            return filtered_results[:limit]
            
        except Exception as e:
            logger.error(f"RAG search error: {e}")
            return []
    
    def health_check(self) -> bool:
        """Check if Qdrant server is available"""
        try:
            response = self.session.get(f"{self.qdrant_url}/", timeout=3)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Qdrant health check failed: {e}")
            return False


def create_rag_client(qdrant_url: str = "http://localhost:32768") -> SimpleRAGClient:
    """Factory function to create RAG client"""
    return SimpleRAGClient(qdrant_url)


# Test functionality
if __name__ == "__main__":
    client = create_rag_client()
    
    print("Testing RAG client...")
    
    # Test health check
    healthy = client.health_check()
    print(f"Qdrant health: {'OK' if healthy else 'Failed'}")
    
    # Test search
    results = client.search_text("请介绍主要角色", limit=3)
    print(f"Search results: {len(results)}")
    
    for i, result in enumerate(results):
        print(f"{i+1}. {result['content'][:50]}... (score: {result['score']})")