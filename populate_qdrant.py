#!/usr/bin/env python3
"""
Populate Qdrant Database with Novel Data
Based on your working notebook code
"""

import pandas as pd
import qdrant_client
from qdrant_client.http import models
import numpy as np
import json
import torch
import os
from typing import List, Any

def convert_to_list(value):
    """Convert embeddings to list format (from your notebook)"""
    if isinstance(value, list):  
        return value  # Already a valid list
    elif isinstance(value, np.ndarray):  
        return value.tolist()  # Convert NumPy array to list
    elif isinstance(value, torch.Tensor):  
        return value.detach().cpu().tolist()  # Convert PyTorch tensor to list
    elif isinstance(value, str):  
        try:
            return json.loads(value)  # Convert JSON string to list
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON string for embeddings: {value}")
    raise ValueError(f"Unsupported type for embeddings: {type(value)}")

def setup_qdrant_collection(q_client, collection_name: str, vector_size: int = 768):
    """Setup Qdrant collection"""
    try:
        # Delete existing collection if it exists
        try:
            q_client.delete_collection(collection_name)
            print(f"Deleted existing collection '{collection_name}'")
        except Exception:
            pass  # Collection might not exist
        
        # Create new collection
        q_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size, 
                distance=models.Distance.COSINE
            )
        )
        print(f"Created new collection '{collection_name}' with vector size {vector_size}")
        return True
        
    except Exception as e:
        print(f"Error setting up collection: {e}")
        return False

def insert_data_to_qdrant(merged_df, collection_name: str, q_client):
    """Insert data into Qdrant (from your notebook)"""
    total_inserted = 0
    
    for index, row in merged_df.iterrows():
        try:
            vectors = convert_to_list(row['session_chunks_embeddings'])
            chunks = convert_to_list(row['session_chunks'])

            if len(vectors) != len(chunks):
                print(f"Row {index}: Mismatch between vectors and chunks (vectors: {len(vectors)}, chunks: {len(chunks)})")
                continue

            print(f"Processing row {index}: {len(vectors)} vectors")

            # Insert in batches for better performance
            points_batch = []
            for i, (vector, chunk) in enumerate(zip(vectors, chunks)):
                payload = {
                    "coordinate": [index, i],
                    "chunk": chunk
                }
                vector_data = models.PointStruct(
                    id=index * 1000 + i,  
                    vector=vector,
                    payload=payload
                )
                points_batch.append(vector_data)
                
                # Insert in batches of 100
                if len(points_batch) >= 100:
                    q_client.upsert(
                        collection_name=collection_name,
                        points=points_batch
                    )
                    total_inserted += len(points_batch)
                    points_batch = []
            
            # Insert remaining points
            if points_batch:
                q_client.upsert(
                    collection_name=collection_name,
                    points=points_batch
                )
                total_inserted += len(points_batch)

            print(f"Inserted vectors and chunks for row {index}")
            
        except Exception as e:
            print(f"Skipping row {index} due to error: {e}")
    
    print(f"Total points inserted: {total_inserted}")
    return total_inserted

def load_and_populate():
    """Main function to load data and populate Qdrant"""
    print("Starting Qdrant Population Process")
    print("=" * 50)
    
    # Check if data file exists
    data_paths = [
        "D:/embedding/mojin_embeddings.parquet",
        "D:\\embedding\\mojin_embeddings.parquet",
        "../mojin_embeddings.parquet",
        "./mojin_embeddings.parquet"
    ]
    
    data_path = None
    for path in data_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if not data_path:
        print("Could not find mojin_embeddings.parquet file")
        print("Tried paths:")
        for path in data_paths:
            print(f"  - {path}")
        print("\nPlease ensure the file exists or update the path in this script.")
        return False
    
    print(f"Found data file: {data_path}")
    
    # Load data
    try:
        print("Loading embeddings data...")
        df = pd.read_parquet(data_path)
        print(f"Loaded data with {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
        
        # Check data structure
        if len(df) > 0:
            sample_embeddings = df['session_chunks_embeddings'].iloc[0]
            sample_chunks = df['session_chunks'].iloc[0]
            
            if isinstance(sample_embeddings, np.ndarray):
                vector_size = len(sample_embeddings[0]) if len(sample_embeddings) > 0 else 768
            else:
                vector_size = 768  # Default
                
            print(f"Detected vector size: {vector_size}")
            print(f"Sample row has {len(sample_chunks)} chunks")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return False
    
    # Initialize Qdrant client
    try:
        print("Connecting to Qdrant...")
        q_client = qdrant_client.QdrantClient("http://localhost:6333")
        
        # Test connection
        collections = q_client.get_collections()
        print(f"Connected to Qdrant, found {len(collections.collections)} existing collections")
        
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")
        print("Make sure Qdrant is running on http://localhost:6333")
        return False
    
    # Setup collection
    collection_name = "test_novel2"
    print(f"Setting up collection '{collection_name}'...")
    
    if not setup_qdrant_collection(q_client, collection_name, vector_size):
        return False
    
    # Insert data
    print("Inserting data into Qdrant...")
    total_inserted = insert_data_to_qdrant(df, collection_name, q_client)
    
    if total_inserted > 0:
        print(f"\nSUCCESS! Inserted {total_inserted} data points into Qdrant")
        
        # Verify insertion
        try:
            collection_info = q_client.get_collection(collection_name)
            count_result = q_client.count(collection_name)
            print(f"Collection status: {collection_info.status}")
            print(f"Final point count: {count_result.count}")
            
            if count_result.count != total_inserted:
                print(f"Warning: Expected {total_inserted} points but found {count_result.count}")
            
        except Exception as e:
            print(f"Could not verify insertion: {e}")
        
        return True
    else:
        print("No data was inserted")
        return False

if __name__ == "__main__":
    success = load_and_populate()
    
    if success:
        print("\nQdrant population completed successfully!")
        print("You can now test your character simulation system with real data.")
    else:
        print("\nQdrant population failed.")
        print("Please check the error messages above and try again.")