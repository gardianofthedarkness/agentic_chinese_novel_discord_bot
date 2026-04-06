#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd

# Define the file path
save_path = "D:/embedding/mojin_embeddings.parquet"

# Read the CSV file into a pandas DataFrame
df = pd.read_parquet(save_path)


# In[8]:


# Print the column headings
print("Column headings of the merged dataset:")
print(df.columns.tolist())


# In[3]:


import os
import boto3
import fitz  # PyMuPDF
from io import BytesIO
import pandas as pd
# Print the column headings
print("Column headings of the merged dataset:")
print(df.columns.tolist())


# In[10]:


#print(df['session_chunks_embeddings'].iloc[0])
print(type(df['session_chunks_embeddings'].iloc[0]))


# In[ ]:





# In[14]:


import qdrant_client
from qdrant_client.http import models
import numpy as np
import json
import torch  # Import PyTorch

# Initialize Qdrant client
q_client = qdrant_client.QdrantClient("http://localhost:6333")

# Create collection only if it doesn't exist
collection_name = "test_novel2"
if not q_client.collection_exists(collection_name):
    q_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE)  # Assuming 768-dimensional vectors
    )

# Function to safely convert embeddings into lists
def convert_to_list(value):
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

# Function to insert data into Qdrant
def insert_data_to_qdrant(merged_df, collection_name):
    for index, row in merged_df.iterrows():
        try:
            vectors = convert_to_list(row['session_chunks_embeddings'])
            chunks = convert_to_list(row['session_chunks'])

            if len(vectors) != len(chunks):
                raise ValueError(f"Row {index}: Mismatch between vectors and chunks (vectors: {len(vectors)}, chunks: {len(chunks)})")

            print(f"Number of vectors in row {index}: {len(vectors)}")

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
                q_client.upsert(
                    collection_name=collection_name,
                    points=[vector_data]
                )

            print(f"Inserted vectors and chunks for row {index} into Qdrant.")
        except Exception as e:
            print(f"Skipping row {index} due to error: {e}")

# Run insertion function
insert_data_to_qdrant(df, collection_name)

print("Data insertion complete!")


# In[ ]:





# In[ ]:





# In[25]:


from typing import List, Dict

def search_vector(q_client, embedded_vector: List[float], collection_name: str, top_k: int = 5) -> List[Dict[str, any]]:
    """
    Perform a search in Qdrant using an embedded vector and return the chunks and coordinates.
    
    :param q_client: The initialized Qdrant client.
    :param embedded_vector: The vector to search for.
    :param collection_name: The name of the Qdrant collection to search within.
    :param top_k: The number of nearest vectors to retrieve.
    :return: List of dictionaries containing chunks and their corresponding coordinates.
    """
    print("Performing search...")

    # Perform search using Qdrant
    search_result = q_client.search(
        collection_name=collection_name,
        query_vector=embedded_vector,
        limit=top_k
    )
    
    if search_result:
        # Collect the chunks and coordinates of the nearest vectors
        results = [
            {
                "chunk": result.payload['chunk'],
                "coordinate": result.payload['coordinate']
            }
            for result in search_result
        ]
        return results
    else:
        print("No search results found.")
        return []

# Example usage:
# q_client = initialize_your_qdrant_client()
# results = search_vector(q_client, your_embedded_vector, "vector_collection")
# for result in results:
#     print(f"Chunk: {result['chunk']}, Coordinate: {result['coordinate']}")


# In[17]:


import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
def get_local_prompt_embedding(prompt):
    """
    Generates an embedding for a single prompt string using the local M3e-Small embedding model.
    
    Args:
        prompt (str): The text prompt to embed.
    
    Returns:
        torch.Tensor: The embedding tensor for the given prompt.
    """
    embedding = embedding_model.encode(prompt, convert_to_tensor=True)
    return embedding


# In[20]:


# Load BERT-based tokenizer & embedding model
tokenizer = AutoTokenizer.from_pretrained("moka-ai/m3e-small")
embedding_model = SentenceTransformer("moka-ai/m3e-small")


# In[29]:


prompt2 = "右方之火能用的魔法都有哪些"
prompt = 'what is the annuity payment plan'
prompt2_embedding = get_local_prompt_embedding(prompt2)


# In[30]:


results = search_vector(q_client, prompt2_embedding, "test_novel2")
for result in results:
    print(f"Chunk: {result['chunk']}, Coordinate: {result['coordinate']}")


# In[32]:


get_ipython().system('pip install azure-ai-inference')


# In[34]:


import torch
from sentence_transformers import SentenceTransformer
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential


AZURE_ENDPOINT = "https://DeepSeek-V3-xqjkp.eastus2.models.ai.azure.com"
AZURE_API_KEY = "zT7QmBLIeeILTQmuG5bnnwsgkFOC8gfw"
client = ChatCompletionsClient(
    endpoint=AZURE_ENDPOINT,
    credential=AzureKeyCredential(AZURE_API_KEY),
)





# In[35]:


# === HyDE Prompt Template ===
HYPOTHETICAL_ANSWER_TEMPLATE = """
你是一位小说创作助理，你的任务是根据提出的问题，结合小说的写作风格，生成一段合理的、详细的、风格一致的假设性文本，以便用于语义搜索。

请遵循以下要求：
1. 模拟小说中真实段落的写作风格。
2. 回答中应包含足够细节，以便向量检索系统能够匹配到相似的真实内容。
3. 如有提供的背景片段，请参考其语言风格与表达方式。
4. 避免直接重复问题内容，而是通过合理推理与想象，生成与之高度相关的自然语言段落。

【问题】
{text}

{background_section}

【请输出模拟的小说段落】
"""


# In[36]:


# === Template Formatter ===
def build_hypothetical_prompt(user_prompt, background_chunks=None):
    background_text = ""
    if background_chunks:
        joined_background = "\n".join(background_chunks)
        background_text = f"\n【背景片段】\n{joined_background}"

    return HYPOTHETICAL_ANSWER_TEMPLATE.replace("{text}", user_prompt).replace("{background_section}", background_text)

# === Use Azure ChatCompletionsClient to Generate Hypothetical Answer ===
def convert_text_to_json(text):
    try:
        response = client.complete(
            messages=[
                {"role": "system", "content": "你是一位中文小说创作代理。请根据输入，模拟输出小说风格的段落。"},
                {"role": "user", "content": text}
            ]
        )
        raw_output = response.choices[0].message.content.strip()
        print("\n📘 Hypothetical Answer Generated:\n", raw_output)
    except Exception as e:
        raw_output = f"Error: {str(e)}"
    
    return raw_output


# In[37]:


def generate_hypothetical_query(prompt, q_client, top_k_for_context=2):
    print("🔍 Step 1: Retrieve Initial Chunks for Hypothetical Generation")

    prompt_embedding = embedding_model.encode(prompt, convert_to_tensor=True)
    initial_results = search_vector(q_client, prompt_embedding, "test_novel2", top_k=top_k_for_context)

    for i, result in enumerate(initial_results):
        print(f"\nInitial Chunk {i+1}:\n{result['chunk']}")
        print(f"Coordinate: {result['coordinate']}")

    supporting_text = [res["chunk"] for res in initial_results]
    full_prompt = build_hypothetical_prompt(prompt, supporting_text)

    print("\n🧠 Step 2: Generating Hypothetical Answer with Azure")
    hypothetical_answer = convert_text_to_json(full_prompt)

    return hypothetical_answer


# In[49]:


def search_vector_with_neighbors(
    q_client,
    embedded_vector: List[float],
    collection_name: str,
    top_k: int = 5
) -> List[Dict[str, any]]:
    """
    Search using vector and return each hit's chunk + its valid neighbors.
    Handles 2D coordinate like [row_index, chunk_index], staying within the same row.
    Adds print statements for debugging.
    """
    print("🔍 Step 1: Vector Search")
    search_results = q_client.search(
        collection_name=collection_name,
        query_vector=embedded_vector,
        limit=top_k
    )

    all_coords_to_fetch = set()
    primary_coords = []

    for result in search_results:
        coord = result.payload.get("coordinate")
        print(f"📎 Raw coordinate: {coord}")

        if not (isinstance(coord, list) and len(coord) == 2 and all(isinstance(x, int) for x in coord)):
            print("   ⚠️ Skipping invalid coordinate:", coord)
            continue

        row_id, chunk_id = coord
        primary_coords.append((row_id, chunk_id))
        all_coords_to_fetch.update([
            (row_id, chunk_id - 1),
            (row_id, chunk_id),
            (row_id, chunk_id + 1)
        ])

    print(f"✅ Primary coords: {primary_coords}")
    print(f"📌 All needed coords (with neighbors): {sorted(all_coords_to_fetch)}")

    print("📦 Step 2: Fetch all points from Qdrant")
    points, _ = q_client.scroll(
        collection_name=collection_name,
        with_payload=True,
        limit=10000
    )

    coord_to_chunk = {}
    max_chunk_index_per_row = {}

    for point in points:
        coord = point.payload.get("coordinate")
        chunk = point.payload.get("chunk")

        if isinstance(coord, list) and len(coord) == 2:
            row_id, chunk_id = coord
            coord_tuple = (row_id, chunk_id)
            coord_to_chunk[coord_tuple] = chunk

            # Track max chunk index per row
            if row_id not in max_chunk_index_per_row:
                max_chunk_index_per_row[row_id] = chunk_id
            else:
                max_chunk_index_per_row[row_id] = max(max_chunk_index_per_row[row_id], chunk_id)

    print(f"🗺 Total chunked coords loaded: {len(coord_to_chunk)}")
    print(f"🗂 Example row max indices: {list(max_chunk_index_per_row.items())[:5]}")

    print("🧵 Step 3: Stitch neighbors")

    joined_results = []

    for (row_id, chunk_id) in primary_coords:
        max_chunk_id = max_chunk_index_per_row.get(row_id, chunk_id)

        parts = []

        if chunk_id > 0:
            prev = coord_to_chunk.get((row_id, chunk_id - 1))
            if prev:
                parts.append(prev)

        main = coord_to_chunk.get((row_id, chunk_id))
        if main:
            parts.append(main)

        if chunk_id < max_chunk_id:
            next_chunk = coord_to_chunk.get((row_id, chunk_id + 1))
            if next_chunk:
                parts.append(next_chunk)

        stitched_text = "\n".join(parts).strip()

        joined_results.append({
            "coordinate": [row_id, chunk_id],
            "joined_chunk": stitched_text
        })

    print(f"✅ Generated {len(joined_results)} stitched results.")
    return joined_results


# In[43]:


def search_with_hypothetical_embedding(hypothetical_text, q_client, top_k_for_results=5):
    print("\n🔄 Step 3: Final Search using HyDE Embedding with Neighbor Context")

    # Step 1: Embed the hypothetical text
    hyde_embedding = embedding_model.encode(hypothetical_text, convert_to_tensor=True)

    # Step 2: Search with neighbors
    hyde_results = search_vector_with_neighbors(
        q_client,
        embedded_vector=hyde_embedding,
        collection_name="test_novel2",
        top_k=top_k_for_results
    )

    # Step 3: Print results
    for i, result in enumerate(hyde_results):
        print(f"\n[HyDE] Result {i+1} (Center Coord: {result['coordinate']}):")
        print(result['joined_chunk'])

    return hyde_results



# In[42]:





# In[71]:


query = "前方之风的能力都有什么"


# In[72]:


# Step 1: Generate HyDE prompt using top 2 context chunks
hypothetical_text = generate_hypothetical_query(query, q_client, top_k_for_context=3)


# In[73]:


# Step 2: Perform final retrieval using HyDE embedding with top 5 results
final_results = search_with_hypothetical_embedding(hypothetical_text, q_client, top_k_for_results=5)


# In[ ]:




