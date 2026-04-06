#!/usr/bin/env python
# coding: utf-8

# In[1]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[15]:


get_ipython().system('pip install langchain')


# In[16]:


pip install ebooklib beautifulsoup4 pandas


# In[19]:


import os
import pandas as pd
from ebooklib import epub, ITEM_DOCUMENT  # Import ITEM_DOCUMENT correctly
from bs4 import BeautifulSoup

def extract_text_from_epub(epub_path):
    """Extracts text from an EPUB file and prints debug info."""
    book = epub.read_epub(epub_path)
    text = []
    
    for item in book.get_items():
        if item.get_type() == ITEM_DOCUMENT:  # Use imported constant
            soup = BeautifulSoup(item.get_body_content(), "html.parser")
            text.append(soup.get_text())

    extracted_text = "\n".join(text)
    print(f"Extracted {len(extracted_text)} characters from {os.path.basename(epub_path)}")
    
    return extracted_text

def process_epub_files(directory):
    """Reads all EPUB files in a directory, extracts text, and stores them in a DataFrame."""
    epub_files = [f for f in os.listdir(directory) if f.endswith('.epub')]
    
    print(f"Found {len(epub_files)} EPUB files in the directory.")
    
    data = []
    processed_files = 0

    for epub_file in epub_files:
        epub_path = os.path.join(directory, epub_file)
        text = extract_text_from_epub(epub_path)

        if text.strip():  # Ensure we add only non-empty text
            data.append({"filename": epub_file, "text": text})
            processed_files += 1
        else:
            print(f"Warning: No text extracted from {epub_file}")

    print(f"Processed {processed_files} out of {len(epub_files)} EPUB files.")

    return pd.DataFrame(data)

# Set your local directory path containing EPUB files
epub_directory = "D:/mojin_novel"
df = process_epub_files(epub_directory)



# In[ ]:





# In[17]:


pip install cnlp


# In[20]:


import tiktoken
import re
from io import StringIO
import boto3
from langchain.text_splitter import RecursiveCharacterTextSplitter



def tiktoken_len(text):
    tokens = encoding.encode(text, disallowed_special=())
    return len(tokens)

# Chunking setup
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=20,
    length_function=tiktoken_len,
    separators=["\n\n", "\n", " ", ""]
)

class SessionTextSplitter:
    def __init__(self, session_pattern, chunk_size=1000, chunk_overlap=20):
        self.session_pattern = session_pattern
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text):
        # Split the text by the session pattern
        session_splits = re.split(self.session_pattern, text)
        chunks = []
        for split in session_splits:
            # Further split the text into chunks of the specified size
            chunks.extend(self.chunk_text(split))
        return chunks

    def chunk_text(self, text):
        length = len(text)
        chunks = []
        for i in range(0, length, self.chunk_size - self.chunk_overlap):
            chunk = text[i:i + self.chunk_size]
            chunks.append(chunk)
        return chunks

# Define the session pattern for splitting
session_pattern = r'\n\d{1,2}\n\)'

# Create the session text splitter
session_splitter = SessionTextSplitter(session_pattern)



# Tokenize and chunk text
df['session_chunks'] = df['text'].apply(lambda x: session_splitter.split_text(x))

# Count the number of chunks in each row for the first 50 rows
chunk_counts = df['session_chunks'].head(50).apply(len)




# In[21]:


# Count the number of chunks in each row for the first 50 rows
chunk_counts = df['session_chunks'].head(50).apply(len)

# Print the chunk counts
print(chunk_counts)


# In[14]:


def verify_session_chunks(df):
    for i, row in df.iterrows():
        chunks = row['session_chunks']
        
        # Check if the value is a list
        if not isinstance(chunks, list):
            print(f"Row {i} error: Expected list, but got {type(chunks)}")
            print(f"Value: {chunks}\n")
            continue
        
        # Check if all elements in the list are strings
        if not all(isinstance(chunk, str) for chunk in chunks):
            print(f"Row {i} error: Not all elements are strings.")
            print(f"Value: {chunks}\n")

    print("Verification complete!")

# Run verification
verify_session_chunks(df)




# In[ ]:





# In[ ]:





# In[18]:


pip install transformers sentence-transformers torch


# In[8]:


import pandas as pd
from transformers import AutoTokenizer

# Load BERT-based tokenizer from M3e-Small
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-zh")

# Function to tokenize a list of text chunks
def bert_tokenize_list(chunks):
    """Tokenizes a list of text chunks using M3e-Small tokenizer."""
    return [tokenizer(chunk, padding=True, truncation=True, return_tensors="pt") for chunk in chunks]

# Function to process tokenization and store in a new column
def tokenize_session_chunks(data_frame):
    num_rows = len(data_frame)
    tokenized_chunks_list = []

    for index, row in enumerate(data_frame.itertuples(), start=1):
        chunks = row.session_chunks  # Original list of strings
        tokenized_chunks = bert_tokenize_list(chunks)  # Tokenized list of lists
        tokenized_chunks_list.append(tokenized_chunks)

        # Print progress every 10 chunks
        for i in range(0, len(chunks), 10):
            print(f"Processed {min(i + 10, len(chunks))}/{len(chunks)} chunks in row {index}/{num_rows}...")

    # Store the tokenized version in a new column
    data_frame['session_chunks_tokenized'] = tokenized_chunks_list

    return data_frame

# Apply tokenization
df = tokenize_session_chunks(df)


# In[ ]:





# In[22]:


import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# Load BERT-based tokenizer & embedding model
tokenizer = AutoTokenizer.from_pretrained("moka-ai/m3e-small")
embedding_model = SentenceTransformer("moka-ai/m3e-small")

# Function to tokenize text chunks
def bert_tokenize_list(chunks):
    """Tokenizes a list of text chunks using M3e-Small tokenizer."""
    return [tokenizer(chunk, padding=True, truncation=True, return_tensors="pt") for chunk in chunks]

# Function to embed tokenized chunks
def embed_tokenized_chunks(data_frame):
    """Embeds the tokenized chunks using the M3e-Small model with real-time progress tracking."""
    
    num_rows = len(data_frame)
    session_embeddings_list = []

    for row_index, row in enumerate(data_frame.itertuples(), start=1):
        original_chunks = row.session_chunks  # Original list of strings
        num_chunks = len(original_chunks)

        # Print row progress
        print(f"\nProcessing row {row_index}/{num_rows}... (Total {num_chunks} chunks)")

        # Tokenize each chunk
        tokenized_chunks = bert_tokenize_list(original_chunks)

        # Convert tokenized chunks back to text format (for embeddings)
        processed_texts = [" ".join(tokenizer.convert_ids_to_tokens(tc["input_ids"][0])) for tc in tokenized_chunks]

        # Generate embeddings for all chunks in this row
        embeddings_list = []
        for i in range(0, num_chunks, 50):  # Process in batches of 50
            batch_texts = processed_texts[i:i + 50]
            batch_embeddings = embedding_model.encode(batch_texts, convert_to_tensor=True)
            embeddings_list.extend(batch_embeddings)

            # Print chunk progress inside row
            print(f"  Processed {min(i + 50, num_chunks)}/{num_chunks} chunks in row {row_index}/{num_rows}...")

        # Store the row embeddings
        session_embeddings_list.append(torch.stack(embeddings_list))

    # Store embeddings in a new column
    data_frame['session_chunks_embeddings'] = session_embeddings_list

    print("\n✅ Embedding process completed for all rows!")
    
    return data_frame

# Apply tokenization & embedding
df = embed_tokenized_chunks(df)


# In[23]:


import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

# Load BGE-large-zh tokenizer & embedding model
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-zh")
embedding_model = AutoModel.from_pretrained("BAAI/bge-large-zh")
embedding_model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_model.to(device)

# Function to tokenize text chunks
def bert_tokenize_list(chunks):
    """Tokenizes a list of text chunks using BGE-large-zh tokenizer."""
    return [tokenizer(chunk, padding=True, truncation=True, return_tensors="pt").to(device) for chunk in chunks]

# Function to embed tokenized chunks
def embed_tokenized_chunks(data_frame):
    """Embeds the tokenized chunks using BGE-large-zh with real-time progress tracking."""
    
    num_rows = len(data_frame)
    session_embeddings_list = []

    for row_index, row in enumerate(data_frame.itertuples(), start=1):
        original_chunks = row.session_chunks  # Original list of strings
        num_chunks = len(original_chunks)

        # Print row progress
        print(f"\nProcessing row {row_index}/{num_rows}... (Total {num_chunks} chunks)")

        # Tokenize each chunk
        tokenized_chunks = bert_tokenize_list(original_chunks)

        # Convert tokenized chunks back to text format (for embeddings)
        processed_texts = [" ".join(tokenizer.convert_ids_to_tokens(tc["input_ids"][0])) for tc in tokenized_chunks]

        # Generate embeddings for all chunks in this row
        embeddings_list = []
        for i in range(0, num_chunks, 50):  # Process in batches of 50
            batch_texts = processed_texts[i:i + 50]

            # Tokenize batch again (this time as a batch for efficiency)
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = embedding_model(**inputs)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling

            embeddings_list.extend(batch_embeddings)

            # Print chunk progress inside row
            print(f"  Processed {min(i + 50, num_chunks)}/{num_chunks} chunks in row {row_index}/{num_rows}...")

        # Store the row embeddings
        session_embeddings_list.append(torch.stack(embeddings_list))

    # Store embeddings in a new column
    data_frame['session_chunks_embeddings'] = session_embeddings_list

    print("\n✅ Embedding process completed for all rows!")
    
    return data_frame

# Apply tokenization & embedding
df = embed_tokenized_chunks(df)


# In[ ]:





# In[10]:


# Examine the dimensionality of embeddings
for index, row in df.iterrows():
    embeddings = row['session_chunks_embeddings']  # Retrieve embeddings for this row
    print(f"\nRow {index}:")
    print(f"  Total chunks: {len(embeddings)}")
    print(f"  Shape of first chunk embedding: {embeddings[0].shape}")  # Should be (768,)
    
    # Check if all embeddings are of the same dimension
    all_same_dim = all(e.shape == (512,) for e in embeddings)
    print(f"  Consistent embedding size across chunks? {'Yes' if all_same_dim else 'No'}")

    # Limit to 3 rows for examination
    if index == 2:
        break


# In[ ]:





# In[9]:


pip install pyarrow


# In[12]:


print(type(df['session_chunks_embeddings'].iloc[0]))  # Expected: list
print(len(df['session_chunks_embeddings'].iloc[0]))  # Expected: ~300 (number of vectors)
print(len(df['session_chunks_embeddings'].iloc[0][0]))  # Expected: 768 (vector dimension)


# In[ ]:





# In[13]:


df["session_chunks_embeddings"] = df["session_chunks_embeddings"].apply(lambda x: x.detach().cpu().tolist())


# In[ ]:





# In[14]:


import pyarrow as pa
import pyarrow.parquet as pq

# Define the save path
save_path = "D:/embedding/mojin_embeddings_baai.parquet"

# Keep only the required columns
df_to_save = df[["filename", "session_chunks", "session_chunks_embeddings"]]

# Convert DataFrame to an Arrow Table
table = pa.Table.from_pandas(df_to_save)

# Write to a Parquet file
pq.write_table(table, save_path)

print(f"\n✅ DataFrame saved successfully as Parquet at: {save_path}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[76]:





# In[24]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ##### rag_search_instance2.enable_tracing(True)

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[11]:





# In[ ]:





# In[ ]:




