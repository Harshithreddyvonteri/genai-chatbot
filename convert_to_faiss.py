import pickle
import numpy as np
import chromadb
import faiss

CHROMA_PATH = "data/vectorstore/v2_chroma"
FAISS_PATH = "data/vectorstore/v2_faiss.pkl"

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection("gitlab_v2")
all_items = collection.get(include=['embeddings', 'metadatas', 'documents'])

embeddings = np.array(all_items['embeddings'], dtype=np.float32)
documents = all_items['documents']
metadata = all_items['metadatas']

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

with open(FAISS_PATH, "wb") as f:
    pickle.dump({"index": index, "documents": documents, "metadata": metadata}, f)

print(f"FAISS index saved to {FAISS_PATH}")
