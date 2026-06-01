from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


model = SentenceTransformer('all-MiniLM-L6-v2')

def build_index(chunks):
    if not chunks:
        return faiss.IndexFlatL2(384), [], chunks

    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings).astype('float32'))

    return index, embeddings, chunks


def search(query, index, chunks, k=3):
    if not chunks or index.ntotal == 0:
        return []

    
    query_vec = model.encode([query])
    query_vec = np.array(query_vec).astype('float32')

    distances, indices = index.search(query_vec, k)

    results = []
    for i, d in zip(indices[0], distances[0]):
        if i != -1 and i < len(chunks) and d < 1.2:
            results.append(chunks[i])

    return results
