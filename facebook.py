from sentence_transformers import SentenceTransformer
import pymongo
import numpy as np
import faiss

# Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# MongoDB connection
client = pymongo.MongoClient('mongodb://localhost:27017/')
db = client['wikipedia']
collection = db['embeddings']

# Fetch texts from MongoDB
data = list(collection.find({}, {'_id': 1, 'text': 1}))
texts = [item['text'] for item in data]
ids = [item['_id'] for item in data]

# Compute new embeddings
embeddings = model.encode(texts)
embeddings = np.array(embeddings, dtype=np.float32)

# Update MongoDB with new embeddings
for i, emb in enumerate(embeddings):
    collection.update_one({'_id': ids[i]}, {'$set': {'embedding': emb.tolist()}})

# Initialize FAISS index with updated dimension
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Function to query similar texts
def query_similar_texts(new_text, top_k=5):
    new_embedding = model.encode([new_text])
    
    # Print the shape and data type of the new embedding
    print("New embedding shape:", new_embedding.shape)
    print("New embedding dtype:", new_embedding.dtype)
    
    # Ensure the new embedding has the same dimension as the indexed embeddings
    if new_embedding.shape[1] != dimension:
        raise ValueError(f"Dimension mismatch: Index dimension {dimension}, but query embedding dimension {new_embedding.shape[1]}")

    distances, indices = index.search(new_embedding, top_k)
    print("Distances:", distances)
    print("Indices:", indices)
    
    similar_ids = [ids[i] for i in indices[0]]
    similar_texts = collection.find({'_id': {'$in': similar_ids}})
    return list(similar_texts)

# Example usage
new_text = "logbinary"
similar_texts = query_similar_texts(new_text, top_k=5)
for text in similar_texts:
    print(text['text'])
