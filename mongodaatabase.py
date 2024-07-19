from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['wikipedia']
collection = db['embeddings']

# Create text index on the extracted_text field
collection.create_index([('extracted_text', 'text')])

# Load all embeddings and document IDs from MongoDB
documents = list(collection.find({}, {"_id": 1, "embedding": 1}))
embeddings = np.array([doc['embedding'] for doc in documents]).astype('float32')  # Ensure embeddings are float32
doc_ids = [doc['_id'] for doc in documents]

# Initialize Faiss index
index = faiss.IndexFlatL2(embeddings.shape[1])  # Use the dimensionality of embeddings
index.add(embeddings)  # Add the embeddings to the index

# Function to search using both text and vector search
def search_text_and_vector(input_text, k=5):
    # Text Search
    text_search_query = {'$text': {'$search': input_text}}
    text_results = collection.find(text_search_query)
    
    # Vector Search
    # Generate embedding for the input text
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    input_embedding = model.encode(input_text).reshape(1, -1).astype('float32')
    
    # Perform vector search
    distances, indices = index.search(input_embedding, k)
    
    # Collect results
    vector_results = [collection.find_one({'_id': doc_ids[idx]}, {'_id': 0, 'extracted_text': 1})['extracted_text'] for idx in indices[0]]
    
    # Combine results
    print("Text Search Results:")
    for doc in text_results:
        print(doc['extracted_text'])

    print("\nVector Search Results:")
    for text in vector_results:
        print(text)

# Example usage
new_text = "logbinary"
search_text_and_vector(new_text)
