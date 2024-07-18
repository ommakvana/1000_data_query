from flask import Flask, request, jsonify, render_template
from pymongo import MongoClient
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

client = MongoClient('mongodb://localhost:27017/')
db = client['wikipedia']
collection = db['embeddings']

model = SentenceTransformer('all-mpnet-base-v2')

def embed_text(text):
    return model.encode(text)

@app.route('/')
def index():
    return render_template('index.html')

from sklearn.metrics.pairwise import cosine_similarity

import re

def search(query):
    query_vector = embed_text(query)
    if query_vector is None:
        return []

    results = collection.find({})
    similarities = []

    embeddings = []
    doc_details = []

    for result in results:
        embedding = np.array(result['embedding'])
        embeddings.append(embedding)
        doc_details.append((result['url'], result['text']))

    embeddings = np.array(embeddings)

    n_neighbors = min(5, len(embeddings))
    if n_neighbors == 0:
        return []

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(embeddings)
    distances, indices = nbrs.kneighbors(query_vector.reshape(1, -1))
    scores = np.dot(embeddings[indices[0]], query_vector)

    seen_urls = set()
    for i in range(len(indices[0])):
        idx = indices[0][i]
        url, text = doc_details[idx]

        if url in seen_urls:
            continue

        seen_urls.add(url)
        score = scores[i]
        max_score = np.max(scores)
        percentage = (score / max_score) * 100 if max_score > 0 else 0

        matching_paragraphs = []
        paragraphs = text.split("\n\n")

        for paragraph in paragraphs:
            normalized_query = query.lower()
            normalized_paragraph = paragraph.lower()

            if normalized_query in normalized_paragraph:
                highlighted_paragraph = re.sub(
                    fr"\b({re.escape(normalized_query)})\b",
                    r"<strong>\1</strong>",
                    paragraph,
                    flags=re.IGNORECASE
                )
                matching_paragraphs.append(highlighted_paragraph)

        # Append result with percentage, URL, text, and matching paragraphs
        similarities.append((percentage, url, text, matching_paragraphs))

    similarities.sort(key=lambda x: x[0], reverse=True)
    return similarities[:5]




@app.route('/search', methods=['POST'])
def query_similar():
    data = request.json
    query_text = data.get('query_text', '')
    if not query_text:
        return jsonify({"error": "query_text is required"})

    results = search(query_text)
    if not results:
        return jsonify({"message": "No similar texts found."})

    similar_texts = []
    for percentage, url, text,matching_paragraphs in results:
        similar_texts.append({
            'similarity': percentage,
            'url': url,
            'text': text,
            'matching_paragraphs':matching_paragraphs
        })

    return jsonify(similar_texts)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
