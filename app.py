from flask import Flask, request, jsonify, render_template
from pymongo import MongoClient
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

client = MongoClient('localhost', 27017)
db = client['wikipedia']
collection = db['embeddings']
model = SentenceTransformer('all-mpnet-base-v2')

documents = list(collection.find())
embeddings = np.array([doc['embedding'] for doc in documents])
n_neighbors = min(5, len(documents))
nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(embeddings)

def embed_text(text):
    return model.encode(text)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query_text = request.json.get('query_text', '')
    if not query_text:
        return jsonify({"error": "query_text is required"}), 400

    query_vector = embed_text(query_text)
    if query_vector is None:
        return jsonify({"error": "Error embedding the query text."}), 500

    similarities = cosine_similarity([query_vector], embeddings).flatten()
    indices = similarities.argsort()[-n_neighbors:][::-1]
    scores = similarities[indices]

    seen_urls = set()
    output_lines = [f"Query results for '{query_text}':", ""]

    for i, idx in enumerate(indices):
        url = documents[idx]['url']
        if url in seen_urls:
            continue

        seen_urls.add(url)
        text = documents[idx].get('text', '')
        
        if not text:
            print(f"No text found for URL: {url}")
            continue

        score = scores[i]
        max_score = np.max(scores)
        percentage = (score / max_score) * 100 if max_score > 0 else 0

        output_lines.append(f"URL: {url}")
        output_lines.append('-' * 50)

        matching_paragraphs = []

        if text:
            for paragraph in text.split("\n\n"):
                if re.search(re.escape(query_text), paragraph, re.IGNORECASE):
                    highlighted_paragraph = re.sub(
                        re.escape(query_text),
                        r"<strong>\g<0></strong>",
                        paragraph,
                        flags=re.IGNORECASE
                    )
                    matching_paragraphs.append(highlighted_paragraph)

            if matching_paragraphs:
                output_lines.append("Matching paragraphs:")
                for match in matching_paragraphs:
                    output_lines.append(f"{match}")
            else:
                output_lines.append("No matching paragraphs found.")

        output_lines.append(f"Score: {score:.2f}, Percentage: {percentage:.2f}%")
        output_lines.append('-' * 50)

    if not output_lines:
        output_lines.append(f"No matches found for '{query_text}'.")

    return jsonify({"message": "\n".join(output_lines)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
