import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
import numpy as np
from flask import Flask,render_template,request,jsonify
import json
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-mpnet-base-v2')

client = MongoClient('mongodb://localhost:27017/')
db = client['wikipedia']
collection = db['embeddings']

def scrap_text_from_url(url):
    try:
        response= requests.get(url)
        soup= BeautifulSoup(response.content,'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([para.get_text()for para in paragraphs])
        return text
    except Exception as e:
        print(f"Error scraping {url}:{e}")
        return ""
    
def store_embedding(url,text):
    embedding= model.encode(text)
    collection.insert_one({
        'url':url,
        'text':text,
        'embedding':embedding.tolist()
    })

def process_and_store_url(urls):
    for url in urls:
        text = scrap_text_from_url(url)
        if text:
            store_embedding(url,text)

def query_embedding(query):    
    embedding = model.encode(query)
    query_vector = np.array(embedding).reshape(-1,1)
    return query_vector

def find_similar_texts(query_vector,top_k=5):
    results = collection.find({})
    similarities = [] 
    for result in results:
        db_vector = np.array(result['embedding']).reshape(-1,1)
        similarity = cosine_similarity(query_vector,db_vector)[0][0]
        similarities.append((similarity,result['text']))
    similarities.sort(key=lambda x: x[0],reverse=True)
    return similarities[:top_k]

def search(query):
    query_vector = query_embedding(query)
    similar_text= find_similar_texts(query_vector)
    return similar_text

app = Flask(__name__)
@app.route('/search',methods=['POST'])
def search_endpoint():
    data = request.json
    query= data.get('query','')
    if not query:
        return jsonify({"error":"query is required"})
    results=search(query)
    return jsonify(results)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    urls =[
    "https://en.wikipedia.org/wiki/Artificial_intelligence"

    ]
    process_and_store_url(urls)

    app.run(host='0.0.0.0',port=5000)
