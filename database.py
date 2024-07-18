import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

client = MongoClient('mongodb://localhost:27017/')
db = client['wikipedia']
collection = db['embeddings']

model = SentenceTransformer('all-mpnet-base-v2')

urls = [
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://logbinary.com/"
]

def store_embedding(url, text):
    embedding = model.encode(text).tolist()
    collection.insert_one({
        'url': url,
        'text': text,
        'embedding': embedding
    })

def process_and_store_url(urls):
    for url in urls:
        existing_doc = collection.find_one({'url': url})
        if existing_doc:
            print(f"Document for {url} already exists. Skipping..")
            continue
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text()
        
            store_embedding(url, text)
            print(f"Successfully processed and inserted document for {url}")

        except requests.exceptions.RequestException as e:
            print(f"Error fetching content from {url}: {e}")
        except Exception as e:
            print(f"Error processing {url}: {e}")

process_and_store_url(urls)
print("Data initialization completed.")

client.close() 