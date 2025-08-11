from fastapi import FastAPI, Request
from sentence_transformers import SentenceTransformer

app = FastAPI()
model = SentenceTransformer('all-mpnet-base-v2')

@app.post('/embed')
async def embed(request: Request):
    data = await request.json()
    texts = data['texts']
    embeddings = model.encode(texts).tolist()
    return {"embeddings": embeddings}