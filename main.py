from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone
import os

app = FastAPI()

# OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))

# Model for incoming article
class KBArticle(BaseModel):
    id: str
    title: str
    content: str


@app.get("/")
def root():
    return {"status": "KB Sync API running"}


@app.post("/kb-sync")
def kb_sync(article: KBArticle):

    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=article.content
    )

    vector = embedding.data[0].embedding

    index.upsert(
        vectors=[
            {
                "id": article.id,
                "values": vector,
                "metadata": {
                    "title": article.title
                }
            }
        ]
    )

    return {"status": "synced"}
