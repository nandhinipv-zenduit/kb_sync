import os
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from fastapi import FastAPI
from openai import AsyncOpenAI
from pinecone import Pinecone

# -------------------------
# CONFIG
# -------------------------

ORG_ID = "25508736"
DESK_API_DOMAIN = "https://desk.zoho.com/api/v1"

ZOHO_DESK_CLIENT_ID = os.environ["ZOHO_DESK_CLIENT_ID"]
ZOHO_DESK_CLIENT_SECRET = os.environ["ZOHO_DESK_CLIENT_SECRET"]
ZOHO_DESK_REFRESH_TOKEN = os.environ["ZOHO_DESK_REFRESH_TOKEN"]

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]

INDEX_NAME = "zenduit-kb-index"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

# -------------------------

app = FastAPI()

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# -------------------------
# TOKEN
# -------------------------

async def get_access_token(session):

    url = "https://accounts.zoho.com/oauth/v2/token"

    payload = {
        "refresh_token": ZOHO_DESK_REFRESH_TOKEN,
        "client_id": ZOHO_DESK_CLIENT_ID,
        "client_secret": ZOHO_DESK_CLIENT_SECRET,
        "grant_type": "refresh_token"
    }

    async with session.post(url, data=payload) as resp:
        data = await resp.json()
        return data["access_token"]

# -------------------------
# FETCH ARTICLE
# -------------------------

async def fetch_article(session, article_id, headers):

    url = f"{DESK_API_DOMAIN}/articles/{article_id}"

    async with session.get(url, headers=headers) as resp:

        if resp.status != 200:
            return None

        return await resp.json()

# -------------------------
# CLEAN
# -------------------------

def clean_html(text):
    return BeautifulSoup(text, "html.parser").get_text(separator=" ")

# -------------------------
# CHUNK
# -------------------------

def chunk_text(text):

    words = text.split()
    chunks = []

    start = 0

    while start < len(words):

        end = start + CHUNK_SIZE
        chunk = words[start:end]

        chunks.append(" ".join(chunk))

        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks

# -------------------------
# EMBED
# -------------------------

async def embed_batch(chunks):

    res = await openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=chunks
    )

    return [x.embedding for x in res.data]

# -------------------------
# UPDATE ARTICLE
# -------------------------

async def update_article(article_id):

    async with aiohttp.ClientSession() as session:

        token = await get_access_token(session)

        headers = {
            "Authorization": f"Zoho-oauthtoken {token}",
            "orgId": ORG_ID
        }

        article = await fetch_article(session, article_id, headers)

        if not article:
            return {"status": "not_found"}

        if article.get("status") != "Published":
            index.delete(filter={"articleId": article_id})
            return {"status": "deleted"}

        title = article.get("title", "")
        body = article.get("answer", "")
        modified = article.get("modifiedTime")

        text = clean_html(title + " " + body)

        chunks = chunk_text(text)

        embeddings = await embed_batch(chunks)

        # remove old vectors
        index.delete(filter={"articleId": article_id})

        vectors = []

        for i, emb in enumerate(embeddings):

            vectors.append({
                "id": f"{article_id}_{i}",
                "values": emb,
                "metadata": {
                    "articleId": article_id,
                    "chunk": i,
                    "title": title,
                    "modified": modified
                }
            })

        index.upsert(vectors=vectors)

        return {"status": "updated", "chunks": len(vectors)}

# -------------------------
# API ENDPOINT
# -------------------------

@app.post("/update_kb")

async def update_kb(payload: dict):

    article_id = payload.get("article_id")

    if not article_id:
        return {"error": "article_id missing"}

    result = await update_article(article_id)

    return result
