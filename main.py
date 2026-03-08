import os
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from tqdm import tqdm
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
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
BATCH_SIZE = 100
CONCURRENT_REQUESTS = 25

# -------------------------------

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# -------------------------------
# TOKEN
# -------------------------------

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

# -------------------------------
# FETCH ARTICLE IDS
# -------------------------------

async def fetch_article_ids(session, token):

    headers = {
        "Authorization": f"Zoho-oauthtoken {token}",
        "orgId": ORG_ID
    }

    start = 1
    limit = 50
    ids = []

    print("Fetching article list...")

    while True:

        url = f"{DESK_API_DOMAIN}/articles?from={start}&limit={limit}"

        async with session.get(url, headers=headers) as resp:
            data = await resp.json()

        articles = data.get("data", [])

        if not articles:
            break

        ids.extend([a["id"] for a in articles])

        start += limit

    print("Total articles:", len(ids))

    return ids

# -------------------------------
# FETCH ARTICLE
# -------------------------------

async def fetch_article(session, article_id, headers):

    url = f"{DESK_API_DOMAIN}/articles/{article_id}"

    async with session.get(url, headers=headers) as resp:

        if resp.status != 200:
            return None

        return await resp.json()

# -------------------------------
# TEXT CLEAN
# -------------------------------

def clean_html(text):

    return BeautifulSoup(text, "html.parser").get_text(separator=" ")

# -------------------------------
# CHUNK TEXT
# -------------------------------

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

# -------------------------------
# EMBEDDING
# -------------------------------

async def embed(text):

    res = await openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return res.data[0].embedding

# -------------------------------
# PROCESS ARTICLE
# -------------------------------

async def process_article(session, article_id, headers):

    article = await fetch_article(session, article_id, headers)

    if not article:
        return []

    if article.get("status") != "Published":
        return []

    title = article.get("title", "")
    body = article.get("answer", "")

    modified = article.get("modifiedTime")

    text = clean_html(title + " " + body)

    chunks = chunk_text(text)

    vectors = []

    for i, chunk in enumerate(chunks):

        embedding = await embed(chunk)

        vectors.append({
            "id": f"{article_id}_{i}",
            "values": embedding,
            "metadata": {
                "articleId": article_id,
                "chunk": i,
                "title": title,
                "modified": modified
            }
        })

    return vectors

# -------------------------------
# SYNC
# -------------------------------

async def sync_articles():

    async with aiohttp.ClientSession() as session:

        token = await get_access_token(session)

        headers = {
            "Authorization": f"Zoho-oauthtoken {token}",
            "orgId": ORG_ID
        }

        article_ids = await fetch_article_ids(session, token)

        semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

        async def sem_task(article_id):

            async with semaphore:
                return await process_article(session, article_id, headers)

        tasks = [sem_task(aid) for aid in article_ids]

        vectors = []

        print("Processing articles...")

        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks)):

            result = await future

            vectors.extend(result)

            if len(vectors) >= BATCH_SIZE:

                index.upsert(vectors=vectors)

                vectors = []

        if vectors:
            index.upsert(vectors=vectors)

        print("KB Sync Complete")

# -------------------------------
# RUN
# -------------------------------

if __name__ == "__main__":
    asyncio.run(sync_articles())
