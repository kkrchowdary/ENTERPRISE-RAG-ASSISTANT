from fastapi import FastAPI

from app.api.routes import chat, documents, health

app = FastAPI(title="Minimal RAG Assistant")
app.include_router(health.router)
app.include_router(documents.router)
app.include_router(chat.router)