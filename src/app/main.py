from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from app.api.routes.chat import router as chat_router
from app.services.rag import RAGService


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Inicializando RAGService...")

    app.state.rag_service = RAGService()

    logger.info("RAGService inicializado")

    yield

    logger.info("Cerrando aplicación...")
    app.state.rag_service = None


app = FastAPI(
    title="Mi Salud RAG API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(chat_router)