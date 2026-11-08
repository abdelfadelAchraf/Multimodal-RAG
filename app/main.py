
from fastapi import FastAPI

from app.core.config import settings
from app.api.routes import health

#  Create our FastAPI app instance 
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    debug=settings.DEBUG,
    description="A Multimodal RAG Knowledge Assistant"
)


# Register API routers
app.include_router(
    health.router,
    prefix=settings.API_V1_PREFIX
)


@app.get("/")
async def root():
    return {
        "message": "Welcome to multimodal rag API",
        "version": settings.APP_VERSION
    }
