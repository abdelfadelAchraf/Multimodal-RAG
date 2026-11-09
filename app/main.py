
from fastapi import FastAPI

from app.core.config import settings
from app.api.routes import health , documents , search, chat
from fastapi.middleware.cors import CORSMiddleware
#  Create our FastAPI app instance 
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    debug=settings.DEBUG,
    description="A Multimodal RAG Knowledge Assistant"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
app.include_router(
    health.router,
    prefix=settings.API_V1_PREFIX
)
app.include_router(
    documents.router,
    prefix=settings.API_V1_PREFIX
)
app.include_router(
    search.router,
    prefix=settings.API_V1_PREFIX
)
app.include_router(
    chat.router,
    prefix=settings.API_V1_PREFIX
)



@app.get("/")
async def root():
    return {
        "message": "Welcome to multimodal rag API",
        "version": settings.APP_VERSION
    }
