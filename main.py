from fastapi import FastAPI
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

from app.api import chat, store

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173",],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/chat")
app.include_router(store.router, prefix="/stores")

@app.get("/")
async def root():
    return {"message": "Welcome to the TableTalk"}