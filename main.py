from fastapi import FastAPI
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from app.kafka.consumer import start_consumer, stop_consumer
from app.api import chat, store
from check_db_status import check_database_status

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    start_consumer(app)   # 스레드 시작
    yield
    stop_consumer(app)    # 스레드 종료

app = FastAPI(lifespan=lifespan)

# app = FastAPI()
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

@app.get("/check_document/{store_id}")
async def check_document(store_id: int):
    return check_database_status(store_id)