import os
import json
import asyncio

from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
# from confluent_kafka import Producer, Consumer, KafkaException


from app.api import chat, store

load_dotenv()

# BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP")
# USERNAME  = os.getenv("KAFKA_USERNAME")
# PASSWORD  = os.getenv("KAFKA_PASSWORD")
# CA_PATH   = os.getenv("KAFKA_CA_PATH", "/certs/ca.crt")

# producer_conf = {
#     "bootstrap.servers": BOOTSTRAP,
#     "security.protocol": "SASL_SSL",
#     "sasl.mechanisms": "SCRAM-SHA-512",
#     "sasl.username": USERNAME,
#     "sasl.password": PASSWORD,
#     "ssl.ca.location": CA_PATH,
# }

# consumer_conf = {
#     **producer_conf,
#     "group.id": "fastapi-local-test",
#     "auto.offset.reset": "earliest",
# }

# producer: Producer | None = None
# consumer: Consumer | None = None
# consumer_task: asyncio.Task | None = None

# async def consume_loop():
#     try:
#         while True:
#             msg = consumer.poll(1.0)
#             if msg is None:
#                 await asyncio.sleep(0.1)
#                 continue
#             if msg.error():
#                 print(f"Consumer error: {msg.error()}")
#                 continue

#             value = msg.value().decode("utf-8")
#             print(f"[CONSUMED] {value}")
#             # 👉 여기서 DB 저장, 비즈니스 로직 실행 등 처리
#     except asyncio.CancelledError:
#         pass

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     global producer, consumer, consumer_task

#     # --- startup ---
#     producer = Producer(producer_conf)
#     consumer = Consumer(consumer_conf)
#     consumer.subscribe(["demo"])  # 실제 토픽명으로 교체

#     loop = asyncio.get_event_loop()
#     consumer_task = loop.create_task(consume_loop())

#     print("Kafka Producer + Consumer started")
#     yield

#     # --- shutdown ---
#     if consumer:
#         consumer.close()
#     if producer:
#         producer.flush()
#     if consumer_task:
#         consumer_task.cancel()
#         try:
#             await consumer_task
#         except asyncio.CancelledError:
#             pass

#     print("Kafka Producer + Consumer stopped")

# app = FastAPI(lifespan=lifespan)

# @app.post("/publish")
# async def publish(payload: dict):
#     def delivery_report(err, msg):
#         if err is not None:
#             print(f"Delivery failed: {err}")
#         else:
#             print(f"Message delivered to {msg.topic()} [{msg.partition()}]")

#     producer.produce(
#         topic="demo",  # 실제 토픽명으로 교체
#         value=json.dumps(payload).encode("utf-8"),
#         callback=delivery_report,
#     )
#     producer.poll(0)
#     return {"ok": True}

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