# import json, threading
# from confluent_kafka import Consumer
# from app.kafka.config import build_consumer_conf
# from app.kafka.handlers import handle_store, handle_menu
#
# TOPICS = ["store-updated", "menu-updated"]
# _stop_event = threading.Event()
#
# def _consume_messages():
#     consumer = Consumer(build_consumer_conf())
#     consumer.subscribe(TOPICS)
#
#     try:
#         while not _stop_event.is_set():
#             msg = consumer.poll(1.0)
#             if msg is None:
#                 continue
#             if msg.error():
#                 print(f"[KAFKA][ERR] {msg.error()}")
#                 continue
#             topic = msg.topic()
#             try:
#                 raw_event = json.loads(msg.value().decode("utf-8"))
#                 print(f"[CONSUMED] topic={topic} partition={msg.partition()} offset={msg.offset()} key={msg.key()}")
#                 if topic == "store-updated":
#                     handle_store(raw_event)
#                 elif topic == "menu-updated":
#                     handle_menu(raw_event)
#                 else:
#                     print(f"[WARN] unknown topic: {topic}")
#             except Exception as e:
#                 print(f"[KAFKA][HANDLER][ERR] {e} value={msg.value()}")
#     finally:
#         consumer.close()
#         print("[KAFKA] consumer closed")
#
# def start_consumer(app):
#     _stop_event.clear()
#     thread = threading.Thread(target=_consume_messages, daemon=True)
#     thread.start()
#     app.state.kafka_thread = thread
#     print("🚀 AI Kafka Consumer started.")
#
# def stop_consumer(app):
#     _stop_event.set()
#     thread = getattr(app.state, "kafka_thread", None)
#     if thread:
#         thread.join(timeout=10)
#     print("🛑 AI Kafka Consumer stopped.")


# app/kafka/consumer.py (가칭)
import os, json, threading, sys
from confluent_kafka import Consumer
from app.kafka.config import build_consumer_conf
from app.kafka.handlers import handle_store, handle_menu

# ✔️ 토픽을 환경변수에서 파싱 (기본값 유지)
TOPICS = [t.strip() for t in os.getenv("KAFKA_TOPICS", "store-updated,menu-updated").split(",") if t.strip()]

_stop_event = threading.Event()

def _consume_messages():
    conf = build_consumer_conf()
    consumer = Consumer(conf)

    def on_assign(c, parts):
        print(f"[KAFKA] joined group='{conf.get('group.id')}' assignment={[ (p.topic, p.partition) for p in parts ]}")
        sys.stdout.flush()

    consumer.subscribe(TOPICS, on_assign=on_assign)
    print(f"[KAFKA] subscribe topics={TOPICS} bootstrap={conf.get('bootstrap.servers')} proto={conf.get('security.protocol')} mech={conf.get('sasl.mechanism')}")
    sys.stdout.flush()

    try:
        empty_ticks = 0
        while not _stop_event.is_set():
            msg = consumer.poll(1.0)
            if msg is None:
                empty_ticks += 1
                if empty_ticks in (5, 30):  # 초반/지속 무소비 힌트 로그
                    print(f"[KAFKA] no message yet (ticks={empty_ticks})")
                    sys.stdout.flush()
                continue

            empty_ticks = 0
            if msg.error():
                print(f"[KAFKA][ERR] {msg.error()}")
                sys.stdout.flush()
                continue

            topic = msg.topic()
            try:
                raw = msg.value()
                raw_event = json.loads(raw.decode("utf-8") if raw else "{}")
                print(f"[CONSUMED] topic={topic} partition={msg.partition()} offset={msg.offset()} key={msg.key()}")
                sys.stdout.flush()

                if topic == "store-updated":
                    handle_store(raw_event)
                elif topic == "menu-updated":
                    handle_menu(raw_event)
                else:
                    print(f"[KAFKA][WARN] unknown topic: {topic}")
                    sys.stdout.flush()

            except Exception as e:
                print(f"[KAFKA][HANDLER][ERR] {e} value={msg.value()}")
                sys.stdout.flush()
    finally:
        consumer.close()
        print("[KAFKA] consumer closed")
        sys.stdout.flush()

def start_consumer(app):
    _stop_event.clear()
    thread = threading.Thread(target=_consume_messages, daemon=True)
    thread.start()
    app.state.kafka_thread = thread
    print("🚀 AI Kafka Consumer started.")

def stop_consumer(app):
    _stop_event.set()
    thread = getattr(app.state, "kafka_thread", None)
    if thread:
        thread.join(timeout=10)
    print("🛑 AI Kafka Consumer stopped.")
