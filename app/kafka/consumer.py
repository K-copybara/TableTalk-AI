import json, threading
from confluent_kafka import Consumer
from app.kafka.config import build_consumer_conf
from app.kafka.handlers import handle_store, handle_menu

TOPICS = ["store-updated", "menu-updated"]
_stop_event = threading.Event()

def _consume_messages():
    consumer = Consumer(build_consumer_conf())
    consumer.subscribe(TOPICS)
    
    try:
        while not _stop_event.is_set():
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                print(f"[KAFKA][ERR] {msg.error()}")
                continue
            topic = msg.topic()
            try:
                raw_event = json.loads(msg.value().decode("utf-8"))
                print(f"[CONSUMED] topic={topic} partition={msg.partition()} offset={msg.offset()} key={msg.key()}")
                if topic == "store-updated":
                    handle_store(raw_event)
                elif topic == "menu-updated":
                    handle_menu(raw_event)
                else:
                    print(f"[WARN] unknown topic: {topic}")
            except Exception as e:
                print(f"[KAFKA][HANDLER][ERR] {e} value={msg.value()}")
    finally:
        consumer.close()
        print("[KAFKA] consumer closed")

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