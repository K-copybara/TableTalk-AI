import os

def build_consumer_conf():
    return {
        "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP"),
        "security.protocol": "SASL_SSL",
        "sasl.mechanisms": "SCRAM-SHA-512",
        "sasl.username": os.getenv("KAFKA_USERNAME"),
        "sasl.password": os.getenv("KAFKA_PASSWORD"),
        "ssl.ca.location": os.getenv("KAFKA_CA_PATH", "/certs/ca.crt"),
        "group.id": "ai-service-group",
        "auto.offset.reset": "earliest",
    }