import os

def build_consumer_conf():
    return {
        "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", ""),
        "security.protocol": os.getenv("KAFKA_SECURITY_PROTOCOL", "SASL_SSL"),
        "sasl.mechanism": os.getenv("KAFKA_SASL_MECHANISM", "SCRAM-SHA-512"),
        "sasl.username": os.getenv("KAFKA_USERNAME", ""),
        "sasl.password": os.getenv("KAFKA_PASSWORD", ""),
        "ssl.ca.location": os.getenv("KAFKA_SSL_CA_LOCATION", "/etc/ssl/strimzi/ca.crt"),
        "group.id": os.getenv("KAFKA_GROUP_ID", "ai-service-group"),
        "auto.offset.reset": os.getenv("KAFKA_AUTO_OFFSET_RESET", "earliest"),
        "enable.auto.commit": True,

        "client.id": os.getenv("KAFKA_CLIENT_ID", "ai-server-consumer"),
        # "debug": os.getenv("KAFKA_DEBUG", ""),  # 필요할 때만 활성화
    }