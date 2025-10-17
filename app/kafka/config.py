# import os
#
# def build_consumer_conf():
#     return {
#         "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVER"),
#         "security.protocol": "SASL_PLAINTEXT",
#         "sasl.mechanisms": "SCRAM-SHA-512",
#         "sasl.username": os.getenv("KAFKA_USERNAME"),
#         "sasl.password": os.getenv("KAFKA_PASSWORD"),
#         "ssl.ca.location": os.getenv("KAFKA_CA_PATH", "/certs/ca.crt"),
#         "group.id": "ai-service-group",
#         "auto.offset.reset": "earliest",
#     }

# app/kafka/config.py
import os

def build_consumer_conf():
    return {
        # ✔️ 환경변수 이름 바로잡기
        "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", ""),

        # ✔️ TLS 사용
        "security.protocol": os.getenv("KAFKA_SECURITY_PROTOCOL", "SASL_SSL"),

        # ✔️ 정확한 키 이름 (단수)
        "sasl.mechanism": os.getenv("KAFKA_SASL_MECHANISM", "SCRAM-SHA-512"),
        "sasl.username": os.getenv("KAFKA_USERNAME", ""),
        "sasl.password": os.getenv("KAFKA_PASSWORD", ""),

        # ✔️ 컨테이너 경로 사용
        "ssl.ca.location": os.getenv("KAFKA_SSL_CA_LOCATION", "/etc/ssl/strimzi/ca.crt"),

        # ✔️ 운영에 안전한 기본값
        "group.id": os.getenv("KAFKA_GROUP_ID", "ai-service-group"),
        "auto.offset.reset": os.getenv("KAFKA_AUTO_OFFSET_RESET", "earliest"),
        "enable.auto.commit": True,

        # (선택) 튜닝/관찰용
        "client.id": os.getenv("KAFKA_CLIENT_ID", "ai-server-consumer"),
        # "debug": os.getenv("KAFKA_DEBUG", ""),  # 필요할 때만 활성화
    }
