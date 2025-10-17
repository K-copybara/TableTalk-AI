from app.schemas.events import StoreEnvelope, MenuEnvelope
from app.services.vectorstore_service import vectorstore_service

def handle_store(raw_event: dict): # 상점 관련 변경
    event = StoreEnvelope.model_validate(raw_event)
    if event.eventType == "DELETED":
        vectorstore_service.delete_store(event.storeId)
    else:
        if not event.store:
            raise ValueError("store payload missing")
        vectorstore_service.upsert_store_info(event.storeId, event.store.model_dump())

def handle_menu(raw_event: dict): # 메뉴 관련 변경
    event = MenuEnvelope.model_validate(raw_event)
    if event.eventType == "DELETED":
        vectorstore_service.delete_menu(event.storeId, event.menuId)
    elif event.eventType == "CREATED":
        if not event.menu: raise ValueError("menu payload missing")
        vectorstore_service.add_menu(event.storeId, event.menuId, event.menu.model_dump())
    elif event.eventType == "UPDATED":
        if not event.menu: raise ValueError("menu payload missing")
        vectorstore_service.update_menu(event.storeId, event.menuId, event.menu.model_dump())
    else:
        raise ValueError(f"unknown event_type: {event.eventType}")