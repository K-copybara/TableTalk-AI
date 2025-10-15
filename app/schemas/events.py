from typing import Optional, List, Literal
from pydantic import BaseModel, Field
from app.schemas.storeSchema import StoreSchema, MenuSchema

EventType = Literal["CREATED", "UPDATED", "DELETED"]

class StoreEnvelope(BaseModel):
    store_id: int
    event_type: EventType
    timestamp: str
    store: Optional[StoreSchema] = None   # DELETED면 없음

class MenuEnvelope(BaseModel):
    store_id: int
    menu_id: int
    event_type: EventType
    timestamp: str
    menu: Optional[MenuSchema] = None     # DELETED면 없음