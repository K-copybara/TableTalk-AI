from typing import Optional, List, Literal
from pydantic import BaseModel, Field
from app.schemas.storeSchema import StoreSchema, MenuSchema

EventType = Literal["CREATED", "UPDATED", "DELETED"]

class StoreEnvelope(BaseModel):
    storeId: int
    eventType: EventType
    timestamp: str
    store: Optional[StoreSchema] = None   # DELETED면 없음

class MenuEnvelope(BaseModel):
    storeId: int
    menuId: int
    eventType: EventType
    timestamp: str
    menu: Optional[MenuSchema] = None     # DELETED면 없음