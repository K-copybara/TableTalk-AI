from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class StoreSchema(BaseModel):
    name: str
    description: str
    hours: List[Optional[str]]
    break_time: List[Optional[str]]

class MenuSchema(BaseModel):
    menu_name: str
    price: int
    description: str
    spiciness: int
    allergens: List[str]
    extra_info: str

    