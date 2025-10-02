from fastapi import APIRouter, HTTPException, status
from typing import Dict

from app.schemas.storeSchema import StoreSchema, MenuSchema
from app.services.vectorstore_service import vectorstore_service

router = APIRouter()

@router.post("/{store_id}/info", status_code=status.HTTP_201_CREATED)
async def upsert_store_info(store_id: int, info_data: StoreSchema) -> Dict[str, str]:
    """가게 정보를 생성하거나 업데이트"""
    try:
        doc_id = vectorstore_service.upsert_store_info(store_id, info_data.model_dump())
        return {"message": "Store info upserted successfully", "doc_id": doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upsert store info: {e}")

@router.post("/{store_id}/menus/{menu_id}", status_code=status.HTTP_201_CREATED)
async def add_menu(store_id: int, menu_id:int, menu_data: MenuSchema) -> Dict[str, str]:
    """가게에 새로운 메뉴를 추가"""
    try:
        doc_id = vectorstore_service.add_menu(store_id, menu_id, menu_data.model_dump())
        return {"message": "Menu added successfully", "doc_id": doc_id}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Failed to add menu: {e}")

@router.put("/{store_id}/menus/{menu_id}")
async def update_menu(store_id: int, menu_id: int, menu_data: MenuSchema) -> Dict[str, str]:
    """특정 메뉴의 정보를 업데이트"""
    try:
        doc_id = vectorstore_service.update_menu(store_id, menu_id, menu_data.model_dump())
        return {"message": "Menu updated successfully", "doc_id": doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update menu: {e}")

@router.delete("/{store_id}/menus/{menu_id}", status_code=status.HTTP_200_OK)
async def delete_menu(store_id: int, menu_id: int) -> Dict[str, str]:
    """특정 메뉴를 삭제"""
    try:
        doc_id = vectorstore_service.delete_menu(store_id, menu_id)
        return {"message": "Menu deleted successfully", "doc_id": doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete menu: {e}")

@router.delete("/{store_id}", status_code=status.HTTP_200_OK)
async def delete_store(store_id: int) -> Dict[str, str]:
    """가게와 관련된 모든 정보(가게 정보, 모든 메뉴)를 삭제"""
    try:
        vectorstore_service.delete_store(store_id)
        return {"message": f"Store {store_id} and all its data deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete store: {e}")