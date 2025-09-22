import pprint
from app.services.vectorstore_service import vectorstore_service

# --- 확인할 가게 ID ---
STORE_ID_TO_CHECK = 1
# ---------------------

def check_database_status(store_id: int):
    """
    VectorStoreService를 사용해 특정 가게의 데이터를 조회하고 출력합니다.
    """
    print("=" * 50)
    print(f"🔍 데이터베이스에서 Store ID: {store_id} 의 정보를 조회합니다...")
    print("=" * 50)

    try:
        data = vectorstore_service.get_documents_by_store_id(store_id)
        
        ids = data.get('ids', [])
        metadatas = data.get('metadatas', [])
        
        if not ids:
            print(f"❌ Store ID: {store_id} 에 해당하는 데이터가 없습니다.")
            return

        print(f"✅ 총 {len(ids)}개의 문서를 찾았습니다.\n")

        # 보기 좋게 분리해서 출력
        store_info_count = 0
        menu_count = 0
        
        for i, metadata in enumerate(metadatas):
            doc_type = metadata.get('type')
            if doc_type == 'store_info':
                store_info_count += 1
                print(f"--- 🏢 가게 정보 ---")
                pprint.pprint(metadata)
                print("-" * 20)
            elif doc_type == 'menu':
                if menu_count == 0:
                    print(f"--- 🍔 메뉴 정보 ---")
                menu_count += 1
                pprint.pprint(metadata)
                print("-" * 20)

        print(f"\n📄 요약: 가게 정보 {store_info_count}개, 메뉴 {menu_count}개")

    except Exception as e:
        print(f"🚨 데이터 조회 중 오류 발생: {e}")

if __name__ == "__main__":
    check_database_status(STORE_ID_TO_CHECK)
