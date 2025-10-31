import logging
import os
from typing import Dict, Any, List, Optional

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_teddynote.retrievers import KiwiBM25Retriever
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStoreService:
    """
    ChromaDB 벡터스토어와의 상호작용을 관리하는 클래스
    """
    def __init__(self):
        # 데이터베이스 파일 저장 경로
        self.persist_directory = './app/db/chroma_db'

        # OpenAI 임베딩 모델 초기화
        os.environ['OPENAI_API_KEY'] = settings.OPENAI_API_KEY
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # ChromaDB 클라이언트 초기화
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )

    def _create_store_info_document(self, store_id: int, info_data: Dict[str, Any]) -> Document:
        """가게 정보를 Langchain Document로 변환"""
        page_content = f"""가게 이름은 {info_data['name']} 입니다. {info_data['description']}
        일요일부터 토요일까지의 영업시간은 다음과 같습니다. {info_data['hours']}
        일요일부터 토요일까지의 브레이크 타임은 다음과 같습니다. {info_data['break_time']}"""

        metadata = {
            "type" : "store_info",
            "store_id": store_id,
        }

        return Document(page_content=page_content, metadata=metadata)
    
    def _create_menu_document(self, store_id:int, menu_id:int, menu_data: Dict[str, Any]) -> Document:
        """메뉴 정보를 Langchain Document로 변환"""
        allergen_list = menu_data.get('allergens', [])
        allergen_str = ",".join(allergen_list)

        SPICY_DESC = {
            0: "전혀 맵지 않은 음식입니다.",
            1: "약간 매운 맛이 있지만 많이 맵지는 않습니다.",
            2: "중간 정도로 매운 음식입니다.",
            3: "매운 맛이 강한 음식입니다.",
            4: "가장 매운 음식입니다.",
        }

        page_content = f"""메뉴의 이름은 {menu_data['menu_name']}입니다. 가격은 {menu_data['price']}원입니다.
                        {menu_data['description']}
                        {menu_data['extra_info']}
                        {SPICY_DESC[menu_data['spiciness']]}
                        알레르기를 유발할 수 있는 재료들은 다음과 같습니다. {menu_data['allergens']}
                        """
        metadata = {
            "type" : "menu",
            "store_id" : store_id,
            "menu_id" : menu_id,
            "menu_name" : menu_data['menu_name'],
            "price" : menu_data['price'],
            "allergens" : allergen_str
        }
        return Document(page_content=page_content, metadata=metadata)
    
    def upsert_store_info(self, store_id: int, info_data: Dict[str, Any]) -> str:
        """가게 정보를 벡터DB에 추가하거나 업데이트"""
        doc = self._create_store_info_document(store_id, info_data)
        doc_id = f"store_{store_id}_info"

        self.vectorstore.add_documents([doc], ids=[doc_id])
        logger.info(f"가게 정보가 성공적으로 추가/수정되었습니다: {doc_id}")
        return doc_id
    
    def upsert_menu(self, store_id:int, menu_id:int, menu_data: Dict[str, Any])->str:
        """새로운 메뉴를 벡터 DB에 추가하거나 수정"""      
        doc = self._create_menu_document(store_id, menu_id, menu_data)
        doc_id = f"store_{store_id}_menu_{menu_id}"

        self.vectorstore.add_documents([doc], ids=[doc_id])
        logger.info(f"새로운 메뉴가 성공적으로 추가/수정되었습니다: {doc_id}")
        return doc_id
    
    # def update_menu(self, store_id: int, menu_id:int, menu_data:Dict[str, Any])->str:
    #     """기존 메뉴를 벡터 DB에서 수정"""
    #     menu_data['menu_id'] = menu_id

    #     doc = self._create_menu_document(store_id, menu_id, menu_data)
    #     doc_id = f"store_{store_id}_menu_{menu_id}"

    #     self.vectorstore.update_document(document_id=doc_id, document=doc)
    #     logger.info(f"메뉴 정보가 성공적으로 수정되었습니다: {doc_id}")
    #     return doc_id

    def delete_menu(self, store_id: int, menu_id: int) -> str:
        """특정 메뉴를 벡터 DB에서 삭제"""
        doc_id = f"store_{store_id}_menu_{menu_id}"
        self.vectorstore.delete(ids=[doc_id])
        logger.info(f"메뉴가 성공적으로 삭제되었습니다: {doc_id}")
        return doc_id
    
    def delete_store(self, store_id: int) -> bool:
        """특정 가게의 모든 정보(가게 정보, 모든 메뉴)를 벡터DB에서 삭제"""
        
        self.vectorstore.delete(where={"store_id": store_id})
        logger.info(f"가게 ID {store_id}의 모든 정보가 성공적으로 삭제되었습니다.")
        return True
    
    def get_all_allergens(self, store_id: int) -> List[str]:
        """특정 가게의 모든 알레르겐 목록을 반환"""
        docs = self.vectorstore.get(
            where={"$and": [{"store_id": store_id}, {"type": "menu"}]}
        ).get("metadatas", [])

        allergens = set()

        for metadata in docs:
            allergens_str = metadata.get("allergens", "")
            if allergens_str:
                allergens_list = [allergen.strip() for allergen in allergens_str.split(',')]
                allergens.update(allergens_list)

        return list(allergens)
    
    def find_document(self, store_id: int, query: str, type: str, k=1) -> List[Document]:
        """메뉴 이름/가게 이름으로 가장 유사한 메뉴 문서 하나를 검색합니다."""
        return self.vectorstore.similarity_search(
            query=query,
            k=k,
            filter={"$and": [{"store_id": store_id}, {"type": type}]}
        )
    
    def find_conditional_document(self, store_id: int, query: str, type: str, k=1, filters: Optional[List[Dict[str, Any]]] = None,) -> List[Document]:
        """사용자의 조건에 따른 메뉴 문서를 검색합니다."""
        
        if filters is not None and not isinstance(filters, list):
            raise TypeError("filters must be a List[Dict[str, Any]] or None")
        
        base_filter: List[Dict[str, Any]] = [{"store_id": store_id}, {"type": type}]
        if(filters):
            base_filter.extend(filters)
        
        final_filter = {"$and": base_filter}
        print(final_filter)

        return self.vectorstore.similarity_search_with_relevance_scores(
            query=query,
            k=k,
            filter=final_filter
        )
    
    def ensemble_search(self, store_id: int, query: str, k: int=5, filters: Optional[List[Dict[str, Any]]]=None) -> List[Document]:
        """
        BM25 (키워드)와 Chroma (유사도) 검색을 결합한 앙상블 검색을 수행합니다.
        """
        base_filter: List[Dict[str, Any]] = [{"store_id": store_id}, {"type": "menu"}]
        if(filters):
            base_filter.extend(filters)
        final_filter = {"$and": base_filter}

        filtered_results = self.vectorstore.get(where=final_filter, include=["metadatas", "documents"])
        
        filtered_docs: List[Document] = []
        if filtered_results and filtered_results.get("ids"):
            for i in range(len(filtered_results["ids"])):
                filtered_docs.append(
                    Document(
                        page_content=filtered_results["documents"][i],
                        metadata=filtered_results["metadatas"][i]
                    )
                )
        if not filtered_docs:
            logger.info("필터 조건에 맞는 문서가 없습니다.")
            return []
        
        bm25_retriever = KiwiBM25Retriever.from_documents(filtered_docs)
        bm25_retriever.k = k

        chroma_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={'k': k, 'filter': final_filter}
        )

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, chroma_retriever],
            weights=[0.5, 0.5],
            search_type="rrf", 
        )

        results = ensemble_retriever.invoke(query)
        return results
    
    def get_documents_by_store_id(self, store_id: int) -> Dict[str, Any]:
        """
        특정 가게 ID에 해당하는 모든 문서를 가져옴 (디버깅용)
        """
        results = self.vectorstore.get(
            where={"store_id": store_id},
            include=["metadatas", "documents"] 
        )
        return results

    
vectorstore_service = VectorStoreService()