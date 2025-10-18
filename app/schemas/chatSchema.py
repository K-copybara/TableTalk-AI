import operator
from typing import Optional, List, Dict, Any, TypedDict, Union, Literal
from typing_extensions import Annotated
from pydantic import BaseModel, Field
from enum import Enum
from langchain_core.messages import BaseMessage

class ChatRequest(BaseModel):
    customer_key: str
    user_input: str
    store_id: int
    table_id: int

class ChatResponse(BaseModel):
    answer: str

# --- LangGraph 상태 및 의도 정의 ---
class Intent(str, Enum):
    """챗봇이 분류할 수 있는 사용자의 의도(Intent) 집합"""
    
    # --- 핵심 기능 의도 ---
    ADD_TO_CART = "add_to_cart"              # 1. 장바구니 추가
    SEND_REQUEST = "send_request"            # 2. 요청사항 전달
    GET_STORE_INFO = "get_store_info"        # 3. 가게 정보 제공
    GET_MENU_INFO = "get_menu_info"          # 4. 특정 메뉴 정보 제공
    RECOMMEND_MENU = "recommend_menu"        # 5. 사용자 쿼리 기반 메뉴 추천
    
    # --- 대화 관리 의도 ---
    # GREET = "greet"                        # 인사 또는 대화 시작
    CHITCHAT = "chitchat"                    # 6. 기능과 무관한 일반 대화
    CONFIRM = "confirm"                      # 사용자의 긍정 응답 (예: "네", "맞아요")
    DENY = "deny"                            # 사용자의 부정 응답 (예: "아니요", "취소")

class RouteQuery(BaseModel):
    """
    파라미터가 필요 없는 단순 의도 분류를 위한 모델.
    LLM은 다른 TaskParams가 적절하지 않을 때 이 모델을 선택해야 한다.
    """
    type: Literal["route"]
    intent: Intent = Field(description="분류된 사용자의 최종 의도")

class MenuItem(BaseModel):
    """장바구니에 담길 개별 메뉴 항목."""
    menu_name: str = Field(description="메뉴 이름")
    quantity: int = Field(default=1, description="수량")
    menu_id: Optional[int] = Field(default=None, description="메뉴 ID")

class AddToCartParams(BaseModel):
    """'장바구니 추가' 태스크에 필요한 파라미터."""
    type: Literal["add_to_cart"]
    items: List[MenuItem] = Field(description="장바구니에 추가할 메뉴 항목들의 리스트")

class SendRequestParams(BaseModel):
    """'요청사항 전달' 태스크에 필요한 파라미터."""
    type: Literal["send_request"]
    request_note: str = Field(description="가게에 전달할 요청사항 내용")

class GetMenuInfoParams(BaseModel):
    """'특정 메뉴 정보 조회' 태스크에 필요한 파라미터."""
    type: Literal["get_menu_info"]
    menu_name: str = Field(description="정보를 조회할 메뉴의 이름")


class MenuQuerySchema(BaseModel):
    """사용자의 메뉴 추천 쿼리에서 추출한 조건들을 담는 스키마."""
    query_description: Optional[str] = Field(
        default=None, description="RAG 벡터 검색에 사용할 사용자의 묘사 (예: '따뜻하고 든든한 국물 요리')"
    )
    price_max: Optional[int] = Field(default=None, description="최대 가격 제한")
    price_min: Optional[int] = Field(default=None, description="최소 가격 제한")
    
    allergies_include: Optional[List[str]] = Field(
        default=None, description="반드시 포함해야 할 알레르기 유발 물질"
    )
    allergies_exclude: Optional[List[str]] = Field(
        default=None, description="반드시 제외해야 할 알레르기 유발 물질"
    )
    menu_exclude: Optional[List[str]] = Field(
        default=None, description="추천에서 제외할 특정 메뉴 이름"
    )
    is_popular: Optional[bool] = Field(default=None, description="인기 메뉴를 원하는지 여부")
    
    # 추천 유형을 결정하는 핵심 필드들
    single_result: bool = Field(
        default=True, description="결과가 메뉴 하나로만 이루어져야 하는 경우 True"
    )
    num_food: Optional[int] = Field(
        default=None, description="추천이 필요한 음식의 개수"
    )


class ChatState(TypedDict, total=False):
    """챗봇의 전체 대화 흐름을 관리하는 상태"""
    # --- 기본 대화 관리 ---
    store_id: int
    table_id: int
    customer_key: str
    messages: Annotated[List[BaseMessage], operator.add]
    intent: Optional[Intent]
    awaiting: Optional[str] # "confirmation" 등, 사용자의 특정 응답을 기다리는 상태

    # --- API 호출 관련 ---
    task_params: Optional[Union[
        AddToCartParams,
        SendRequestParams,
        GetMenuInfoParams,
        RouteQuery
    ]] # API 호출에 필요한 파라미터 ({"menu_name": "짜장면", "quantity": 1})
    api_result: Optional[dict] # API 호출 성공/실패 결과

    # --- RAG 및 검색 결과 ---
    retrieval: Dict[str, Any] # RAG 검색 결과 {"menu": {...}, "store": {...}}
    search_results: Optional[List[Dict[str, Any]]] # 메뉴 추천 후보군 리스트

    # --- 메뉴 추천 전용 ---
    menu_query_params: Optional[MenuQuerySchema] # 메뉴 추천 조건들을 담는 객체

    # --- 최종 결과 ---
    response: Optional[str] # 사용자에게 보여줄 최종 응답 메시지