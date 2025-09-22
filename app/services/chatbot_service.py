import os
import ast
import logging
from typing import Optional, List, Dict, Any, TypedDict, Literal

# LangChain 및 LangGraph 관련 라이브러리
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

# 외부 서비스 및 설정
from app.services.vectorstore_service import VectorStoreService, vectorstore_service
from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 1. 대화 기록(History) 관리 ---
# 세션별 대화 기록을 인메모리에 저장합니다. 운영 환경에서는 Redis 등으로 교체할 수 있습니다.
_session_histories = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _session_histories:
        _session_histories[session_id] = ChatMessageHistory()
    return _session_histories[session_id]


# --- 2. LangGraph 상태 및 의도 정의 ---

class ChatState(TypedDict, total=False):
    """LangGraph의 각 노드를 거치며 전달될 상태 객체"""
    input: str
    store_id: int
    chat_history: List[BaseMessage]
    intent: str
    params: Dict[str, Any]
    prev_params: Optional[Dict[str, Any]]
    excluded_allergens: List[str]
    required_allergens: List[str]
    result: Any
    docs: List[Any] #검색된 문서를 임시 저장

class Intent(BaseModel):
    """LLM이 사용자의 의도를 분류하고 파라미터를 추출하기 위한 Pydantic 모델"""
    kind: Literal[
        "add_to_cart", "send_request", "get_information", "allergy_filtering", 
        "get_store_info", "get_menu_detail"
    ]
    menu_name: Optional[str] = None
    quantity: Optional[int] = None
    query: Optional[str] = None
    max_price: Optional[int] = None
    is_popular: Optional[bool] = None
    user_allergy_request: Optional[str] = None
    allergy_mode: Optional[Literal["include", "exclude"]] = "exclude"
    excluded_menus: Optional[List[str]] = None
    request_message: Optional[str] = None


# --- 3. 핵심 서비스 클래스 ---

class ChatbotService:
    """LangGraph 기반의 챗봇 로직을 총괄하는 서비스 클래스"""

    def __init__(self, vs:VectorStoreService):

        self.vectorstore_service = vs
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        self.find_menu_documents = tool(self._find_menu_documents)
        self.map_allergens_to_ingredients = tool(self._map_allergens_to_ingredients)
        self.add_to_cart = tool(self._add_to_cart)
        self.send_request_to_store = tool(self._send_request_to_store)
        self.get_store_info = tool(self._get_store_info)
        self.get_menu_detail = tool(self._get_menu_detail)

        # 나중에 agent에 노출하려면 이 리스트 사용
        self.tools = [
            self.find_menu_documents,
            self.map_allergens_to_ingredients,
            self.add_to_cart,
            self.send_request_to_store,
            self.get_store_info,
            self.get_menu_detail,
        ]

        # --- LangGraph 워크플로우 빌드 ---
        workflow = StateGraph(ChatState)

        workflow.add_node("classify", self.classify_node)
        workflow.add_node("add_to_cart", self.add_to_cart_node)
        workflow.add_node("send_request", self.send_request_node)
        workflow.add_node("get_information", self.get_information_node)
        workflow.add_node("allergy_map", self.allergy_map_node)
        workflow.add_node("generate_response", self.generate_response_node)
        workflow.add_node("get_store_info", self.get_store_info_node)
        workflow.add_node("get_menu_detail", self.get_menu_detail_node)

        workflow.set_entry_point("classify")

        workflow.add_conditional_edges(
            "classify",
            self._route_from_classify,
            {
                "add_to_cart": "add_to_cart",
                "send_request": "send_request",
                "get_information": "get_information",
                "allergy_filtering": "allergy_map",
                "get_store_info": "get_store_info",
                "get_menu_detail": "get_menu_detail"
            }
        )

        workflow.add_edge("allergy_map", "get_information") # 알레르기 변환 후에는 항상 추천 노드로 이동
        workflow.add_edge("get_information", "generate_response")

        workflow.add_edge("add_to_cart", END)
        workflow.add_edge("send_request", END)
        workflow.add_edge("generate_response", END)
        workflow.add_edge("get_store_info", END)
        workflow.add_edge("get_menu_detail", END)

        checkpointer = MemorySaver()
        self.graph = workflow.compile(checkpointer=checkpointer)

        self.graph_with_history = RunnableWithMessageHistory(
            self.graph,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="result",
        )

    # --- 도구(Tool) 정의 ---
    def _find_menu_documents(self,
        store_id: int, query: str, max_price: Optional[int] = None, 
        excluded_allergens: Optional[List[str]] = None, required_allergens: Optional[List[str]] = None,
        excluded_menus: Optional[List[str]] = None
    ):
        """사용자의 조건에 맞는 메뉴 Document 리스트를 데이터베이스에서 검색하여 반환"""
        filters = []
        if max_price is not None:
            filters.append({"price": {"$lte": max_price}})

        combined_filter = {"$and": filters} if filters else None
        print(f"사전 필터 생성 VectorDB 필터 조건: {combined_filter}")

        candidate_docs = []
        search_query = query if query else "맛있는 메뉴 추천"
        # 추후 인기도 반영 로직 필요

        candidate_docs = self.vectorstore_service.search_menus(
            store_id=store_id,
            query=search_query,
            filter_dict=combined_filter,
            k=10
        )
        print(f"📚 [VectorDB 검색] 쿼리: '{search_query}', 필터: {combined_filter}")

        final_results = []
        required_allergens_set = set(required_allergens or [])
        excluded_allergens_set = set(excluded_allergens or [])
        excluded_menus_set = set(excluded_menus or [])

        print(f"✅ 포함 조건: {sorted(list(required_allergens_set))} / 🙅 제외 조건: {sorted(list(excluded_allergens_set))}")
        print(f"🙅 이름 제외 조건: {sorted(list(excluded_menus_set))}")

        for doc in candidate_docs:
            menu_name = doc.metadata.get("menu_name", "")
            if any(keyword in menu_name for keyword in excluded_menus_set if keyword):
                continue

            allergens_str = doc.metadata.get("allergens", "")
            menu_allergens = {allergen.strip() for allergen in allergens_str.split(',')} if allergens_str else set()

            if required_allergens_set and not menu_allergens.intersection(required_allergens_set):
                continue

            if excluded_allergens_set and menu_allergens.intersection(excluded_allergens_set):
                continue

            final_results.append(doc)

        if not final_results:
            return [] 
        return final_results

    
    def _map_allergens_to_ingredients(self, store_id: int, user_allergy_request: str) -> List[str]:
        """사용자가 언급한 자연어 알레르기 요청을 실제 재료 목록으로 매핑"""
        available_allergens = self.vectorstore_service.get_all_allergens(store_id)
        prompt = (
            f"사용자 알레르기 요청: \"{user_allergy_request}\"\n"
            f"가게 알레르겐 라벨 목록: {available_allergens}\n\n"
            "위 목록에 존재하는 라벨만 사용하여 관련 알레르겐을 파이썬 리스트로 반환하세요.\n"
            "오직 리스트만 출력하세요. 예: ['대두','땅콩']"
        )
        #추후 JSON + json.loads로 안정화 필요

        print(f"🧠 [LLM 호출] 알레르겐 변환: '{user_allergy_request}' -> ? (대상: {available_allergens})")
        response = self.llm.invoke(prompt)
        llm_output = response.content.strip() # type: ignore
        print(f"🧠 [LLM 응답] 알레르겐 변환: '{user_allergy_request}' -> {llm_output}")

        try:
            ingredients = ast.literal_eval(llm_output)

            if isinstance(ingredients, list):
                return ingredients
            else:
                print(f"⚠️ [파싱 경고] LLM 응답이 리스트 형태가 아닙니다: {llm_output}")
                return []
        except (SyntaxError, ValueError) as e:
            # LLM 응답이 파싱 불가능한 형태일 경우 (예: "['땅콩', '잣'")
            logger.info(f"🚨 [파싱 오류] LLM 응답을 파싱할 수 없습니다: {llm_output}, 오류: {e}")
            return []

    def _add_to_cart(self, store_id: int, menu_name: str, quantity: int):
        """사용자가 말한 메뉴 이름을 기반으로 ID를 찾고, 장바구니에 추가하는 모든 과정을 처리합니다."""
        print(f"📚 [VectorDB 조회] 메뉴 ID 검색 -> 가게 id: {store_id}, 이름: '{menu_name}'")

        docs = self.vectorstore_service.find_document(
                query=menu_name,
                store_id=store_id,
                type="menu"
            )

        if docs:
            menu_name = docs[0].metadata.get("menu_name")
            menu_id = docs[0].metadata.get("menu_id")

            if menu_id is not None:
                logger.info(f"🔍 [검색 결과] '{menu_name}'의 ID는 {menu_id} 입니다.")
                logger.info(f"✅ [API 호출] 아이디: {menu_id}, 수량: {quantity} -> 장바구니 추가")
                # 실제 백엔드 API 호출 로직
                return f"'{menu_name}' {quantity}개를 장바구니에 성공적으로 추가했습니다."

        logger.info(f"⚠️ [검색 실패] '{menu_name}' 메뉴를 찾을 수 없습니다.")
        return f"'{menu_name}'이라는 메뉴를 찾을 수 없습니다. 메뉴 이름을 다시 확인해주세요."

    def _send_request_to_store(self, request_message: str):
        """물티슈, 앞치마 등 고객의 요청사항을 가게에 전달합니다."""
        logger.info(f"✅ [API 호출] 요청사항: '{request_message}' -> 가게 전달")
        return f"'{request_message}' 요청을 가게에 전달했습니다."

    def _get_store_info(self, store_id: int) -> str:
        """가게 정보를 불러옵니다."""
        docs = self.vectorstore_service.find_document(
                query="가게 정보",
                store_id=store_id,
                type="store_info"
            )
        if docs:
            content = docs[0].page_content
            print(f"가게 정보: {content}")
            return content

        return ""
    
    def _get_menu_detail(self, store_id: int, menu_name: str) -> str:
        """메뉴 정보를 불러옵니다."""
        docs = self.vectorstore_service.find_document(
                query=menu_name,
                store_id=store_id,
                type="menu",
                k=5
            )
        if docs:
            content = docs[0].page_content
            print(f"메뉴 정보: {content}")
            return content

        return ""
    

        

    # --- LangGraph 노드 함수 (클래스 메서드로 변환) ---

    def classify_node(self, state: ChatState) -> Dict[str, Any]:
        """LLM을 사용해 사용자의 의도를 분류하고 파라미터를 추출"""
        hist = self._render_history(state.get("chat_history", []))
        prev_params = state.get("prev_params")

        system = (
            "당신은 음식점 고객응대 보조입니다. 사용자의 요청 의도를 아래 5종 중 하나로 분류하세요.\n"
            "- add_to_cart: 장바구니에 메뉴 추가\n"
            "- send_request: 가게에 대한 요청사항 전달\n"
            "- get_information: 특정 조건(가격, 인기, 추가 묘사 등)에 맞는 메뉴 추천\n"
            "- allergy_filtering: 특정 재료의 포함/미포함 여부로 메뉴 정보 요청\n"
            "- get_store_info: 가게의 이름, 설명, 영업시간, 브레이크타임 등 가게에 대한 정보 요청\n"
            "- get_menu_detail: 특정 메뉴의 가격/맵기/알레르겐/설명에 대한 요청\n"
            "알레르기 분류 규칙:\n"
            "- 사용자가 재료가 **'들어간', '포함된'** 메뉴를 찾으면, `kind`는 'allergy_filtering'로, `allergy_mode`는 'include'로 설정하세요.\n"
            "- 사용자가 재료가 **'없는', '안 들어간', '뺀'** 메뉴를 찾으면, `kind`는 'allergy_filtering'로, `allergy_mode`는 'exclude'로 설정하세요.\n\n"
            "**예시:**\n"
            "- 사용자 입력: '콩 들어간 메뉴 뭐있어?' -> kind: 'allergy_filtering', user_allergy_request: '콩', allergy_mode: 'include'\n"
            "- 사용자 입력: '콩 안들어간 메뉴 추천해줘' -> kind: 'allergy_filtering', user_allergy_request: '콩', allergy_mode: 'exclude'"
            "사용자가 '그거', '저거', '첫번째꺼', '마지막 메뉴' 등 대명사나 지시어를 사용하여 요청할 경우, 반드시 'chat_history'를 참고하여 그것이 어떤 특정 메뉴를 지칭하는지 명확히 파악해야 합니다."
            "조건 설정 규칙:\n"
            "1. 새로운 요청이 이전 조건과 연관이 없다면, 이전 조건을 완전히 무시하고 새로운 조건만 생성하세요.\n"
            "2. 새로운 요청이 이전 조건을 수정하거나 추가하는 것이라면, 이전 조건을 계승하여 변경된 부분만 반영하세요.\n"
            "3. 메뉴 추천 후 사용자가 '그거 말고', '다른 거', '~ 빼고', '~ 제외하고' 등의 표현으로 같은 조건으로 다시 추천을 요청한다면,"
            "`chat_history`에서 이전에 추천했던 메뉴 이름을 정확히 찾아내 `excluded_menus` 리스트에 담아주세요.\n"
            "4. 사용자의 요청이 모호하면 `chat_history`를 참고하여 의도를 명확히 하세요.\n\n"
            "--- 예시 ---\n"
            "- 이전 조건: {'query': '안 매운 메뉴'}\n"
            "- 새로운 요청: '그럼 10000원 이하인 걸로 찾아줘'\n"
            "- 갱신된 최종 조건 -> {'query': '안 매운 메뉴', 'max_price': 10000}\n\n"
            "필요한 모든 파라미터는 JSON으로 출력해야 합니다."
        )
        print(f"🤖 의도 분류 시도")
        classifier = self.llm.with_structured_output(Intent, method="function_calling")
        intent: Intent = classifier.invoke([
                {"role": "system", "content": system},
                {"role": "system", "content": f"이전 조건: {prev_params if prev_params else '없음'}"},
                {"role": "system", "content":f"chat_history:\n{hist}"},
                {"role": "user", "content": state['input']}
            ])
        print(f"🤖 분류된 의도: {intent.kind}")
        print(f"⚙️ 추출된 파라미터: {intent.model_dump(exclude_none=True)}")

        params = intent.model_dump(exclude_none=True)
        if "query" not in params or not params["query"]:
            params["query"] = state.get("input", "").strip()

        return{
            "intent" : intent.kind,
            "params" : intent.model_dump(exclude_none=True),
        }
    
    def add_to_cart_node(self, state: ChatState) -> Dict[str, Any]:
        print("--- 3. 장바구니 추가 (ADD TO CART) ---")
        p = state["params"]
        quantity = p.get("quantity", 1)

        names: List[str] = []
        if p.get("menu_name"):
            names = [p["menu_name"]]
        else:
            prev_docs = state.get("docs")
            if isinstance(prev_docs, list):
                names = [
                    d.metadata.get("menu_name")
                    for d in prev_docs
                    if isinstance(d, Document) and d.metadata.get("menu_name")
                ]

        if not names:
                return {
                    "result": "어떤 메뉴를 담을지 명확하게 말씀해주시겠어요? 😊"
                }

        msgs = []
        for name in names:
                res = self.add_to_cart.invoke({
                    "store_id": state["store_id"],
                    "menu_name": name,
                    "quantity": quantity
                })
                msgs.append(res)

        return {"result": "\n".join(msgs)}

    def send_request_node(self, state: ChatState) -> Dict[str, Any]:
        print("--- 4. 요청 전달 (SEND REQUEST) ---")
        msg = state["params"].get("request_message") or state["input"]
        result = self.send_request_to_store.invoke({"request_message": msg})
        return {"result": result}
    
    def generate_response_node(self, state: ChatState) -> Dict[str,Any]:
        """검색된 정보와 사용자 질문을 바탕으로 지정된 형식에 맞춰 최종 답변을 생성합니다."""
        print("--- 최종 답변 생성 (GENERATE) ---")

        retrieved_docs = state["result"]
        params = state.get("params",{})

        if not retrieved_docs:
            return {"result": "아쉽지만 해당 조건에 맞는 메뉴를 찾지 못했습니다. 다른 조건으로 질문해주시겠어요?"}

        info_text = retrieved_docs[0].page_content

        query = state["input"]

        prompt = f"""당신은 고객에게 메뉴를 안내하는 친절한 음식점 점원입니다.
            아래 '검색된 정보'를 바탕으로 [출력 형식]을 준수하되, 자연스러운 문장으로 생성하세요.
            '알레르기를 유발할 수 있는 재료'는 질문과 관련있는 경우에만 답변에 포함하세요.

            [사용자의 질문]
            {query}

            [검색된 메뉴 정보]
            {info_text}

            [출력 형식]
            {{질문 조건}}에 해당하는 메뉴는 {{메뉴 이름}}이(가) 있습니다!
            가격은 {{가격}}원이며, {{메뉴 설명}}.

            ---
            (만약 추천할 메뉴가 여러 개라면 위 형식을 반복하여 응답을 생성합니다.)
            ---

            [답변]
            """

        response = self.llm.invoke(prompt)
        return {"result": response.content}
    
    def get_information_node(self, state: ChatState) -> Dict[str, Any]:
        print("--- 5. 정보 검색 (get_information) ---")
        p = state["params"]

        raw_query = p.get("query")
        user_text = state.get("input", "")
        query = (raw_query or user_text or "메뉴 추천").strip()

        res = self.find_menu_documents.invoke({
                "store_id":state["store_id"],
                "query":query,
                "max_price":p.get("max_price"),
                "is_popular":p.get("is_popular", False),
                "excluded_allergens":state.get("excluded_allergens"),
                "required_allergens":state.get("required_allergens"),
                "excluded_menus": p.get("excluded_menus")
            })
        return {"result": res, "docs": res}
    
    def allergy_map_node(self, state: ChatState) -> Dict[str, Any]:
        print("--- 6. 알레르기 변환 (ALLERGY MAP) ---")
        p = state["params"]
        user_req = p.get("user_allergy_request") or state["input"]
        mode = p.get("allergy_mode", "exclude")

        mapped = self.map_allergens_to_ingredients.invoke({
                "store_id": state["store_id"],
                "user_allergy_request": user_req
            })

        if mode == "include":
            return {"required_allergens": mapped}
        else:
            return {"excluded_allergens": mapped}
        
    def get_store_info_node(self, state: ChatState) -> Dict[str,Any]:
        query = state["input"]
        print("사용자 질문: ", query)
        store_info = self.get_store_info.invoke({"store_id": state["store_id"]})

        prompt = f"""당신은 고객에게 가게 정보를 안내하는 친절한 음식점 점원입니다.
            아래 '가게 정보'에서 사용자의 질문에 해당하는 정보를 찾아 답변을 생성하세요.
            사용자의 질문과 관련없는 정보는 답변에 포함하지 마세요.
            사용자의 질문과 관련된 정보를 찾지 못한 경우 '죄송합니다. 해당 정보를 찾지 못했습니다.'라고 답변하세요.

            [가게 정보]
            {store_info}

            [사용자의 질문]
            {query}

            [답변]
            """

        response = self.llm.invoke(prompt)
        return {"result": response.content}

    def get_menu_detail_node(self, state: ChatState) -> Dict[str,Any]:
        query = state["input"]
        print("사용자 질문: ", query)

        menu_name = state["params"].get("menu_name")
        if not menu_name and isinstance(state.get("docs"), list):
                # 직전 결과에서 menu_name 추출
                for d in state["docs"]:
                    if d.metadata.get("menu_name"):
                        menu_name = d.metadata["menu_name"]
                        break

        if not menu_name:
            return {"result": "어떤 메뉴를 말씀하시는지 다시 한 번 알려주세요."}
        menu_detail = self.get_menu_detail.invoke({"store_id": state["store_id"], "menu_name":menu_name})

        prompt = f"""당신은 고객에게 메뉴 정보를 안내하는 친절한 음식점 점원입니다.
            아래 '메뉴 정보'에서 사용자의 질문에 해당하는 정보를 찾아 답변을 생성하세요.
            사용자의 질문과 관련없는 정보는 답변에 포함하지 마세요.
            사용자의 질문과 관련된 정보를 찾지 못한 경우 '죄송합니다. 해당 정보를 찾지 못했습니다.'라고 답변하세요.
            '알레르기 유발 재료'는 특정 재료에 대해 묻는 질문인 경우에만 활용하세요.

            [메뉴 정보]
            {menu_detail}

            [사용자의 질문]
            {query}

            [답변]
            """

        response = self.llm.invoke(prompt)
        return {"result": response.content}

    def _route_from_classify(self, state: ChatState):
        return state["intent"]
    
    def _render_history(self, messages: List[BaseMessage], k: int = 6) -> str:
        simple = []
        for m in messages[-k:]:
            role = getattr(m, "type", getattr(m, "role", "user"))
            content = getattr(m, "content", "")
            simple.append(f"{role}: {content}")
        return "\n".join(simple)
    
    # --- Public 메서드 ---

    def process_chat(self, session_id: str, user_input: str, store_id: int) -> str:
        """
        사용자 입력을 받아 전체 챗봇 플로우를 실행하고 최종 답변을 반환합니다.
        """
        # LangGraph 실행에 필요한 초기 상태값 설정
        current_state = self.graph.get_state(config={"configurable": {"thread_id": session_id, "session_id": session_id}})
        prev_params = current_state.values.get("params") if current_state else None
        
        state = {"input": user_input, "store_id": store_id, "prev_params": prev_params}
        
        # 대화 기록이 포함된 그래프 실행
        output = self.graph_with_history.invoke(
            state,
            config={"configurable": {
            "thread_id": session_id,  # LangGraph의 상태(state) 저장용
            "session_id": session_id   # RunnableWithMessageHistory의 대화 기록 저장용
        }}
        )
        return output.get("result", "오류가 발생했습니다.")

chatbot_service = ChatbotService(vs=vectorstore_service)