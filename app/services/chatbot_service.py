import os
import ast
import logging
from typing import Optional, List, Dict, Any, Union

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
from langgraph.types import Command
from typing_extensions import Annotated
from pydantic import Field

from app.schemas.chatSchema import (
    ChatState,
    Intent,
    RouteQuery,
    AddToCartParams,
    SendRequestParams,
    GetMenuInfoParams,
)

# 외부 서비스 및 설정
from app.services.vectorstore_service import VectorStoreService, vectorstore_service
from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 1. 대화 기록(History) 관리 ---


# --- 3. 핵심 서비스 클래스 ---

class ChatbotService:
    """LangGraph 기반의 챗봇 로직을 총괄하는 서비스 클래스"""

    def __init__(self, vs:VectorStoreService):

        self.vectorstore_service = vs
        self.llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.1, max_tokens=150)

        # --- LangGraph 워크플로우 빌드 ---
        workflow = StateGraph(ChatState)

        workflow.add_node("classify_intent", self.classify_intent) # 의도 분류
        workflow.add_node("route_confirmation", self.route_confirmation) # 노드 라우팅
        workflow.add_node("ask_add_to_cart_confirmation", self.ask_add_to_cart_confirmation) # 장바구니 추가 확인
        workflow.add_node("call_add_to_cart_api", self.call_add_to_cart_api) # 장바구니 추가
        workflow.add_node("ask_request_confirmation", self.ask_request_confirmation) # 요청사항 확인
        workflow.add_node("call_request_api", self.call_request_api) # 요청사항 전송
        workflow.add_node

        workflow.set_entry_point("classify_intent") 
        workflow.add_edge("classify_intent", "route_confirmation")

        checkpointer = MemorySaver()
        self.graph = workflow.compile(checkpointer=checkpointer)

    # --- 도구(Tool) 정의 ---
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
    def classify_intent(self, state: ChatState) -> Dict[str, Any]:
        """사용자의 최신 메시지를 기반으로 의도를 분류하고, 파라미터를 추출하여 state를 업데이트합니다."""
        print(">> Node: classify_intent")

        tools = [RouteQuery, AddToCartParams, SendRequestParams, GetMenuInfoParams]
        llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
        model = llm.bind_tools(tools)

        prompt = ChatPromptTemplate.from_messages([
            ("system",
                """
                당신은 음식점 챗봇의 의도 분류기입니다.
                사용자의 최근 메시지와 이전 대화 내용을 바탕으로, 사용자의 의도를 아래 6종 중 하나로 분류하고 가장 적절한 도구(Tool)로 필요한 정보를 추출하세요.
                도구는 'type' 필드로 구분됩니다.
                type은 route|add_to_cart|send_request|get_menu_info 중 하나입니다.
                추가 파라미터가 필요 없는 경우는 route를 사용하여 intent만 반환하세요.
                - add_to_cart: 장바구니에 메뉴 추가
                - send_request: 가게에 대한 요청사항 전달
                - get_store_info: 가게에 대한 정보 제공 (가게 이름, 설명, 영업시간, 브레이크타임 등)
                - get_menu_info: 이름이 명시된 메뉴에 대한 정보 제공 (특정 메뉴의 가격/맵기/알레르기 유발 재료 등)
                - recommend_menu: 사용자의 요청에 따른 메뉴 검색 및 정보 제공
                - chitchat: 기능과 무관한 일반 대화
                - confirm: 사용자의 긍정 응답 (예: "네", "맞아요")
                - deny: 사용자의 부정 응답 (예: "아니요", "취소")
                필요한 모든 파라미터는 반드시 JSON으로 응답해야 합니다.
                """
             ),
            ("human", "이전 대화:\n{history}\n\n최신 사용자 메시지: {userInput}")
        ])

        messages = state['messages']
        history = "\n".join([f"{'User' if msg.type == 'human' else 'Bot'}: {msg.content}" for msg in messages[:-1]])
        userInput = messages[-1].content

        result = (prompt | model).invoke({"history": history, "userInput": userInput})
        type_name = result.tool_calls[0]["name"]
        params = result.tool_calls[0]["args"]

        print(result)

        if type_name == "RouteQuery":
            print(f"Intent classified as: {type_name}")
            data = RouteQuery.model_validate(params)
            return {"intent": data.intent}

        if type_name == "AddToCartParams":
            print(f"Intent classified as: {type_name}")
            data = AddToCartParams.model_validate(params)
            return {"intent": Intent.ADD_TO_CART, "task_params": data}

        if type_name == "SendRequestParams":
            print(f"Intent classified as: {type_name}")
            data = SendRequestParams.model_validate(params)
            return {"intent": Intent.SEND_REQUEST, "task_params": data}

        if type_name == "GetMenuInfoParams":
            print(f"Intent classified as: {type_name}")
            data = GetMenuInfoParams.model_validate(params)
            return {"intent": Intent.GET_MENU_INFO, "task_params": data}
            
        # 예외 처리
        return {"intent": Intent.CHITCHAT}
    
    def route_confirmation(self, state: ChatState):
        """
        state를 기반으로 이동할 노드를 확인합니다.
        """
        print(">> Node: route_confirmation")

        awaiting = state.get("awaiting")
        intent: Optional[Intent] = state.get("intent")

        if awaiting == "confirmation_add_to_cart":
            if intent == Intent.CONFIRM:
                return Command(goto="call_add_to_cart_api")
            elif intent == Intent.DENY:
                return Command(
                    goto=END,
                    update={
                        "response": "네, 추가적으로 필요하신 사항이 있다면 말씀해주세요.",
                        "task_params": None, # 작업 완료 후 파라미터 초기화
                        "awaiting": None
                    }
                )
            return Command(
                goto=END,
                update={
                    "response": "네/아니오로 알려주세요. 장바구니에 추가할까요?",
                }
            )
        if awaiting == "confirmation_request":
            if intent == Intent.CONFIRM:
                return Command(goto="call_request_api")
            elif intent == Intent.DENY:
                return Command(
                    goto=END,
                    update={
                        "response": "네, 추가적으로 필요하신 사항이 있다면 말씀해주세요.",
                        "task_params": None, # 작업 완료 후 파라미터 초기화
                        "awaiting": None
                    }
                )
            return Command(
                goto=END,
                update={
                    "response": "네/아니오로 알려주세요. 요청사항을 전송할까요?",
                }
            )
        
        if intent == Intent.ADD_TO_CART:
            return Command(goto="ask_add_to_cart_confirmation")
        
        if intent == Intent.SEND_REQUEST:
            return Command(goto="ask_request_confirmation")
        
        return Command(
            goto=END,
            update={"response": "지금은 장바구니 기능만 테스트 중이에요. 담을 메뉴를 말씀해 주세요!"}
        )


    def ask_add_to_cart_confirmation(self, state: ChatState) -> Dict[str, Any]:
        """
        장바구니 추가 전, 사용자에게 확인 질문을 생성하고
        'awaiting' 상태를 업데이트합니다.
        """
        print(">> Node: ask_add_to_cart_confirmation")

        task_params = state.get("task_params")
        print(task_params)
        if not isinstance(task_params, AddToCartParams):
            # 잘못된 파라미터가 들어왔다면, 에러 응답을 생성하고 종료
            error_message = "죄송합니다, 메뉴를 장바구니에 담는 중 오류가 발생했습니다. 다시 시도해 주세요."
            return {"response": error_message}
        
        menu_name = task_params.menu_name
        quantity = task_params.quantity or 1
        response_message = f"'{menu_name}' {quantity}개를 장바구니에 추가할까요?"
        awaiting_status = "confirmation_add_to_cart"

        print(f"Asking confirmation: {response_message}")
        print(f"State awaiting set to: {awaiting_status}")

        return Command(
            goto=END,
            update={"response": response_message, "messages": [AIMessage(content=response_message)], "awaiting": awaiting_status}
        )

    def call_add_to_cart_api(self, state: ChatState) -> Dict[str, Any]:
        """
        사용자의 확인 후, 실제 백엔드 API를 호출하여 장바구니에 메뉴를 추가합니다.
        """
        print(">> Node: call_add_to_cart_api")

        task_params = state.get("task_params")
        if not isinstance(task_params, AddToCartParams):
            return {"response": "오류가 발생했습니다."}

        menu_name = task_params.menu_name
        quantity = task_params.quantity or 1

        try:
            # --- 실제 API 호출 부분 ---
            payload = {"menu_name": menu_name, "quantity": quantity}
            # response = requests.post(BACKEND_API_URL, json=payload)
            # response.raise_for_status()  # 200번대 응답이 아니면 에러 발생
            
            # api_result = response.json()
            # print(f"API call successful: {api_result}")
            
            # 작업이 성공적으로 끝났으므로, 관련 상태를 초기화하고 결과를 저장
            response_message = f"'{menu_name}' {quantity}개가 장바구니에 성공적으로 추가되었습니다."
            return Command(
                goto=END,
                update={
                    "api_result": "success", # 수정 필요
                    "response": response_message,
                    "task_params": None, # 작업 완료 후 파라미터 초기화
                    "awaiting": None,    # 대기 상태 해제
                    "messages": [AIMessage(content=response_message)]
                }
            )
        
        # except requests.exceptions.RequestException as e:
        except Exception as e:
            print(f"API call failed: {e}")
            # API 호출 실패 시 사용자에게 보여줄 메시지
            return Command(
                goto=END,
                update={
                    "api_result": {"error": str(e)},
                    "response": "죄송합니다, 서버 통신 중 오류가 발생하여 장바구니에 추가하지 못했습니다.",
                    "task_params": None,
                    "awaiting": None,
                    "messages": [AIMessage(content="죄송합니다, 서버 통신 중 오류가 발생하여 장바구니에 추가하지 못했습니다.")]
                }
            )
        
    def ask_request_confirmation(self, state: ChatState) -> Dict[str, Any]:
        """
        장바구니 추가 전, 사용자에게 확인 질문을 생성하고
        'awaiting' 상태를 업데이트합니다.
        """
        print(">> Node: ask_request_confirmation")

        task_params = state.get("task_params")
        print(task_params)
        if not isinstance(task_params, SendRequestParams):
            # 잘못된 파라미터가 들어왔다면, 에러 응답을 생성하고 종료
            error_message = "죄송합니다, 요청사항을 확인하는 중 오류가 발생했습니다. 다시 시도해 주세요."
            return {"response": error_message}
        
        request_note = task_params.request_note
        response_message = f"가게에 '{request_note}' 라고 요청사항을 전송할까요?"
        awaiting_status = "confirmation_request"

        print(f"Asking confirmation: {response_message}")
        print(f"State awaiting set to: {awaiting_status}")

        return Command(
            goto=END,
            update={"response": response_message, "messages": [AIMessage(content=response_message)], "awaiting": awaiting_status}
        )

    def call_request_api(self, state: ChatState) -> Dict[str, Any]:
        """
        사용자의 확인 후, 실제 백엔드 API를 호출하여 요청사항을 전송합니다.
        """
        print(">> Node: call_request_api")

        task_params = state.get("task_params")
        if not isinstance(task_params, SendRequestParams):
            return {"response": "오류가 발생했습니다."}

        request_note = task_params.request_note

        try:
            # --- 실제 API 호출 부분 ---
            payload = {"menu_name": request_note }
            # response = requests.post(BACKEND_API_URL, json=payload)
            # response.raise_for_status()  # 200번대 응답이 아니면 에러 발생
            
            # api_result = response.json()
            # print(f"API call successful: {api_result}")
            
            # 작업이 성공적으로 끝났으므로, 관련 상태를 초기화하고 결과를 저장
            response_message = f"요청사항을 전송하였습니다! 잠시만 기다려주세요."
            return Command(
                goto=END,
                update={
                    "api_result": "success", # 수정 필요
                    "response": response_message,
                    "task_params": None, # 작업 완료 후 파라미터 초기화
                    "awaiting": None,    # 대기 상태 해제
                    "messages": [AIMessage(content=response_message)]
                }
            )
        
        # except requests.exceptions.RequestException as e:
        except Exception as e:
            print(f"API call failed: {e}")
            # API 호출 실패 시 사용자에게 보여줄 메시지
            return Command(
                goto=END,
                update={
                    "api_result": {"error": str(e)},
                    "response": "죄송합니다, 서버 통신 중 오류가 발생하여 요청사항을 전송하지 못했습니다.",
                    "task_params": None,
                    "awaiting": None,
                    "messages": [AIMessage(content="죄송합니다, 서버 통신 중 오류가 발생하여 요청사항을 전송하지 못했습니다.")]
                }
            )
        
    # def get_store_info_node(self, state: ChatState) -> Dict[str,Any]:
    #     query = state["input"]
    #     print("사용자 질문: ", query)
    #     store_info = self.get_store_info.invoke({"store_id": state["store_id"]})

    #     prompt = f"""당신은 고객에게 가게 정보를 안내하는 친절한 음식점 점원입니다.
    #         아래 '가게 정보'에서 사용자의 질문에 해당하는 정보를 찾아 답변을 생성하세요.
    #         사용자의 질문과 관련없는 정보는 답변에 포함하지 마세요.
    #         사용자의 질문과 관련된 정보를 찾지 못한 경우 '죄송합니다. 해당 정보를 찾지 못했습니다.'라고 답변하세요.

    #         [가게 정보]
    #         {store_info}

    #         [사용자의 질문]
    #         {query}

    #         [답변]
    #         """

    #     response = self.llm.invoke(prompt)
    #     return {"result": response.content}

    # def get_menu_detail_node(self, state: ChatState) -> Dict[str,Any]:
    #     query = state["input"]
    #     print("사용자 질문: ", query)

    #     menu_name = state["params"].get("menu_name")
    #     if not menu_name and isinstance(state.get("docs"), list):
    #             # 직전 결과에서 menu_name 추출
    #             for d in state["docs"]:
    #                 if d.metadata.get("menu_name"):
    #                     menu_name = d.metadata["menu_name"]
    #                     break

    #     if not menu_name:
    #         return {"result": "어떤 메뉴를 말씀하시는지 다시 한 번 알려주세요."}
    #     menu_detail = self.get_menu_detail.invoke({"store_id": state["store_id"], "menu_name":menu_name})

    #     prompt = f"""당신은 고객에게 메뉴 정보를 안내하는 친절한 음식점 점원입니다.
    #         아래 '메뉴 정보'에서 사용자의 질문에 해당하는 정보를 찾아 답변을 생성하세요.
    #         사용자의 질문과 관련없는 정보는 답변에 포함하지 마세요.
    #         사용자의 질문과 관련된 정보를 찾지 못한 경우 '죄송합니다. 해당 정보를 찾지 못했습니다.'라고 답변하세요.
    #         '알레르기 유발 재료'는 특정 재료에 대해 묻는 질문인 경우에만 활용하세요.

    #         [메뉴 정보]
    #         {menu_detail}

    #         [사용자의 질문]
    #         {query}

    #         [답변]
    #         """

    #     response = self.llm.invoke(prompt)
    #     return {"result": response.content}
    
    # --- Public 메서드 ---

    def process_chat(self, session_id: str, user_input: str, store_id: int) -> str:
        """
        사용자 입력을 받아 전체 챗봇 플로우를 실행하고 최종 답변을 반환합니다.
        """
        initial_state: ChatState = {
            "store_id": store_id,
            "messages": [HumanMessage(content=user_input)]
        }

        config = {"configurable": {"thread_id":session_id}}
        output_state: ChatState = self.graph.invoke(initial_state, config=config)
        return output_state.get("response", "오류가 발생했습니다.")

chatbot_service = ChatbotService(vs=vectorstore_service)