import os
import ast
import logging
import requests
from typing import Optional, List, Dict, Any, Union

# LangChain 및 LangGraph 관련 라이브러리
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.redis import RedisSaver
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
    MenuQuerySchema,
    MenuItem
)

# 외부 서비스 및 설정
from app.services.vectorstore_service import VectorStoreService, vectorstore_service
from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        workflow.add_node("get_store_info", self.get_store_info) # 가게 정보 제공
        workflow.add_node("get_menu_info", self.get_menu_info) # 메뉴 정보 제공
        workflow.add_node("chitchat", self.chitchat) # 기본 대화

        workflow.add_node("extract_menu_params", self.extract_menu_params) # 추천 조건 추출
        workflow.add_node("get_popular_menus", self.get_popular_menus) # 인기 메뉴 목록 조회
        workflow.add_node("execute_search", self.execute_search) # 문서 검색
        workflow.add_node("rerank_documents", self.rerank_documents) # 유사도 + 인기도 기반 스코어링
        workflow.add_node("generate_single_recommendation", self.generate_single_recommendation) # 한 가지 메뉴 응답 생성
        workflow.add_node("generate_multiple_recommendation", self.generate_multiple_recommendation) # 여러 가지 메뉴 응답 생성

        workflow.set_entry_point("classify_intent") 
        workflow.add_edge("classify_intent", "route_confirmation")

        REDIS_URL = os.getenv("REDIS_URL")

        ttl_config = {
            "default_ttl": 60,  # Expire checkpoints after 60 minutes
            "refresh_on_read": True,  # Reset expiration time when reading checkpoints
        }

        with RedisSaver.from_conn_string(REDIS_URL, ttl=ttl_config) as checkpointer:
            checkpointer.setup()

        self.graph = workflow.compile(checkpointer=checkpointer)

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
                사용자의 최근 메시지와 이전 대화 내용을 바탕으로, 사용자의 의도를 아래 8종 중 하나로 분류하고 가장 적절한 도구(Tool)로 필요한 정보를 추출하세요.
                도구는 'type' 필드로 구분됩니다.
                type은 route|add_to_cart|send_request|get_menu_info 중 하나입니다.
                추가 파라미터가 필요 없는 경우는 route를 사용하여 intent만 반환하세요.
                - add_to_cart: 장바구니에 메뉴 추가
                - send_request: 가게에 대한 요청사항 전달
                - recommend_menu: 사용자의 조건에 따른 메뉴 검색 및 정보 제공. 추천 요청은 전부 이 의도에 해당하며 문장에 명시적으로 나와있지 않더라도 조건이 있는 경우 이 의도에 포함 (예시 : ~는 뭐가 있어? ~한 메뉴 있어?)
                - get_store_info: 가게에 대한 정보 제공 (가게 이름, 설명, 영업시간, 브레이크 타임 등)
                - get_menu_info: 특정 한 개 메뉴에 대한 설명 제공 (특정 메뉴의 가격/맵기/알레르기 유발 재료 등)
                - chitchat: 기능과 무관한 일반 대화
                - confirm: 사용자의 긍정 응답 (예: "네", "맞아요", "어", "응")
                - deny: 사용자의 부정 응답 (예: "아니요", "취소")
                필요한 모든 파라미터는 반드시 JSON으로 응답해야 합니다.
                """
             ),
            ("human", "이전 대화:\n{history}\n\n최신 사용자 메시지: {userInput}")
        ])

        messages = state['messages']
        history = "\n".join([f"{'User' if msg.type == 'human' else 'Bot'}: {msg.content}" for msg in messages[-9:-1]])
        userInput = messages[-1].content

        result = (prompt | model).invoke({"history": history, "userInput": userInput})
        type_name = result.tool_calls[0]["name"]
        params = result.tool_calls[0]["args"]

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
                        "messages": [AIMessage(content="네, 추가적으로 필요하신 사항이 있다면 말씀해주세요.")],
                        "task_params": None, # 작업 완료 후 파라미터 초기화
                        "awaiting": None
                    }
                )
            return Command(
                goto=END,
                update={
                    "response": "네/아니오로 알려주세요. 요청사항을 전송할까요?",
                    "messages": [AIMessage(content="네/아니오로 알려주세요. 요청사항을 전송할까요?")]
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
                        "messages": [AIMessage(content="네, 추가적으로 필요하신 사항이 있다면 말씀해주세요.")],
                        "task_params": None, # 작업 완료 후 파라미터 초기화
                        "awaiting": None
                    }
                )
            
            return Command(
                goto=END,
                update={
                    "response": "네/아니오로 알려주세요. 요청사항을 전송할까요?",
                    "messages": [AIMessage(content="네/아니오로 알려주세요. 요청사항을 전송할까요?")]
                }
                
            )
        
        if intent == Intent.ADD_TO_CART:
            return Command(goto="ask_add_to_cart_confirmation")
        
        if intent == Intent.SEND_REQUEST:
            return Command(goto="ask_request_confirmation")
        
        if intent == Intent.GET_STORE_INFO:
            return Command(goto="get_store_info")
        
        if intent == Intent.GET_MENU_INFO:
            return Command(goto="get_menu_info")
        
        if intent == Intent.CHITCHAT:
            return Command(goto="chitchat")
        
        if intent == Intent.RECOMMEND_MENU:
            return Command(goto="extract_menu_params")
        
        response = "죄송합니다. 질문 주신 내용을 파악하지 못했습니다. 다시 한번 말씀해주시겠어요?"
        return Command(
            goto=END,
            update={"response": response, "messages": [AIMessage(content=response)]}
        )


    def ask_add_to_cart_confirmation(self, state: ChatState) -> Dict[str, Any]:
        """
        장바구니 추가 전, 사용자에게 확인 질문을 생성하고
        'awaiting' 상태를 업데이트합니다.
        """
        print(">> Node: ask_add_to_cart_confirmation")

        task_params = state.get("task_params")
        store_id = state["store_id"]

        print(task_params)
        if not isinstance(task_params, AddToCartParams):
            # 잘못된 파라미터가 들어왔다면, 에러 응답을 생성하고 종료
            error_message = "죄송합니다, 메뉴를 장바구니에 담는 중 오류가 발생했습니다. 다시 시도해 주세요."
            return {"response": error_message}
        
        validated_items: List[MenuItem] = []
        for item in task_params.items:
            # DB에서 가장 유사한 메뉴 1개를 검색
            search_results: List[Document] = self.vectorstore_service.find_document(
                store_id=store_id,
                query=item.menu_name,
                type="menu",
                k=1
            )
        
            # 검색 결과가 있을 경우에만, 검증된 아이템으로 간주하고 추가
            if search_results:
                found_doc = search_results[0]
                correct_menu_name = found_doc.metadata.get("menu_name", item.menu_name)
                validated_items.append(
                    MenuItem(menu_name=correct_menu_name, quantity=item.quantity)
                )
        if not validated_items:
            msg = "담을 수 있는 메뉴를 찾지 못했어요. 다른 메뉴로 다시 말씀해 주세요."
            return Command(goto=END, update={"response": msg, "messages": [AIMessage(content=msg)]})        

        item_strings = [f"'{item.menu_name}' {item.quantity}개" for item in validated_items]
        items_text = ", ".join(item_strings)
        response_message = f"{items_text}를 장바구니에 추가할까요?"
        
        if len(item_strings) == 1:
            response_message = f"{item_strings[0]}를 장바구니에 추가할까요?"
        else:
            items_text = ", ".join(item_strings)
            response_message = f"{items_text}를 장바구니에 추가할까요?"


        awaiting_status = "confirmation_add_to_cart"

        print(f"Asking confirmation: {response_message}")
        print(f"State awaiting set to: {awaiting_status}")

        updated_task_params = AddToCartParams(items=validated_items, type="add_to_cart")

        return Command(
            goto=END,
            update={"response": response_message, "messages": [AIMessage(content=response_message)], "awaiting": awaiting_status, "task_params":updated_task_params}
        )

    def call_add_to_cart_api(self, state: ChatState) -> Dict[str, Any]:
        """
        사용자의 확인 후, 실제 백엔드 API를 호출하여 장바구니에 메뉴를 추가합니다.
        """
        print(">> Node: call_add_to_cart_api")

        task_params = state.get("task_params")
        if not isinstance(task_params, AddToCartParams):
            return {"response": "죄송합니다. 오류가 발생했습니다."}

        try:
            # --- 실제 API 호출 부분 ---
            items_payload = [item.model_dump() for item in task_params.items]
            payload = {"items": items_payload}
            # response = requests.post(BACKEND_API_URL, json=payload)
            # response.raise_for_status()  # 200번대 응답이 아니면 에러 발생
            
            # api_result = response.json()
            # print(f"API call successful: {api_result}")
            
            # 작업이 성공적으로 끝났으므로, 관련 상태를 초기화하고 결과를 저장
            item_strings = [f"'{item.menu_name}' {item.quantity}개" for item in task_params.items]
            items_text = ", ".join(item_strings)
            response_message = f"{items_text}가 장바구니에 성공적으로 추가되었습니다."
            
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
            return {"response": "죄송합니다. 오류가 발생했습니다."}

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
        
    def get_store_info(self, state: ChatState) -> Dict[str, Any]:
        """
        사용자 질문을 기반으로 ChromaDB에서 가게 정보를 RAG로 검색하고,
        LLM을 통해 최종 답변을 생성합니다.
        """
        print(">> Node: get_store_info")

        store_id = state["store_id"]
        question = state['messages'][-1].content
        history = "\n".join([f"{'User' if msg.type == 'human' else 'Bot'}: {msg.content}" for msg in state['messages'][-9:-1]])
        print(question)

        docs = self.vectorstore_service.find_document(
                query="가게 정보",
                store_id=store_id,
                type="store_info"
            )
        
        if(docs):
            content = docs[0].page_content
            llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

            prompt = ChatPromptTemplate.from_messages([("system",
                """당신은 레스토랑 챗봇입니다. 아래에 제공된 [가게 정보]와 [대화 기록]만을 사용하여 사용자의 '[질문]'에 대해 간결하고 친절하게 답변하세요.
                정보가 없다면, 정보를 찾을 수 없다고 솔직하게 답변해야 합니다. 절대 정보를 지어내지 마세요.
                
                [가게 정보]
                {context}

                [대화 기록]
                {history}
                """),
                ("human", "[질문]\n{question}")])
            
            chain = prompt | llm | StrOutputParser()

            response = chain.invoke({
                "context": content,
                "question": question,
                "history":history
            })

            return Command(
                goto=END,
                update={"response": response, "messages": [AIMessage(content=response)]}
            )

        else:
            response = "죄송합니다. 문의하신 정보를 찾지 못했습니다."
            return Command(
                goto=END,
                update={"response": response, "messages": [AIMessage(content=response)]}
            )

    def get_menu_info(self, state: ChatState) -> Dict[str, Any]:
        """
        사용자 질문을 기반으로 ChromaDB에서 메뉴 정보를 RAG로 검색하고,
        LLM을 통해 최종 답변을 생성합니다.
        """
        print(">> Node: get_menu_info")

        store_id = state["store_id"]
        question = state['messages'][-1].content
        history = "\n".join([f"{'User' if msg.type == 'human' else 'Bot'}: {msg.content}" for msg in state['messages'][-9:-1]])

        task_params = state.get("task_params")

        docs = self.vectorstore_service.find_document(
                query=task_params.menu_name,
                store_id=store_id,
                type="menu",
                k=5
            )
        
        if(docs):
            content = docs[0].page_content
            llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

            prompt = ChatPromptTemplate.from_messages([("system",
                """당신은 레스토랑 챗봇입니다. 오직 아래에 제공된 [메뉴 정보]와 [대화 기록]을 사용하여 사용자의 '[질문]'에 대해 간결하고 친절하게 답변하세요.
                알레르기 유발 재료 정보는 알레르기에 대한 질문인 경우에만 포함하세요.
                정보가 없다면, 정보를 찾을 수 없다고 솔직하게 답변해야 합니다. 절대 정보를 지어내지 마세요.
                
                [가게 정보]
                {context}

                [대화 기록]
                {history}
                """),
                ("human", "[질문]\n{question}")])
            
            chain = prompt | llm | StrOutputParser()

            response = chain.invoke({
                "context": content,
                "question": question,
                "history":history
            })

            return Command(
                goto=END,
                update={"response": response, "messages": [AIMessage(content=response)]}
            )

        else:
            response = "죄송합니다. 문의하신 정보를 찾지 못했습니다."
            return Command(
                goto=END,
                update={"response": response, "messages": [AIMessage(content=response)]}
            )
    
    def chitchat(self, state: ChatState) -> Dict[str, Any]:
        """
        명시된 의도 외의 메시지에 대한 자연스러운 답변을 제공합니다.
        """
        print(">> Node: chitchat")

        question = state['messages'][-1].content
        history = "\n".join([f"{'User' if msg.type == 'human' else 'Bot'}: {msg.content}" for msg in state['messages'][-5:-1]])

        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.5)

        prompt = ChatPromptTemplate.from_messages([("system",
            """당신은 레스토랑 챗봇입니다. [대화 내역]을 반영하여 사용자의 '[메시지]'에 대해 간결하고 친절하게 답변하세요.
            
            [대화 내역]
            {history}
            """),
            ("human", "[메시지]\n{question}")])
        
        chain = prompt | llm | StrOutputParser()

        response = chain.invoke({
            "question": question,
            "history": history
        })

        return Command(
            goto=END,
            update={"response": response, "messages": [AIMessage(content=response)], "task_params": None}
        )
    
    def extract_menu_params(self, state:ChatState) -> Dict[str, Any] :
        """
        사용자 질문을 분석하여 메뉴 추천에 필요한 모든 조건들을
        'MenuQuerySchema' 객체로 추출합니다.
        """
        print(">> Node: extract_menu_params")

        llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
        structured_llm = llm.with_structured_output(MenuQuerySchema, method="function_calling")

        prompt = ChatPromptTemplate.from_messages([
            ("system", 
            """당신은 레스토랑 챗봇의 메뉴 추천 조건 분석 전문가입니다.
            사용자의 최근 메시지와 이전 대화 내용을 바탕으로, 'MenuQuerySchema'의 모든 필드를 최대한 정확하게 채워주세요.

            필드별 가이드라인:
            - query_description: '따뜻한 국물 요리', '매콤한 메뉴'처럼 맛이나 형태를 묘사하는 부분을 추출하세요. 없으면 null로 두세요.
            - price_max / price_min: '만원 이하', '2만원에서 3만원 사이' 같은 가격 정보를 숫자로 변환하여 추출하세요.
            - allergies_include: '땅콩 있는', '갑각류 들어간' 같은 특정 재료 포함 요구사항을 추출하세요.
            - allergies_exclude: '땅콩 빼고', '갑각류 없는' 같은 특정 재료 제외 요구사항을 추출하세요.
            - menu_exclude: 추천에서 제외할 메뉴 이름을 이전 대화의 맥락을 반영하여 추출하세요. 없으면 기본값인 None으로 두세요.
            - is_popular: '인기 있는', '잘나가는', '대표 메뉴' 같은 단어가 있으면 True로 설정하세요.
            - single_result: 사용자의 질문에 단일 메뉴로 대답할 수 있다면 True로 설정하세요. '어떤 메뉴들이 있어?', '메뉴 종류' 등 복수로 응답해야 하는 경우 False로 두세요.
            - num_food: '3명이서 먹을', '두 개' 같은 음식 개수를 숫자로 추출하세요. 1명은 1개의 음식을 먹습니다. 없으면 기본값인 None으로 두세요.
            """),
            ("human", "이전 대화:\n{history}\n\n최신 사용자 메시지: {userInput}")
        ])

        messages = state['messages']
        history = "\n".join([f"{'User' if msg.type == 'human' else 'Bot'}: {msg.content}" for msg in messages[:-1]])
        userInput = messages[-1].content

        chain = prompt | structured_llm
        extracted_params = chain.invoke({"history": history, "userInput": userInput})

        print(f"Extracted menu query params: {extracted_params}")

        data = extracted_params.dict()
        if all(data[field] is None for field in data if field not in ["is_popular", "single_result", "num_food"]):
            return Command(
                goto="get_popular_menus",
                update={"menu_query_params": extracted_params}
            )
        else :     
            return Command(
                goto="execute_search",
                update={"menu_query_params": extracted_params}
            )
    
    def get_popular_menus(self, state:ChatState) -> Dict[str, Any]:
        """
        백엔드에서 인기 메뉴 목록을 받아옵니다.
        """
        print(">> Node: get_popular_menus")

        store_id = state["store_id"]

        try:
            # --- 실제 API 호출 부분 ---
            payload = {"store_id": store_id }
            # response = requests.post(BACKEND_API_URL, json=payload)
            # response.raise_for_status()  # 200번대 응답이 아니면 에러 발생
            # api_result = response.json().get("data",[])
            # print(f"API call successful: {api_result}")

            api_result = [
                {
                "menuId": 101,
                "menuName": "탄탄지 샐러드",
                "menuInfo": "국내산 닭가슴살과 고구마무스가 들어간 샐러드",
                "menuPrice": 8600,
                },
                {
                "menuId": 102,
                "menuName": "하가우",
                "menuInfo": "통새우가 가득 들어간 딤섬",
                "menuPrice": 8900,
                },
                {
                "menuId": 103,
                "menuName": "바나나 아이스크림",
                "menuInfo": "달콤한 바나나와 부드러운 바닐라 아이스크림",
                "menuPrice": 4800,
                }
            ]
            
            return Command(
                goto="generate_multiple_recommendation",
                update={
                    "api_result": "success", # 수정 필요
                    "search_results": api_result
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
                    "response": "죄송합니다, 서버 통신 중 오류가 발생하였습니다. 다시 시도해주세요.",
                    "task_params": None,
                    "awaiting": None,
                    "messages": [AIMessage(content="죄송합니다, 서버 통신 중 오류가 발생하였습니다. 다시 시도해주세요.")]
                }
            )


    def execute_search(self, state:ChatState) -> Dict[str, Any] :
        """
        추출된 menu_query_params를 사용하여 ChromaDB에서
        의미 기반 검색과 메타데이터 필터링을 결합한 검색을 수행하여 문서 후보군을 만듭니다.
        """
        print(">> Node: execute_search")

        store_id = state["store_id"]
        params = state.get("menu_query_params")
        if not params:
            print("Error: menu_query_params not found in state.")
            return Command(
                goto=END,
                update={"response": "죄송합니다. 오류가 발생했습니다."}
            )

        filters = []
        if params.price_min is not None:
            filters.append({"price": {"$gte": params.price_min}})
        if params.price_max is not None:
            filters.append({"price": {"$lte": params.price_max}})
        if params.allergies_exclude:
            filters.append({"allergens": {"$nin": params.allergies_exclude}})
        if params.allergies_include:
            filters.append({"allergens": {"$in": params.allergies_include}})
        if params.menu_exclude:
            filters.append({"menu_name": {"$nin": params.menu_exclude}})

        query_text = params.query_description
        if not query_text:
            query_text = state['messages'][-1].content
        
        print(f"Query text for vector search: '{query_text}'")
        
        try:
            results = self.vectorstore_service.find_conditional_document(
                store_id=store_id,
                type="menu",
                query=query_text,
                filters=filters,
                k=10
            )

        except Exception as e:
            print(f"Error during ChromaDB query: {e}")
            return Command(
                goto=END,
                update={"response": "죄송합니다. 오류가 발생했습니다."}
            )
          
        print(f"Found {len(results)} candidate documents.")

        return Command(
            goto="rerank_documents",
            update={"search_results": results}
        )
    
    def rerank_documents(self, state:ChatState) -> Dict[str, Any] :
        """
        검색된 후보군에 인기도를 결합하여 최종 점수를 계산하고,
        사용자 요구에 맞게 결과 개수를 조절합니다.
        """
        print(">> Node: rerank_documents")

        SIMILARITY_WEIGHT = 0.8  # 검색어와의 관련도 가중치
        POPULARITY_WEIGHT = 0.2  # 대중적인 인기도 가중치

        candidate_results = state.get("search_results")
        params = state.get("menu_query_params")
        if not candidate_results:
            return Command(
                goto=END,
                update={"response": "죄송합니다. 다른 조건으로 다시 질문해주시겠어요?"}
            )

        menu_ids = [doc.metadata.get('menu_id') for doc, score in candidate_results if doc.metadata.get('menu_id')]
        # try:
        #     response = requests.post(BACKEND_API_URL, json={"menu_ids": menu_ids})
        #     response.raise_for_status()
        #     popularity_data = response.json().get("data",[])
        #     popularity_map = {item['menu_id']: item['score'] for item in popularity_data}
        # except requests.exceptions.RequestException as e:
        #     print(f"API call for popularity failed: {e}. Proceeding without popularity scores.")
        
        popularity_map = { # 임시 데이터
            1:20,
            2:30,
            3:50,
            4:10,
            5:70,
            6:40,
            7:90,
            8:110,
            9:120,
            10:20,
            11:30,
            12:40,
            13:70,
            14:3,
        }

        scored_results = []

        # 인기도 점수 정규화를 위한 최대/최소값 찾기
        pop_scores = [score for score in popularity_map.values() if score is not None]
        min_pop, max_pop = (min(pop_scores), max(pop_scores)) if pop_scores else (0, 0)
        
        for doc, similarity_score in candidate_results:
            menu_id = doc.metadata.get('menu_id')
            pop_score = popularity_map.get(menu_id, 0)

            # 인기도 점수 정규화 (0~1 사이 값으로 변환)
            if max_pop > min_pop:
                norm_pop_score = (pop_score - min_pop) / (max_pop - min_pop)
            else:
                norm_pop_score = 0.0

            # 최종 점수 = (유사도 * 가중치) + (정규화된 인기도 * 가중치)
            final_score = (similarity_score * SIMILARITY_WEIGHT) + (norm_pop_score * POPULARITY_WEIGHT)
            
            scored_results.append({"doc": doc, "final_score": final_score})

        sorted_results = sorted(scored_results, key=lambda x: x['final_score'], reverse=True)

        final_count = 5 # 기본값은 5개
        if params.num_food is not None and params.num_food > 0:
            final_count = params.num_food
            print(f"Slicing results to {final_count} based on 'num_food'.")
        elif params.single_result:
            final_count = 1
            print(f"Slicing results to 1 based on 'single_result'.")
        else:
            print(f"Slicing results to default count of {final_count}.")

        final_documents = sorted_results[:final_count]

        final_results = [item['doc'] for item in final_documents]
    
        print(f"Reranking complete. Final number of documents: {len(final_results)}")
        print(final_results)

        if params.single_result :
            return Command(
                goto="generate_single_recommendation",
                update={"search_results": final_results}
            )
        else :
            return Command(
                goto="generate_multiple_recommendation",
                update={"search_results": final_results}
            )


    def generate_single_recommendation(self, state:ChatState) -> Dict[str, Any] :
        """
        사용자 질문을 기반으로 한 개의 메뉴에 대해 LLM을 통해 최종 답변을 생성합니다.
        """
        print(">> Node: generate_single_recommendation")

        question = state['messages'][-1].content
        document = state["search_results"][0]

        print(question)

        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

        prompt = ChatPromptTemplate.from_messages([("system",
            """당신은 레스토랑 챗봇입니다. 오직 아래에 제공된 '[메뉴 정보]'를 사용하여 사용자의 '[질문]'에 대해 간결하고 친절하게 메뉴를 추천하세요.
            메뉴의 설명에 만약 사용자의 질문 조건과 어긋나는 내용이 있다면 해당 내용을 명시하세요.
            알레르기 유발 재료 정보는 알레르기에 대한 질문인 경우에만 포함하세요.
            사용자의 조건에 특정 재료 포함 여부가 있는 경우, "해당 답변은 참고용이며, 정확한 알레르기 관련 정보는 꼭 가게에 확인 부탁드립니다." 라는 문장을 포함하십시오.
            절대 없는 정보를 지어내서는 안됩니다.
            
            [메뉴 정보]
            {context}
            """),
            ("human", "[질문]\n{question}")])
        
        chain = prompt | llm | StrOutputParser()

        response = chain.invoke({
            "context": document,
            "question": question
        })

        return Command(
            goto=END,
            update={"response": response, "messages": [AIMessage(content=response)], "search_results" : None}
        )

    def generate_multiple_recommendation(self, state:ChatState) -> Dict[str, Any] :
        """
        사용자 질문을 기반으로 여러 개의 메뉴에 대해 LLM을 통해 최종 답변을 생성합니다.
        """
        print(">> Node: generate_multiple_recommendation")

        print(state)
        question = state['messages'][-1].content
        document = state["search_results"]
        print(question)


        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

        prompt = ChatPromptTemplate.from_messages([("system",
            """당신은 레스토랑 챗봇입니다. 아래에 제공된 '[메뉴 정보]'를 사용하여 사용자의 '[질문]'에 대해 간결하고 친절하게 답변하세요.
            1. 서론: 먼저, 어떤 요청에 기반한 추천인지 간단히 언급하며 대화를 시작하세요. (예: "고객님께서 요청하신 10,000원 이하의 인기 메뉴들로 몇 가지 준비해봤어요.")
            2. 본론 (메뉴 목록):
            - 각 메뉴를 번호나 글머리 기호 목록으로 제시하세요.
            - 각 항목에는 메뉴 이름(굵은 글씨), 가격, 그리고 메뉴의 특징을 한 문장으로 요약한 설명을 포함해야 합니다.
            3. 결론: 목록 제시 후, 다음 행동을 유도하는 질문으로 마무리하세요. (예: "이 중에서 마음에 드는 메뉴가 있으신가요? 원하시는 메뉴를 말씀해주시면 장바구니에 담아드릴게요.")
            알레르기 유발 재료 정보는 알레르기에 대한 질문인 경우에만 포함하세요.
            사용자의 조건에 특정 재료 포함 여부가 있는 경우, "해당 답변은 참고용이며, 정확한 알레르기 관련 정보는 꼭 가게에 확인 부탁드립니다." 라는 문장을 포함하십시오.
            절대 없는 정보를 지어내서는 안됩니다.
            
            [메뉴 정보]
            {context}
            """),
            ("human", "[질문]\n{question}")])
        
        chain = prompt | llm | StrOutputParser()

        response = chain.invoke({
            "context": document,
            "question": question
        })

        return Command(
            goto=END,
            update={"response": response, "messages": [AIMessage(content=response)], "search_results" : None}
        )

    # --- Public 메서드 ---
    def process_chat(self, session_id: str, user_input: str, store_id: int, table_id: int) -> str:
        """
        사용자 입력을 받아 전체 챗봇 플로우를 실행하고 최종 답변을 반환합니다.
        """
        initial_state: ChatState = {
            "store_id": store_id,
            "customer_key": session_id,
            "table_id": table_id,
            "messages": [HumanMessage(content=user_input)]
        }

        config = {"configurable": {"thread_id":session_id}}
        output_state: ChatState = self.graph.invoke(initial_state, config=config)
        return output_state.get("response", "오류가 발생했습니다.")

chatbot_service = ChatbotService(vs=vectorstore_service)