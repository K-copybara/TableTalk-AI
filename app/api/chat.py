from fastapi import APIRouter, HTTPException
from app.schemas.chatSchema import ChatRequest, ChatResponse
from app.services.chatbot_service import chatbot_service

router = APIRouter()

@router.post("/response", response_model=ChatResponse)
async def handle_chat(request: ChatRequest):
    try:
        answer = chatbot_service.process_chat(
            session_id=request.session_id,
            table_id=request.table_id,
            customer_key=request.customer_key,
            user_input=request.user_input,
            store_id=request.store_id,
        )

        return ChatResponse(answer=answer)
    
    except Exception as e:
        print(f"채팅 처리 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"챗봇 응답 생성 중 오류가 발생했습니다. : {e}")
