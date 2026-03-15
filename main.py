import streamlit as st
from typing import TypedDict, List, Literal
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

class ApartmentState(TypedDict):
    messages: List[str]
    category: str  
llm = ChatOllama(model="qwen3:4b", temperature=0)

def classifier_node(state: ApartmentState):
    """
    ทดสอบ การตัดสินใจว่า จะแจ้งซ้่อม หรือ ตอบคำถามทั่วไป
    """
    user_input = state["messages"][-1]
    prompt = f"Categorize this message as 'repair' or 'general': {user_input}"
    response = llm.invoke(prompt)
    category = "repair" if "repair" in response.content.lower() else "general"
    return {"category": category}

def repair_handler(state: ApartmentState):
    """จัดการเรื่องแจ้งซ่อม"""
    return {"messages": state["messages"] + ["AI: รับเรื่องแจ้งซ่อมแล้วครับ ทีมช่างจะติดต่อกลับโดยเร็ว"]}

def general_handler(state: ApartmentState):
    """ตอบคำถามทั่วไป"""
    user_input = state["messages"][-1]
    response = llm.invoke(user_input)
    return {"messages": state["messages"] + [f"AI: {response.content}"]}

def router(state: ApartmentState) -> Literal["repair", "general"]:
    return state["category"]

workflow = StateGraph(ApartmentState)
workflow.add_node("classifier", classifier_node)
workflow.add_node("repair_node", repair_handler)
workflow.add_node("general_node", general_handler)

workflow.set_entry_point("classifier")
workflow.add_conditional_edges("classifier", router, {
    "repair": "repair_node",
    "general": "general_node"
})
workflow.add_edge("repair_node", END)
workflow.add_edge("general_node", END)

app = workflow.compile()

st.title("ระบบจัดการหอพักอัตโนมัติ (Local AI)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_query = st.chat_input("พิมพ์ข้อความแจ้งซ่อมหรือสอบถามได้ที่นี่...")

if user_query:
    st.session_state.chat_history.append(f"User: {user_query}")
    
    initial_state = {"messages": [user_query], "category": ""}
    final_output = app.invoke(initial_state)
    
    ai_response = final_output["messages"][-1]
    st.session_state.chat_history.append(ai_response)

for msg in st.session_state.chat_history:
    st.write(msg)