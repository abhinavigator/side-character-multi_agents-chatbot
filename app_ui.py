





import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# --- Page configuration ---
st.set_page_config(page_title="Side Character AI", page_icon="🤖", layout="wide")

# --- Ensure app modules are importable ---
sys.path.append(str(Path(__file__).resolve().parent / "src"))

# --- App imports ---
from src.side_character_app.app.state import initialize_state, ARCHETYPES
from src.side_character_app.app.agents import create_all_agents
from src.side_character_app.app.graph import create_graph

from pymilvus import MilvusClient
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# --- Avatar mapping ---
AVATAR_MAP = {
    "Wise Mentor": "🧙‍♂️",
    "Comedic Relief": "😂",
    "Skeptical Realist": "🤔",
    "Loyal Sidekick": "🐶",
    "Auto Mode": "🤖",
    "You": "👤",
    "System": "⚙️"
}

# --- Backend initialization (cached) ---
@st.cache_resource
def get_app():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("⚠️ Please set GEMINI_API_KEY in your .env file.")
        st.stop()

    root = Path(__file__).resolve().parent
    db_path = root / "data" / "vector_stores" / "milvus_side_characters.db"
    
    try:
        client = MilvusClient(str(db_path))          # or remote config
    except Exception as e:
        st.error(f"❌ Milvus initialisation failed: {e}")
        st.stop()


    # client = MilvusClient(str(db_path))

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0.7
    )

    agents = create_all_agents(llm, client, embeddings)
    return create_graph(llm, agents)

app = get_app()

# --- Session state initialization ---
if "graph_state" not in st.session_state:
    st.session_state.graph_state = initialize_state()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Sidebar for mode selection ---
st.sidebar.title("🗣️ Interaction Mode")
MODE_OPTIONS = ["Auto Mode"] + ARCHETYPES
st.sidebar.radio("Choose who to talk to:", MODE_OPTIONS, key="mode")

# --- Main title ---
st.title("🤖 Side Character AI Chat")

# --- Callback for handling a new message ---
def submit_message():
    # Get the user's input from the widget
    user_input = st.session_state.user_input
    if not user_input:
        return

    # 1. Append user message to history
    st.session_state.chat_history.append({
        "role": "user",
        "display_name": "You",
        "avatar_key": "You",
        "content": user_input
    })

    # 2. Call the backend
    mode_selected = st.session_state.mode
    user_choice = "" if mode_selected == "Auto Mode" else mode_selected
    payload = {
        **st.session_state.graph_state,
        "input": user_input,
        "user_choice": user_choice
    }
    new_state = app.invoke(
        payload,
        config={"configurable": {"thread_id": "main_convo"}}
    )
    st.session_state.graph_state = new_state

    # 3. Append assistant reply to history
    last_agent = new_state.get("next")
    if last_agent and last_agent != "END":
        reply = new_state["private_conversations"][last_agent][-1].content
        display_name = last_agent
        avatar_key = last_agent
    else:
        reply = new_state["main_conversation"][-1].content
        display_name = "System"
        avatar_key = "System"

    st.session_state.chat_history.append({
        "role": "assistant",
        "display_name": display_name,
        "avatar_key": avatar_key,
        "content": reply
    })

# --- Display the chat history above the input ---
for msg in st.session_state.chat_history:
    with st.chat_message(name=msg["role"], avatar=AVATAR_MAP.get(msg["avatar_key"], "🤖")):
        st.markdown(f"**{msg['display_name']} says:**  \n{msg['content']}")

# --- Input box at the bottom, triggers submit_message() on Enter ---
st.chat_input("Type your message...", key="user_input", on_submit=submit_message)
