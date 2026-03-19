import requests
import streamlit as st
from datetime import datetime
import uuid

HF_ENDPOINT = "https://router.huggingface.co/v1/chat/completions"
MODEL = "meta-llama/Llama-3.2-1B-Instruct"

st.set_page_config(page_title="My AI Chat", layout="wide")
st.title("My AI Chat")
st.caption("Task 1 - Part C: Chat management")

# Requirement: load token using st.secrets["HF_TOKEN"] and do not crash if missing.
try:
    hf_token = st.secrets["HF_TOKEN"].strip()
except Exception:
    hf_token = ""

if not hf_token:
    st.error(
        "Missing Hugging Face token. Add HF_TOKEN to Streamlit secrets "
        "and rerun the app."
    )
    st.stop()

# ── Multi-chat session state ──────────────────────────────────────────────────
if "chats" not in st.session_state:
    st.session_state.chats = {}

if "current_chat_id" not in st.session_state:
    new_id = str(uuid.uuid4())
    st.session_state.chats[new_id] = {
        "title": "Chat 1",
        "timestamp": datetime.now().isoformat(),
        "messages": [],
    }
    st.session_state.current_chat_id = new_id

# ── Sidebar: New Chat button and chat list ────────────────────────────────────
with st.sidebar:
    st.header("💬 Chats")
    
    if st.button("➕ New Chat", use_container_width=True):
        new_id = str(uuid.uuid4())
        chat_count = len(st.session_state.chats) + 1
        st.session_state.chats[new_id] = {
            "title": f"Chat {chat_count}",
            "timestamp": datetime.now().isoformat(),
            "messages": [],
        }
        st.session_state.current_chat_id = new_id
        st.rerun()
    
    st.divider()
    
    # Scrollable chat list
    for chat_id, chat_data in st.session_state.chats.items():
        is_active = chat_id == st.session_state.current_chat_id
        
        col1, col2 = st.sidebar.columns([0.85, 0.15])
        
        with col1:
            # Highlight active chat
            chat_label = f"🔵 {chat_data['title']}" if is_active else chat_data['title']
            if st.button(
                chat_label,
                key=f"chat_{chat_id}",
                use_container_width=True,
            ):
                st.session_state.current_chat_id = chat_id
                st.rerun()
        
        with col2:
            if st.button(
                "✕",
                key=f"delete_{chat_id}",
                help="Delete this chat",
            ):
                del st.session_state.chats[chat_id]
                # If deleted chat was active, switch to another
                if st.session_state.current_chat_id == chat_id:
                    if st.session_state.chats:
                        st.session_state.current_chat_id = next(
                            iter(st.session_state.chats)
                        )
                    else:
                        # No chats left, create a new one
                        new_id = str(uuid.uuid4())
                        st.session_state.chats[new_id] = {
                            "title": "Chat 1",
                            "timestamp": datetime.now().isoformat(),
                            "messages": [],
                        }
                        st.session_state.current_chat_id = new_id
                st.rerun()

# ── Main area: Messages and input ──────────────────────────────────────────────
current_chat = st.session_state.chats[st.session_state.current_chat_id]

# Render conversation history in Streamlit's native chat UI.
for msg in current_chat["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt = st.chat_input("Type your message...")

if prompt:
    current_chat["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            assistant_reply = None

            headers = {"Authorization": f"Bearer {hf_token}"}
            payload = {
                "model": MODEL,
                "messages": current_chat["messages"],
                "max_tokens": 512,
            }

            try:
                response = requests.post(
                    HF_ENDPOINT,
                    headers=headers,
                    json=payload,
                    timeout=30,
                )

                if response.status_code == 200:
                    data = response.json()
                    assistant_reply = data["choices"][0]["message"]["content"]
                    st.write(assistant_reply)
                elif response.status_code in {401, 403}:
                    assistant_reply = (
                        "I could not access the model because the Hugging Face token "
                        "is invalid (401/403)."
                    )
                    st.error(assistant_reply)
                elif response.status_code == 429:
                    assistant_reply = (
                        "I am currently rate limited by Hugging Face (429). "
                        "Please wait and try again."
                    )
                    st.warning(assistant_reply)
                else:
                    assistant_reply = (
                        f"API error {response.status_code}: {response.text}"
                    )
                    st.error(assistant_reply)
            except requests.exceptions.Timeout:
                assistant_reply = "Network timeout. Please try again."
                st.error(assistant_reply)
            except requests.exceptions.RequestException as exc:
                assistant_reply = f"Network error: {exc}"
                st.error(assistant_reply)
            except Exception as exc:
                assistant_reply = f"Unexpected error: {exc}"
                st.error(assistant_reply)

            if assistant_reply is not None:
                current_chat["messages"].append(
                    {"role": "assistant", "content": assistant_reply}
                )
