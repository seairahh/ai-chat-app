import requests
import streamlit as st
from datetime import datetime
import uuid
import json
from pathlib import Path
import time

HF_ENDPOINT = "https://router.huggingface.co/v1/chat/completions"
MODEL = "meta-llama/Llama-3.2-1B-Instruct"
CHATS_DIR = Path("chats")

# Ensure chats directory exists
CHATS_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="My AI Chat", layout="wide")
st.title("My AI Chat")
st.caption("Task 1-2: Chat persistence + Response streaming")

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

# ── Helper functions for JSON persistence ─────────────────────────────────────
def save_chat(chat_id: str, chat_data: dict):
    """Save a chat to a JSON file in the chats/ directory."""
    file_path = CHATS_DIR / f"{chat_id}.json"
    with open(file_path, "w") as f:
        json.dump(chat_data, f, indent=2)

def load_chat(chat_id: str) -> dict:
    """Load a chat from a JSON file."""
    file_path = CHATS_DIR / f"{chat_id}.json"
    if file_path.exists():
        with open(file_path, "r") as f:
            return json.load(f)
    return None

def delete_chat_file(chat_id: str):
    """Delete a chat's JSON file from disk."""
    file_path = CHATS_DIR / f"{chat_id}.json"
    if file_path.exists():
        file_path.unlink()

def load_all_chats() -> dict:
    """Load all chats from the chats/ directory."""
    chats = {}
    for file_path in CHATS_DIR.glob("*.json"):
        try:
            with open(file_path, "r") as f:
                chat_data = json.load(f)
                chat_id = chat_data.get("id", file_path.stem)
                chats[chat_id] = chat_data
        except Exception:
            pass
    return chats

def generate_title_from_message(text: str, max_length: int = 30) -> str:
    """Generate a chat title from the first user message."""
    text = text.strip()
    if len(text) <= max_length:
        return text
    return text[:max_length].rstrip() + "…"

def stream_response(response, placeholder):
    """
    Parse SSE stream from Hugging Face API and update placeholder incrementally.
    Returns the full concatenated response text.
    """
    full_response = ""
    
    try:
        for line in response.iter_lines():
            if not line:
                continue
            
            # Parse SSE format: "data: {json}"
            line_str = line.decode("utf-8") if isinstance(line, bytes) else line
            if line_str.startswith("data: "):
                try:
                    json_str = line_str[6:]  # Remove "data: " prefix
                    if json_str == "[DONE]":
                        break
                    
                    data = json.loads(json_str)
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    token = delta.get("content", "")
                    
                    if token:
                        full_response += token
                        # Update placeholder with accumulated response
                        placeholder.write(full_response)
                        # Add small delay for visibility
                        time.sleep(0.02)
                except (json.JSONDecodeError, KeyError, IndexError):
                    pass
    except Exception as e:
        placeholder.error(f"Streaming error: {e}")
        return None
    
    return full_response if full_response else None

# ── Multi-chat session state with persistence ──────────────────────────────────
if "chats" not in st.session_state:
    # Load all existing chats from disk
    st.session_state.chats = load_all_chats()
    
    # If no chats exist, create the first one
    if not st.session_state.chats:
        new_id = str(uuid.uuid4())
        st.session_state.chats[new_id] = {
            "id": new_id,
            "title": "Chat 1",
            "timestamp": datetime.now().isoformat(),
            "messages": [],
        }
        save_chat(new_id, st.session_state.chats[new_id])

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = next(iter(st.session_state.chats))

# ── Sidebar: New Chat button and chat list ────────────────────────────────────
with st.sidebar:
    st.header("💬 Chats")
    
    if st.button("➕ New Chat", use_container_width=True):
        new_id = str(uuid.uuid4())
        chat_count = len(st.session_state.chats) + 1
        new_chat = {
            "id": new_id,
            "title": f"Chat {chat_count}",
            "timestamp": datetime.now().isoformat(),
            "messages": [],
        }
        st.session_state.chats[new_id] = new_chat
        save_chat(new_id, new_chat)
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
                # Delete from disk
                delete_chat_file(chat_id)
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
                        new_chat = {
                            "id": new_id,
                            "title": "Chat 1",
                            "timestamp": datetime.now().isoformat(),
                            "messages": [],
                        }
                        st.session_state.chats[new_id] = new_chat
                        save_chat(new_id, new_chat)
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
    
    # Auto-generate title from first message if it's still a default
    if len(current_chat["messages"]) == 1 and current_chat["title"].startswith("Chat"):
        current_chat["title"] = generate_title_from_message(prompt)
    
    # Save to disk after adding user message
    save_chat(st.session_state.current_chat_id, current_chat)
    
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            assistant_reply = None
            response_placeholder = st.empty()

            headers = {"Authorization": f"Bearer {hf_token}"}
            payload = {
                "model": MODEL,
                "messages": current_chat["messages"],
                "max_tokens": 512,
                "stream": True,
            }

            try:
                response = requests.post(
                    HF_ENDPOINT,
                    headers=headers,
                    json=payload,
                    timeout=30,
                    stream=True,
                )

                if response.status_code == 200:
                    assistant_reply = stream_response(response, response_placeholder)
                    if assistant_reply is None:
                        assistant_reply = "Streaming completed but no content received."
                elif response.status_code in {401, 403}:
                    assistant_reply = (
                        "I could not access the model because the Hugging Face token "
                        "is invalid (401/403)."
                    )
                    response_placeholder.error(assistant_reply)
                elif response.status_code == 429:
                    assistant_reply = (
                        "I am currently rate limited by Hugging Face (429). "
                        "Please wait and try again."
                    )
                    response_placeholder.warning(assistant_reply)
                else:
                    assistant_reply = (
                        f"API error {response.status_code}: {response.text}"
                    )
                    response_placeholder.error(assistant_reply)
            except requests.exceptions.Timeout:
                assistant_reply = "Network timeout. Please try again."
                response_placeholder.error(assistant_reply)
            except requests.exceptions.RequestException as exc:
                assistant_reply = f"Network error: {exc}"
                response_placeholder.error(assistant_reply)
            except Exception as exc:
                assistant_reply = f"Unexpected error: {exc}"
                response_placeholder.error(assistant_reply)

            if assistant_reply is not None:
                current_chat["messages"].append(
                    {"role": "assistant", "content": assistant_reply}
                )
                # Save to disk after adding assistant response
                save_chat(st.session_state.current_chat_id, current_chat)
