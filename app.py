import streamlit as st
import requests

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Chat", page_icon="🤖")
st.title("🤖 AI Chat")

# ── Token check ────────────────────────────────────────────────────────────────
if "HF_TOKEN" not in st.secrets:
    st.error(
        "Hugging Face token not found. "
        "Add `HF_TOKEN` to your `.streamlit/secrets.toml` file and restart the app."
    )
    st.stop()

hf_token = st.secrets["HF_TOKEN"]

HF_ENDPOINT = "https://router.huggingface.co/v1/chat/completions"
MODEL = "meta-llama/Llama-3.2-1B-Instruct"

# ── Chat history in session state ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Display existing messages ──────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ── Chat input ─────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask me anything...")

if user_input:
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Call Hugging Face Inference Router
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                headers = {"Authorization": f"Bearer {hf_token}"}
                payload = {
                    "model": MODEL,
                    "messages": st.session_state.messages,
                    "max_tokens": 512,
                }
                response = requests.post(
                    HF_ENDPOINT, headers=headers, json=payload, timeout=30
                )

                if response.status_code == 200:
                    reply = response.json()["choices"][0]["message"]["content"]
                    st.write(reply)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": reply}
                    )
                elif response.status_code == 429:
                    st.warning("Rate limit reached. Please wait a moment and try again.")
                else:
                    st.error(f"API error {response.status_code}: {response.text}")

            except requests.exceptions.Timeout:
                st.error("Request timed out. The model may be busy — please try again.")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
