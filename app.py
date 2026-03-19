import requests
import streamlit as st

HF_ENDPOINT = "https://router.huggingface.co/v1/chat/completions"
MODEL = "meta-llama/Llama-3.2-1B-Instruct"

st.set_page_config(page_title="My AI Chat", layout="wide")
st.title("My AI Chat")
st.caption("Task 1 - Part B: Multi-turn conversation")

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

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render conversation history in Streamlit's native chat UI.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt = st.chat_input("Type your message...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            assistant_reply = None

            headers = {"Authorization": f"Bearer {hf_token}"}
            payload = {
                "model": MODEL,
                "messages": st.session_state.messages,
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
                st.session_state.messages.append(
                    {"role": "assistant", "content": assistant_reply}
                )
