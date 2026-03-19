import requests
import streamlit as st

HF_ENDPOINT = "https://router.huggingface.co/v1/chat/completions"
MODEL = "meta-llama/Llama-3.2-1B-Instruct"

st.set_page_config(page_title="My AI Chat", layout="wide")
st.title("My AI Chat")
st.caption("Task 1 - Part A: Page setup and API test connection")

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

st.subheader("API Connection Test")
st.write("Sending hardcoded test message: **Hello!**")

if "part_a_response" not in st.session_state and "part_a_error" not in st.session_state:
    with st.spinner("Contacting Hugging Face API..."):
        try:
            headers = {"Authorization": f"Bearer {hf_token}"}
            payload = {
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 512,
            }
            response = requests.post(
                HF_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=30,
            )

            if response.status_code == 200:
                data = response.json()
                st.session_state.part_a_response = data["choices"][0]["message"]["content"]
            elif response.status_code in {401, 403}:
                st.session_state.part_a_error = (
                    "Invalid Hugging Face token (401/403). "
                    "Check Streamlit secrets and try again."
                )
            elif response.status_code == 429:
                st.session_state.part_a_error = (
                    "Rate limit reached (429). Wait a moment and rerun."
                )
            else:
                st.session_state.part_a_error = (
                    f"API error {response.status_code}: {response.text}"
                )
        except requests.exceptions.Timeout:
            st.session_state.part_a_error = "Network timeout. Please rerun the app."
        except requests.exceptions.RequestException as exc:
            st.session_state.part_a_error = f"Network error: {exc}"
        except Exception as exc:
            st.session_state.part_a_error = f"Unexpected error: {exc}"

if "part_a_response" in st.session_state:
    st.success("API call succeeded.")
    st.markdown("**Model response:**")
    st.write(st.session_state.part_a_response)
elif "part_a_error" in st.session_state:
    st.error(st.session_state.part_a_error)

if st.button("Rerun API Test"):
    st.session_state.pop("part_a_response", None)
    st.session_state.pop("part_a_error", None)
    st.rerun()
