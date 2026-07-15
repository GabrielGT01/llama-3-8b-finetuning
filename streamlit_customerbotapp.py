
import streamlit as st
import requests
import json
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="Customer Support Assistant",
    page_icon="💬",
    layout="wide"
)

# --- API Configuration ---
API_GATEWAY_URL = "https://n4dx8aaa9e.execute-api.eu-central-1.amazonaws.com/invoke-llm-endpoint"




# --- Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono&display=swap');

    * { font-family: 'DM Sans', sans-serif; }

    .stApp {
        background-color: #0f1117;
        color: #e8e8e8;
    }

    /* Hide default streamlit elements */
    #MainMenu, footer, header { visibility: hidden; }

    /* Chat container */
    .chat-container {
        max-width: 780px;
        margin: 0 auto;
        padding: 1rem 0;
    }

    /* User message bubble */
    .user-bubble {
        background: #1e6feb;
        color: #ffffff;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0 8px 80px;
        font-size: 15px;
        line-height: 1.6;
        word-wrap: break-word;
    }

    /* Bot message bubble */
    .bot-bubble {
        background: #1e2130;
        color: #e8e8e8;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 80px 8px 0;
        font-size: 15px;
        line-height: 1.6;
        border: 1px solid #2a2f42;
        word-wrap: break-word;
    }

    /* Sender labels */
    .label-user {
        text-align: right;
        font-size: 11px;
        color: #6b7280;
        margin: 4px 4px 0 0;
        font-family: 'DM Mono', monospace;
    }

    .label-bot {
        text-align: left;
        font-size: 11px;
        color: #6b7280;
        margin: 4px 0 0 4px;
        font-family: 'DM Mono', monospace;
    }

    /* Header */
    .chat-header {
        text-align: center;
        padding: 2rem 0 1rem;
        border-bottom: 1px solid #2a2f42;
        margin-bottom: 1.5rem;
    }

    .chat-header h1 {
        font-size: 22px;
        font-weight: 600;
        color: #e8e8e8;
        margin: 0;
        letter-spacing: -0.3px;
    }

    .chat-header p {
        font-size: 13px;
        color: #6b7280;
        margin: 6px 0 0;
    }

    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        background: #22c55e;
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }

    /* Input area */
    .stTextArea textarea {
        background: #1e2130 !important;
        border: 1px solid #2a2f42 !important;
        border-radius: 12px !important;
        color: #e8e8e8 !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 15px !important;
        padding: 12px 16px !important;
        resize: none !important;
    }

    .stTextArea textarea:focus {
        border-color: #1e6feb !important;
        box-shadow: 0 0 0 2px rgba(30, 111, 235, 0.2) !important;
    }

    /* Button */
    .stButton > button {
        background: #1e6feb !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 28px !important;
        font-size: 15px !important;
        font-weight: 500 !important;
        font-family: 'DM Sans', sans-serif !important;
        width: 100% !important;
        transition: background 0.2s ease !important;
        cursor: pointer !important;
    }

    .stButton > button:hover {
        background: #1a5fd4 !important;
    }

    /* Clear button */
    .clear-btn > button {
        background: transparent !important;
        color: #6b7280 !important;
        border: 1px solid #2a2f42 !important;
        font-size: 13px !important;
        padding: 6px 16px !important;
        width: auto !important;
    }

    /* Divider */
    hr { border-color: #2a2f42 !important; }

    /* Spinner */
    .stSpinner > div { border-top-color: #1e6feb !important; }

    /* Suggested questions */
    .suggestion-chip {
        display: inline-block;
        background: #1e2130;
        border: 1px solid #2a2f42;
        color: #9ca3af;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 13px;
        margin: 4px;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0

# --- Helper: Call API ---
def call_api(prompt: str) -> str:
    headers = {"Content-Type": "application/json"}
    payload = json.dumps({"prompt": prompt})

    try:
        response = requests.post(API_GATEWAY_URL, data=payload, headers=headers, timeout=30)
        response.raise_for_status()
        result = response.json()

        # handle API Gateway wrapper
        if isinstance(result, dict) and "body" in result:
            body = result["body"]
            if isinstance(body, str):
                body = json.loads(body)
        else:
            body = result

        if "error" in body:
            return f"⚠️ Backend error: {body['error']}"

        return body.get("generated_text", "No response generated.")

    except requests.exceptions.Timeout:
        return "⚠️ The request timed out. Please try again."
    except requests.exceptions.HTTPError as e:
        return f"⚠️ HTTP error: {e}"
    except requests.exceptions.RequestException as e:
        return f"⚠️ Connection error: {e}"
    except json.JSONDecodeError:
        return "⚠️ Could not parse the response from the server."
    except Exception as e:
        return f"⚠️ Unexpected error: {e}"


# --- Header ---
st.markdown("""
<div class="chat-header">
    <h1>💬 Customer Support Assistant</h1>
    <p><span class="status-dot"></span>Powered by LLaMA-3 · Fine-tuned on customer support data</p>
</div>
""", unsafe_allow_html=True)

# --- Layout: Chat + Sidebar ---
col_chat, col_side = st.columns([3, 1])

with col_side:
    st.markdown("#### 💡 Try asking")
    suggestions = [
        "What is your return policy?",
        "How do I cancel an order?",
        "How do I track my delivery?",
        "I was charged twice, help!",
        "How do I reset my password?",
        "Where is my refund?",
    ]
    for s in suggestions:
        if st.button(s, key=f"sug_{s}"):
            st.session_state["prefill"] = s

    st.markdown("---")
    st.markdown(f"**Queries this session:** {st.session_state.total_queries}")

    st.markdown("---")
    with st.container():
        st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
        if st.button("🗑 Clear chat"):
            st.session_state.messages = []
            st.session_state.total_queries = 0
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

with col_chat:
    # --- Chat History ---
    for msg in st.session_state.messages:
        ts = msg.get("time", "")
        if msg["role"] == "user":
            st.markdown(f'<div class="label-user">You · {ts}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="label-bot">Assistant · {ts}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="bot-bubble">{msg["content"]}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Input Form ---
    prefill = st.session_state.pop("prefill", "")

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "Your message",
            value=prefill,
            placeholder="Type your question here...",
            height=100,
            label_visibility="collapsed"
        )
        submitted = st.form_submit_button("Send ➤")

    # --- Handle Submission ---
    if submitted and user_input.strip():
        now = datetime.now().strftime("%H:%M")

        # save user message
        st.session_state.messages.append({
            "role":    "user",
            "content": user_input.strip(),
            "time":    now
        })
        st.session_state.total_queries += 1

        # call API
        with st.spinner("Thinking..."):
            reply = call_api(user_input.strip())

        # save assistant message
        st.session_state.messages.append({
            "role":    "assistant",
            "content": reply,
            "time":    datetime.now().strftime("%H:%M")
        })

        st.rerun()

    elif submitted:
        st.warning("Please type a message before sending.")
