"""
app.py  —  Streamlit chatbot with Groq model & personality picker
Prereqs:  pip install streamlit openai python-dotenv
Secrets:  GROQ_API_KEY in .env  (or in Streamlit Cloud’s Secrets)
"""

# ── Streamlit MUST come first so we can set page config immediately ──────────
# ✅ FIRST: Import Streamlit and set page config immediately
import streamlit as st
st.set_page_config(page_title="Groq Chatbot", page_icon="🤖")

# ✅ THEN: Import all other packages
import os
from dotenv import load_dotenv
from openai import OpenAI

# ── Load API key ─────────────────────────────────────────────────────────────
load_dotenv()
GROQ_KEY = os.getenv("GROQ_API_KEY")or st.secrets.get("GROQ_API_KEY", "")
if not GROQ_KEY:
    st.error("⚠️  GROQ_API_KEY not found. Add it to .env or Streamlit Secrets.")
    st.stop()

client = OpenAI(
    api_key=GROQ_KEY,
    base_url="https://api.groq.com/openai/v1",
)

# ── Sidebar: model & personality selectors ───────────────────────────────────
st.sidebar.title("⚙️ Settings")

models = {
    "Llama‑3‑70B (versatile)": "llama-3.3-70b-versatile",
    "Mixtral‑8x7B‑32k": "mixtral-8x7b-32768",
    "Gemma‑7B‑IT": "gemma-7b-it",
}
model_label = st.sidebar.selectbox("Groq model", list(models.keys()))
model_name = models[model_label]

personalities = {
    "Math Teacher": "You are a patient math teacher. "
                    "Only answer math questions. Politely refuse others.",
    "Doctor": "You are a licensed doctor. "
              "Answer general health questions (no prescriptions). Refuse others.",
    "Travel Guide": "You are a helpful travel guide. "
                    "Only give travel advice. Refuse others.",
    "Chef": "You are an experienced chef. "
            "Only discuss cooking, recipes, food. Refuse others.",
    "Tech Support": "You are a tech‑support specialist. "
                    "Only troubleshoot tech issues. Refuse others.",
}
persona_name = st.sidebar.radio("Chatbot personality", list(personalities.keys()))
system_prompt = personalities[persona_name]

st.sidebar.caption("💡 Bot will refuse questions outside its scope.")

# ── Title ────────────────────────────────────────────────────────────────────
st.title(f"🤖 {persona_name} — {model_label}")

# ── Session‑state memory ─────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": system_prompt}
    ]
# Reset history if personality changes
if st.session_state.messages[0]["content"] != system_prompt:
    st.session_state.messages = [
        {"role": "system", "content": system_prompt}
    ]

# ── Replay history ───────────────────────────────────────────────────────────
for m in st.session_state.messages[1:]:
    st.chat_message(m["role"]).write(m["content"])

# ── User input ───────────────────────────────────────────────────────────────
prompt = st.chat_input("Type your message…")

if prompt:
    # show user bubble
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # stream Groq response
    with st.chat_message("assistant"):
        full_resp, placeholder = "", st.empty()
        try:
            stream = client.chat.completions.create(
                model=model_name,
                messages=st.session_state.messages,
                stream=True,
                max_tokens=1024,
                temperature=0.7,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                full_resp += delta
                placeholder.write(full_resp)
        except Exception as err:
            full_resp = f"⚠️ Error: {err}"
            placeholder.write(full_resp)

    st.session_state.messages.append({"role": "assistant", "content": full_resp})
