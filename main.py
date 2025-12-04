import streamlit as st
import requests

API_URL = "http://127.0.0.1:5000/ask"
LOG_URL = "http://127.0.0.1:5000/logs?limit=5"

page = st.sidebar.radio("Navigation", ["Chat", "Logs"])

st.title("🩺 Medical AI Assistant")

# ---------------- CHAT PAGE ----------------
if page == "Chat":

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Ask something...")

    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        try:
            res = requests.post(API_URL, json={"question": user_input})
            answer = res.json().get("answer", "No answer")
        except:
            answer = "⚠️ Backend error"

        st.session_state["messages"].append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.write(answer)


# ---------------- LOGS PAGE ----------------
elif page == "Logs":
    st.subheader("📜 Backend Logs")

    if st.button("Refresh Logs"):
        try:
            res = requests.get(LOG_URL)   # <-- backend returns text
            logs = res.text               # <-- FIX
        except Exception as e:
            logs = f"❌ Could not fetch logs: {str(e)}"

        st.text_area("Backend Logs:", logs, height=500)
