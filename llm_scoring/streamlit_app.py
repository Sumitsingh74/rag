# streamlit_app.py

import streamlit as st
import requests

# --- Sidebar: configure API URL ---
st.sidebar.header("Configuration")
api_url = st.sidebar.text_input(
    "Scoring API URL",
    value="http://127.0.0.1:8001/score"
)

st.title("📝 QA Scoring Front-End")

# --- Session state for accumulating Q&A pairs ---
if "qa_pairs" not in st.session_state:
    st.session_state.qa_pairs = []

# --- Form to add a new Q&A pair ---
with st.form("add_pair_form", clear_on_submit=True):
    st.subheader("Add a Q&A Pair")
    question = st.text_input("Question")
    answer = st.text_area("Answer")
    submitted = st.form_submit_button("➕ Add Pair")
    if submitted:
        if question.strip() and answer.strip():
            st.session_state.qa_pairs.append({
                "question": question.strip(),
                "answer": answer.strip(),
            })
            st.success("Pair added!")
        else:
            st.error("Both question and answer are required.")

# --- Display current list of pairs ---
if st.session_state.qa_pairs:
    st.subheader("Current Q&A Pairs")
    for idx, pair in enumerate(st.session_state.qa_pairs, start=1):
        st.markdown(f"**{idx}. Q:** {pair['question']}")
        st.markdown(f"   **A:** {pair['answer']}")
    st.markdown("---")

    # --- Button to submit all pairs for scoring ---
    if st.button("▶️ Score All Pairs"):
        with st.spinner("Scoring..."):
            try:
                resp = requests.post(api_url, json={"qa_pairs": st.session_state.qa_pairs})
                resp.raise_for_status()
                data = resp.json().get("results", [])
                if not data:
                    st.warning("No results returned.")
                else:
                    st.success("✅ Scoring complete!")
                    st.subheader("Results")
                    for res in data:
                        st.markdown(f"**Q:** {res['question']}")
                        st.markdown(f"**A:** {res['answer']}")
                        # iterate over all other keys
                        for k, v in res.items():
                            if k in ("question", "answer"):
                                continue
                            st.markdown(f"- **{k.replace('_',' ').title()}:** {v}")
                        st.markdown("---")
            except Exception as e:
                st.error(f"API request failed: {e}")

    # --- Button to clear pairs ---
    if st.button("🗑️ Clear All Pairs"):
        st.session_state.qa_pairs.clear()
        st.success("Cleared all pairs.")

else:
    st.info("Add one or more Q&A pairs to get started.")
