import streamlit as st
from rag import ask_bot

st.set_page_config(
    page_title="Zyro Dynamics HR Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Zyro Dynamics HR Assistant")

st.write(
    "Ask me anything about company HR policies, "
    "benefits, leave, payroll, and compliance."
)

question = st.text_input(
    "Enter your HR question:",
    placeholder="Example: How do I request PTO?"
)

if st.button("Ask Assistant"):

    if question.strip():

        with st.spinner("Searching HR policies..."):

            try:
                result = ask_bot(question)

                if isinstance(result, dict):
                    answer = result.get(
                        "answer",
                        "No answer was generated."
                    )
                else:
                    answer = str(result)

                st.subheader("Answer")
                st.write(answer)

            except Exception as e:
                st.error(f"Error: {e}")

    else:
        st.warning("Please enter a question.")
