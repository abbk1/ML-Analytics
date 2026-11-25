import streamlit as st

st.set_page_config(
    page_title="ML Analytics Studio",
    layout="wide"
)

st.title("ML Analytics Studio")

st.subheader("Machine Learning Training and Evaluation")

st.success("Application successfully deployed!")

uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv"]
)

