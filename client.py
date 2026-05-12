import streamlit as st
import os
import requests


def get_response(query):
    response = requests.post(
        "http://localhost:8000/AttentionQnA/invoke", json={"input": query}
    )
    return response.json()["output"]


st.title("An Expert on Attention Is All You Need!")
query = st.text_input("What would you like to know about Attention Is All You Need?")

if query:
    res = get_response(query)
    st.write(res)
