# only for testing streamlit

import streamlit as st
st.title("Uploader test")
f = st.file_uploader("Pick a file")
st.write("Got:", getattr(f, "name", None))
