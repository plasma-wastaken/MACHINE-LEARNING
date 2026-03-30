import streamlit as st
import joblib

vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("svm.jb")

st.title("Fake News Detector")
st.write("Enter a News Article Below to Check")

news_input = st.text_area("News Article:","")

if st.button("check News"):
    if news_input.strip():
        transform_input = vectorizer.transform([news_input])
        prediction = model.predict(transform_input)

        if prediction[0]==1:
            st.success("REAL NEWS")
        else:
            st.error("FALSE NEWS")
    else:
        st.warning("Please Enter Some Text to search.")