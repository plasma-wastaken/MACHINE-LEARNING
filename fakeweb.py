import streamlit as st
import joblib
import re
import requests
from bs4 import BeautifulSoup


model = joblib.load("svm.jb")
vectorizer = joblib.load("vectorizer.jb")


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    return text

# ML PRED


def predict_news(text):
    text_clean = clean_text(text)
    vector = vectorizer.transform([text_clean])
    prediction = model.predict(vector)
    return "Real News" if prediction[0] == 1 else "Fake News"

# WEB SEARCH


def search_news(query):
    try:
        query = query.replace(" ", "+")
        url = f"https://www.google.com/search?q={query}"

        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)

        soup = BeautifulSoup(response.text, "html.parser")

        results = []
        for h in soup.find_all("h3"):
            results.append(h.text)

        return results[:5]

    except:
        return []


def is_related(text, results):
    text_words = set(text.lower().split())

    for r in results:
        r_words = set(r.lower().split())
        if len(text_words.intersection(r_words)) >= 3:
            return True
    return False


st.title("📰 Fake News Detection with Web Verification")

user_input = st.text_area("Enter News Text")

if st.button("Check News"):

    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        ml_result = predict_news(user_input)
        web_results = search_news(user_input)
        related = is_related(user_input, web_results)

        st.subheader("🔍 ML Prediction")
        st.write(ml_result)

        st.subheader("🌐 Web Results")

        if web_results:
            for r in web_results:
                st.write("- " + r)
        else:
            st.write("No results found")

        if ml_result == "Real News" and related:
            verdict = "Likely Real ✅"
            st.success(verdict)

        elif ml_result == "Fake News" and not related:
            verdict = "Likely Fake ❌"
            st.error(verdict)

        else:
            verdict = "Uncertain ⚠️"
            st.warning(verdict)
