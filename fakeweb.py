import streamlit as st
import joblib
import re
import requests

model = joblib.load("svm.jb")
vectorizer = joblib.load("vectorizer.jb")


NEWS_API_KEY = "a8a1ce1609b940fca146c137bc775fb8"
GNEWS_API_KEY = "5f7325b62453daf059458e7331f35cdb"


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    return text


def optimize_query(text):
    words = clean_text(text).split()

    
    keywords = [w for w in words if len(w) > 3]

    return " ".join(keywords[:6])  


def predict_news(text):
    text_clean = clean_text(text)
    vector = vectorizer.transform([text_clean])

    prediction = model.predict(vector)[0]
    score = model.decision_function(vector)[0]

    
    import numpy as np
    confidence = 1 / (1 + np.exp(-score))
    confidence = int(confidence * 100)

    if prediction == 1:
        label = "Real News ✅"
    else:
        label = "Fake News ❌"

    return label, confidence


def search_newsapi(query):
    url = "https://newsapi.org/v2/everything"

    params = {
        "q": query,
        "apiKey": NEWS_API_KEY,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 5
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()

        results = []

        if data.get("status") == "ok":
            for article in data.get("articles", []):
                title = article["title"]
                source = article["source"]["name"]
                results.append(f"{title} ({source})")

        return results

    except:
        return []


def search_gnews(query):
    url = "https://gnews.io/api/v4/search"

    params = {
        "q": query,
        "token": GNEWS_API_KEY,
        "lang": "en",
        "max": 5
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()

        results = []

        if "articles" in data:
            for article in data["articles"]:
                title = article["title"]
                source = article["source"]["name"]
                results.append(f"{title} ({source})")

        return results

    except:
        return []


def combined_search(query):
    results1 = search_newsapi(query)
    results2 = search_gnews(query)

    return list(set(results1 + results2))   


def is_related(text, results):
    text_words = set(clean_text(text).split())

    for r in results:
        r_words = set(clean_text(r).split())

        if len(text_words) == 0:
            continue

        overlap = len(text_words.intersection(r_words)) / len(text_words)

        if overlap > 0.2:
            return True

    return False


trusted_sources = ["bbc", "reuters", "cnn", "ndtv", "the hindu"]

def has_trusted_source(results):
    for r in results:
        for source in trusted_sources:
            if source in r.lower():
                return True
    return False


def final_verdict(ml_result, related, trusted, web_results):

    
    if len(web_results) >= 4 and (related or trusted):
        return "Likely Real ✅"

    
    elif ml_result == "Fake News" and not related:
        return "Likely Fake ❌"

    
    else:
        return "Uncertain ⚠️"


st.title("📰 Fake News Detection (ML + Multi-API)")

user_input = st.text_area("Enter News Text")

if st.button("Check News"):

    if user_input.strip() == "":
        st.warning("Please enter some text")

    else:
        ml_result, confidence = predict_news(user_input)

        query = optimize_query(user_input)

        web_results = combined_search(query)

        related = is_related(user_input, web_results)
        trusted = has_trusted_source(web_results)

        verdict = final_verdict(ml_result, related, trusted, web_results)

       
        st.subheader("🔍 ML Prediction")
        st.write(ml_result)

        st.subheader("📊 Confidence Score")
        st.write(f"{confidence}%")

        if confidence < 50:
         st.warning("⚠️ Low confidence — prediction may not be reliable")

        st.subheader("🌐 API Results")
        st.write(f"Query used: `{query}`")

        if web_results:
            for r in web_results:
                st.write("- " + r)
        else:
            st.write("No results found")

        st.subheader("🎯 Final Verdict")

        if "Real" in verdict:
            st.success(verdict)
        elif "Fake" in verdict:
            st.error(verdict)
        else:
            st.warning(verdict)