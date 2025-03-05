#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk
from textblob import TextBlob

# Download necessary NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")
nltk.download("maxent_ne_chunker")
nltk.download("words")

# Streamlit App Title
st.title("📊 NLP in Taxation - Analyze Unstructured Data")

# Introduction
st.markdown("""
## 🧾 Natural Language Processing (NLP) for Tax Analysis
Upload any dataset containing **taxpayer communications, documents, or financial transactions** to analyze insights.
""")

# File Uploader
uploaded_file = st.file_uploader("📂 Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    # Check file format
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Display dataset
    st.write("### 📋 Uploaded Dataset")
    st.write(df.head())

    # User selects the text column
    text_column = st.selectbox("🔍 Select the column containing text for NLP analysis", df.columns)

    if text_column:
        # Sentiment Analysis
        st.subheader("📊 Sentiment Analysis")
        df["Sentiment"] = df[text_column].apply(lambda x: "Positive" if TextBlob(str(x)).sentiment.polarity > 0 
                                                else "Negative" if TextBlob(str(x)).sentiment.polarity < 0 
                                                else "Neutral")
        st.write(df[[text_column, "Sentiment"]])

        # Named Entity Recognition (NER) using NLTK
        st.subheader("🔍 Named Entity Recognition (NER)")
        def extract_entities(text):
            words = word_tokenize(str(text))
            pos_tags = pos_tag(words)
            named_entities = ne_chunk(pos_tags)
            return [" ".join(c[0] for c in chunk) for chunk in named_entities if hasattr(chunk, 'label')]

        df["Entities"] = df[text_column].apply(extract_entities)
        st.write(df[[text_column, "Entities"]])

        # Keyword Extraction using NLTK
        st.subheader("📌 Keyword Extraction")
        stop_words = set(stopwords.words("english"))
        def extract_keywords(text):
            words = word_tokenize(str(text))
            return [word for word in words if word.is_alpha and word.lower() not in stop_words]

        df["Keywords"] = df[text_column].apply(extract_keywords)
        st.write(df[[text_column, "Keywords"]])

        # Downloadable Processed File
        st.subheader("📥 Download Processed Data")
        output_file = "processed_data.csv"
        df.to_csv(output_file, index=False)
        st.download_button(label="Download CSV", data=df.to_csv(index=False), file_name="Processed_Data.csv", mime="text/csv")

# Conclusion
st.markdown("""
---
### 🎯 Key Takeaways:
- **Upload a dataset to analyze unstructured text using NLP techniques.**
- **Perform sentiment analysis, named entity recognition, and keyword extraction.**
- **Download the processed dataset with insights for further use.**
""")

st.success("🚀 Try uploading a dataset now!")


# In[ ]:




