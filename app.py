from http.client import ImproperConnectionState
import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO 
from pyresparser import ResumeParser
import os 
from docx import Document
from yaml import DocumentEndEvent
from PIL import Image
from utils import ngrams
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


stopw  = set(stopwords.words('english'))


## Data for ML 
df =pd.read_csv('job_final.csv') 
df['test']=df['Job_Description'].apply(lambda x: ' '.join([word for word in str(x).split() if len(word)>2 and word not in (stopw)]))


st.title("Welcome to Resume Parser")

image1 =Image.open("assets/resume.png")
st.image(image1, caption='Online resume Parser ')

uploaded_file = st.file_uploader("Choose a pdf file or word document file")
if uploaded_file is not None:
    print(uploaded_file.name)
    filed = uploaded_file.name
    try:
        doc = Document()
        with open(filed, 'r') as file:
            doc.add_paragraph(file.read())
            doc.save("text.docx")
        data = ResumeParser('text.docx').get_extracted_data()
        #print(data['skills'])
        
        st.subheader("Here are your details")
        for key in data:
            st.write(key, ':', data[key])

    except:
        data = ResumeParser(filed).get_extracted_data()

        st.subheader("Here are your details")
        for key in data:
            st.write(key, ':', data[key])
    



    st.subheader("Some of the Job Recomendations based on your Profile")
    # machine learning path 
    vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams, lowercase=False)
    
    #extraccting the skills and appending into a simple tuple 
    skills2 = data["skills"]
    skills=[]
    skills.append(' '.join(word for word in skills2))
    cleaned_skills = skills


    tfidf = vectorizer.fit_transform(cleaned_skills)
    print(cleaned_skills)
    print('Vec complete ')


    def getNearestN(query):
        queryTFIDF_ = vectorizer.transform(query)
        distances, indices = nbrs.kneighbors(queryTFIDF_)
        return distances, indices
    nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(tfidf)
    unique_org = (df['test'].values)
    distances, indices = getNearestN(unique_org)
    unique_org = list(unique_org)
    matches = []
    for i,j in enumerate(indices):
        dist=round(distances[i][0],2)

        temp = [dist]
        matches.append(temp)
    matches = pd.DataFrame(matches, columns=['Match confidence'])

    
    df['match']=matches['Match confidence']
    df1=df.sort_values('match')
    df2=df1[['Position', 'Company','Location']].head(10).reset_index()

    st.table(df2)
    print(df2) 
