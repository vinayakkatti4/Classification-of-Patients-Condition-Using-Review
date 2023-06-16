#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import pickle
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords

stop = stopwords.words('english')
from PIL import Image


# In[2]:


st.set_page_config(
    page_title="Condition and Drug Name Prediction",
    page_icon=":pill:",
    layout="wide",
    initial_sidebar_state="expanded"
)


# In[3]:


st.title('Drug Classification')


# In[4]:


html_temp="""
<div style ="background-color:Black;padding:10px">
<h2 style="color:white;text-align:center;"> Condition and Drug Name Prediction </h2>
"""


# In[5]:


st.markdown(html_temp,unsafe_allow_html=True)


# In[6]:


image1=Image.open('j.jpeg')
st.image(image1)


# In[7]:


st.subheader('Group 3')


# In[8]:


model=pickle.load(open('drug.pkl','rb'))


# In[9]:


vectorizer=pickle.load(open('transform.pkl','rb'))


# In[10]:


df=pd.read_csv('drugsCom_raw (1).tsv', sep="\t",encoding='latin-1')


# In[11]:


from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
nltk.download('wordnet')


# In[12]:


stop = stopwords.words('english')
lemmatizer = WordNetLemmatizer()


# In[13]:


req=['Depression','High Blood Pressure','Diabetes, Type2']
drug=df[df['condition'].isin(req)]
drug


# In[14]:


def review_to_words(raw_review):
    # 1. Delete HTML 
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    # 2. Make a space
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    # 3. lower letters
    words = letters_only.lower().split()
    # 5. Stopwords 
    meaningful_words = [w for w in words if not w in stop]
    # 6. lemmitization
    lemmitize_words = [lemmatizer.lemmatize(w) for w in meaningful_words]
    # 7. space join words
    return( ' '.join(lemmitize_words))


# In[15]:


text=st.text_area('Enter text here')


# In[16]:


if st.button('submit'):
    cv=vectorizer.transform([text])
    pred=model.predict(cv)[0]
    st.subheader('Condition:')
    st.write(pred)
    df_top=drug[(drug['rating']>=9)&(drug['usefulCount']>=100)].sort_values(by=['rating','usefulCount'], ascending=[False,False])
    drug_list=df_top[df_top['condition']==pred]['drugName'].head(3).tolist()
   
    st.subheader("Recommended Drug")
    for i,drug2 in enumerate(drug_list):
        st.write(i+1,drug2)


# In[ ]:




