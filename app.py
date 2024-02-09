import pandas as pd
import numpy as np
import streamlit as st
import pickle as pkl

df = pkl.load(open("DataFrame.pkl" ,"rb"))
pipe = pkl.load(open("Pipeline1.pkl" ,"rb"))

st.header("Predict Passanger is survived or not in Titanic Disaster" ,divider="rainbow")

# select box ,radio ,number input ,number imput ,select box ,family_size

pclass = st.selectbox("Select Passanger Class" ,df["Pclass"].unique(),index=None ,placeholder="select class......")

sex = st.radio(
    "Choose Your Sex",
    ["male", "female"],
    index=None,
)

age = st.slider('Select Passanger Age', 0, 130, 0)
    
fare = st.number_input("Titanic Ticket Fare" ,value=None , placeholder="Fare Amount (Max Fare amount is 600$) .....")

embarked = st.radio(
    "Choose your town",
    ["S" ,"C" ,"Q"],
    captions=["Southamption" ,"Cherbourg" ,"Queenston"],
    index=None,
)

family_size = st.slider("Select Passanger Family Size" ,1 ,10 ,1)

# Now we are predict 

st.header("",divider='rainbow')

if st.button('Predict'):
    y_pred = pipe.predict(pd.DataFrame(np.array([pclass ,sex ,age ,fare ,embarked ,family_size]).reshape(1,6) ,columns=df.columns[:6]) )
    predict = (y_pred.round()).astype(int)[0]

    if(predict == 1):
        st.subheader("Survived")
    else:
        st.subheader("Not Survived")