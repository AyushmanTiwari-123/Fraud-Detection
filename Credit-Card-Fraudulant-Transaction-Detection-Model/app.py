import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# load data
data = pd.read_csv('creditcard.csv')

# separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# undersample legitimate transactions to balance the classes
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

# split data into training and testing sets
X = data.drop(columns="Class", axis=1)
y = data["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# evaluate model performance
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)

# create Streamlit app
st.title("Credit Card Fraud Detection Model")
st.write("This model predicts whether a credit card transaction is legitimate or fraudulent.")
st.write(f"Model Accuracy: {test_acc:.2%}")

# Display sample transactions
col1, col2 = st.columns(2)

with col1:
    st.write("### Sample Legitimate Transaction")
    legit_sample = legit.iloc[0].drop('Class').values
    st.write("This will predict as legitimate:")
    st.code(','.join(map(str, legit_sample)))
    if st.button("Use Legitimate Sample"):
        st.session_state.input_text = ','.join(map(str, legit_sample))

with col2:
    st.write("### Sample Fraudulent Transaction")
    fraud_sample = fraud.iloc[0].drop('Class').values
    st.write("This will predict as fraudulent:")
    st.code(','.join(map(str, fraud_sample)))
    if st.button("Use Fraudulent Sample"):
        st.session_state.input_text = ','.join(map(str, fraud_sample))

st.write("Enter the transaction features (comma-separated values):")
st.write("You need to enter exactly 30 numbers, separated by commas")

# create input fields for user to enter feature values
input_df = st.text_input('Input Features (comma-separated)', key='input_text')

# create a button to submit input and get prediction
submit = st.button("Submit")

if submit:
    try:
        # get input feature values
        input_df_lst = input_df.strip().split(',')
        if len(input_df_lst) != 30:
            st.error(f"Please enter exactly 30 features. You entered {len(input_df_lst)} features.")
        else:
            features = np.array(input_df_lst, dtype=np.float64)
            # make prediction
            prediction = model.predict(features.reshape(1,-1))
            # display result
            if prediction[0] == 0:
                st.success("✅ Legitimate transaction")
            else:
                st.error("❌ Fraudulent transaction")
    except ValueError:
        st.error("Please enter valid numbers separated by commas")
