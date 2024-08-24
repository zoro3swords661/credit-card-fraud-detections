import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
@st.cache_data
def load_data():
    data = pd.read_csv('creditcard.csv')
    return data

data = load_data()

# Data preprocessing
def preprocess_data(data):
    features = data.drop(columns=['Time', 'Amount', 'Class'])
    labels = data['Class']

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(
        scaled_features, labels, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = preprocess_data(data)

# Train the model
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, 'fraud_detection_model.pkl')
    return model

# Re-train and save the model
model = train_model(X_train, y_train)

def predict(features):
    model = joblib.load('fraud_detection_model.pkl')
    prediction = model.predict([features])
    return prediction[0]

# Streamlit app
st.title('Credit Card Fraud Detection')

# Initialize session state for feature inputs
for i in range(1, 29):
    if f'V{i}' not in st.session_state:
        st.session_state[f'V{i}'] = 0.0

# Input features arranged in rows
feature_inputs = []
cols = st.columns(4)
for i in range(1, 29):
    with cols[(i-1) % 4]:
        feature_inputs.append(st.number_input(f'V{i}', value=st.session_state[f'V{i}'], key=f'V{i}'))

# Button to train the model
if st.button('Train Model'):
    model = train_model(X_train, y_train)
    st.write('Model trained and saved successfully.')

if st.button('Predict'):
    prediction = predict([st.session_state[f'V{i}'] for i in range(1, 29)])
    if prediction == 1:
        st.write('This transaction is fraudulent.')
    else:
        st.write('This transaction is not fraudulent.')

# Visualization state variables
if 'show_real' not in st.session_state:
    st.session_state['show_real'] = False
if 'show_fraud' not in st.session_state:
    st.session_state['show_fraud'] = False
if 'show_corr' not in st.session_state:
    st.session_state['show_corr'] = False
if 'show_class_dist' not in st.session_state:
    st.session_state['show_class_dist'] = False

# Toggle visualization states
cols = st.columns(4)
with cols[0]:
    if st.button('Show Real Transactions'):
        st.session_state['show_real'] = not st.session_state['show_real']
        st.session_state['show_fraud'] = False
        st.session_state['show_corr'] = False
        st.session_state['show_class_dist'] = False
with cols[1]:
    if st.button('Show Fraudulent Transactions'):
        st.session_state['show_fraud'] = not st.session_state['show_fraud']
        st.session_state['show_real'] = False
        st.session_state['show_corr'] = False
        st.session_state['show_class_dist'] = False
with cols[2]:
    if st.button('Show Correlation Matrix'):
        st.session_state['show_corr'] = not st.session_state['show_corr']
        st.session_state['show_real'] = False
        st.session_state['show_fraud'] = False
        st.session_state['show_class_dist'] = False
with cols[3]:
    if st.button('Show Class Distribution'):
        st.session_state['show_class_dist'] = not st.session_state['show_class_dist']
        st.session_state['show_real'] = False
        st.session_state['show_fraud'] = False
        st.session_state['show_corr'] = False

# Render visualizations based on state
if st.session_state['show_real']:
    real_data = data[data['Class'] == 0]
    st.write(real_data.describe())
    fig, ax = plt.subplots()
    sns.histplot(real_data['Amount'], kde=True, ax=ax)
    st.pyplot(fig)

if st.session_state['show_fraud']:
    fraud_data = data[data['Class'] == 1]
    st.write(fraud_data.describe())
    fig, ax = plt.subplots()
    sns.histplot(fraud_data['Amount'], kde=True, ax=ax)
    st.pyplot(fig)

if st.session_state['show_corr']:
    corr_matrix = data.corr()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', ax=ax)
    st.pyplot(fig)

if st.session_state['show_class_dist']:
    class_counts = data['Class'].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=class_counts.index, y=class_counts.values, ax=ax)
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    st.pyplot(fig)