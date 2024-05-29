import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the dataset
df = pd.read_csv('ilanlarson3.csv')

# Data preprocessing
df['Price'] = df['Price'].astype(str).str.replace('.', '').astype(float)
df = df[(df['Price'] >= 1100000) & (df['Price'] <= 15000000)]

df['Total Square of Meter'] = df['Total Square of Meter'].str.replace('.', '').astype(float)
df = df[(df['Total Square of Meter'] >= 100) & (df['Total Square of Meter'] <= 500)]

df = df[~df['Number of room'].isin([12, 10, 1, 2.5, 9, 5.5, 8, 3.5, 4.5, 7])]
df = df[~df['Floor of home'].isin(['Müstakil', 'Villa Tipi', '10', '-3', '9', '-2', '8', 'Bahçe Dublex'])]

df['Credi Accepting'] = df['Credi Accepting'].replace('Bilinmiyor', 'Evet')

# Label encoding
le = LabelEncoder()

df['City'] = le.fit_transform(df['City'])
city_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

df['Town'] = le.fit_transform(df['Town'])
town_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

df['Neighbourhood'] = le.fit_transform(df['Neighbourhood'])
neighbourhood_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

df['Number of room'] = le.fit_transform(df['Number of room'])
df['Floor of home'] = le.fit_transform(df['Floor of home'])
df['Credi Accepting'] = le.fit_transform(df['Credi Accepting'])
df['Kombi Doğalgaz Heating'] = le.fit_transform(df['Kombi Doğalgaz Heating'])

# Model training
X = df[['Total Square of Meter', 'Number of room', 'Floor of home', 'Town', 'City', 'Neighbourhood', 'Credi Accepting', 'Kombi Doğalgaz Heating', 'Number of floor']]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)
y_pred = np.clip(y_pred, 1_000_000, 15_000_000)

# Streamlit interface
st.title("House Price Prediction")

st.sidebar.header("Enter Home Properties")

total_square_meter = st.sidebar.number_input("Total Square of Meter", min_value=100, max_value=500, value=150)
city = st.sidebar.selectbox("City", list(city_mapping.keys()))
town = st.sidebar.selectbox("Town", list(town_mapping.keys()))
neighbourhood = st.sidebar.selectbox("Neighbourhood", list(neighbourhood_mapping.keys()))
number_of_room = st.sidebar.selectbox("Number of room", df['Number of room'].unique())
floor_of_home = st.sidebar.selectbox("Floor of home", df['Floor of home'].unique())
number_of_floor = st.sidebar.number_input("Number of floor", min_value=1, max_value=50, value=5)
credi_accepting = st.sidebar.selectbox("Credi Accepting (1 for Yes - 0 for No)", df['Credi Accepting'].unique())
kombi_dogalgaz_heating = st.sidebar.selectbox("Kombi Doğalgaz Heating (1 for Yes - 0 for No)", df['Kombi Doğalgaz Heating'].unique())

if st.sidebar.button("Predict"):
    # Convert user inputs to encoded values
    city_encoded = city_mapping[city]
    town_encoded = town_mapping[town]
    neighbourhood_encoded = neighbourhood_mapping[neighbourhood]

    user_data = np.array([total_square_meter, number_of_room, floor_of_home, town_encoded, city_encoded, neighbourhood_encoded, credi_accepting, kombi_dogalgaz_heating, number_of_floor]).reshape(1, -1)
    user_data = scaler.transform(user_data)
    prediction = lr_model.predict(user_data)
    prediction = np.clip(prediction, 1_000_000, 15_000_000)
    st.write(f"Estimated Price: {prediction[0]:,.2f} TL")

st.write("Comparison of Actual and Estimated Prices:")
results = pd.DataFrame({'Actual Price': y_test, 'Estimated Price': y_pred})
st.write(results.head())
