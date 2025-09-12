import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# 1. Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("household_dataset.csv")

data = load_data()

st.title("🏡 Household Water & Electricity Prediction")
st.write("This app predicts **daily water usage (liters)** and **electricity usage (kWh)** using Linear Regression.")

# Show dataset preview
if st.checkbox("Show Dataset"):
    st.dataframe(data.head())

# -----------------------------
# 2. Train Model
# -----------------------------
X = data.drop(["water_liters", "electricity_kwh"], axis=1)
y = data[["water_liters", "electricity_kwh"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# -----------------------------
# 3. Model Evaluation
# -----------------------------
from sklearn.metrics import mean_squared_error
import numpy as np

rmse_water = np.sqrt(mean_squared_error(y_test["water_liters"], y_pred[:, 0]))
rmse_elec = np.sqrt(mean_squared_error(y_test["electricity_kwh"], y_pred[:, 1]))

r2_water = r2_score(y_test["water_liters"], y_pred[:, 0])
r2_elec = r2_score(y_test["electricity_kwh"], y_pred[:, 1])

st.subheader("📊 Model Performance")
st.write(f"**Water** - RMSE: {rmse_water:.2f}, R²: {r2_water:.3f}")
st.write(f"**Electricity** - RMSE: {rmse_elec:.2f}, R²: {r2_elec:.3f}")

# -----------------------------
# 4. User Input Form
# -----------------------------
st.subheader("🔮 Predict for Your Home")

num_people = st.number_input("Number of People", 1, 10, 3)
showers_per_day = st.number_input("Showers per Day", 1, 5, 2)
washing_machine_uses = st.number_input("Washing Machine Uses", 0, 3, 1)
dishwasher_uses = st.number_input("Dishwasher Uses", 0, 3, 0)
ac_hours = st.number_input("AC Hours per Day", 0, 24, 6)
fridge_hours = st.number_input("Fridge Hours per Day", 18, 24, 22)
house_area = st.number_input("House Area (sq. meters)", 30, 500, 120)
temperature = st.number_input("Temperature (°C)", 15, 45, 32)

if st.button("Predict"):
    input_data = pd.DataFrame({
        "num_people": [num_people],
        "showers_per_day": [showers_per_day],
        "washing_machine_uses": [washing_machine_uses],
        "dishwasher_uses": [dishwasher_uses],
        "ac_hours": [ac_hours],
        "fridge_hours": [fridge_hours],
        "house_area": [house_area],
        "temperature": [temperature]
    })

    prediction = model.predict(input_data)

    st.success(f"💧 Water Usage: {prediction[0,0]:.2f} liters/day")
    st.success(f"⚡ Electricity Usage: {prediction[0,1]:.2f} kWh/day")

# -----------------------------
# 5. Extra Info
# -----------------------------
st.caption("Built with Streamlit, Scikit-learn, and Linear Regression.")
