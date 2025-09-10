
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# -------------------------
# 1. Generate synthetic dataset
# -------------------------
np.random.seed(42)
n_samples = 200

data = pd.DataFrame({
    'num_people': np.random.randint(1, 7, n_samples),
    'showers_per_day': np.random.randint(1, 4, n_samples),
    'washing_machine_uses': np.random.randint(0, 2, n_samples),
    'dishwasher_uses': np.random.randint(0, 2, n_samples),
    'ac_hours': np.random.randint(0, 10, n_samples),
    'fridge_hours': np.random.randint(18, 24, n_samples),
    'house_area': np.random.randint(50, 200, n_samples),
    'temperature': np.random.randint(20, 40, n_samples)
})

# Targets
data['water_liters'] = (
    data['num_people']*50 +
    data['showers_per_day']*80 +
    data['washing_machine_uses']*70 +
    data['dishwasher_uses']*30 +
    np.random.normal(0, 20, n_samples)
)

data['electricity_kwh'] = (
    data['ac_hours']*1.2 +
    data['fridge_hours']*0.1 +
    data['house_area']*0.05 +
    np.random.normal(0, 2, n_samples)
)

# -------------------------
# 2. Train Linear Regression model
# -------------------------
X = data.drop(['water_liters', 'electricity_kwh'], axis=1)
y = data[['water_liters', 'electricity_kwh']]

model = LinearRegression()
model.fit(X, y)

# -------------------------
# 3. User input for prediction
# -------------------------
print("Enter household details:")
num_people = int(input("Number of people: "))
showers = int(input("Showers per day: "))
washing = int(input("Washing machine uses per day: "))
dishwasher = int(input("Dishwasher uses per day: "))
ac_hours = int(input("AC hours per day: "))
fridge_hours = int(input("Fridge hours per day: "))
house_area = int(input("House area (sqm): "))
temperature = int(input("Temperature (°C): "))

example_home = pd.DataFrame({
    'num_people': [num_people],
    'showers_per_day': [showers],
    'washing_machine_uses': [washing],
    'dishwasher_uses': [dishwasher],
    'ac_hours': [ac_hours],
    'fridge_hours': [fridge_hours],
    'house_area': [house_area],
    'temperature': [temperature]
})

# -------------------------
# 4. Predict
# -------------------------
prediction = model.predict(example_home)
water_liters = prediction[0,0]
electricity_kwh = prediction[0,1]

print(f"\n💧 Predicted Water Usage: {water_liters:.2f} liters/day ({water_liters*1000:.0f} mL/day)")
print(f"⚡ Predicted Electricity Usage: {electricity_kwh:.2f} kWh/day")
