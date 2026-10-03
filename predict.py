
import pandas as pd
import joblib


# ==========================================
# 1. LOAD TRAINED MODEL
# ==========================================

model = joblib.load("house_price_pipeline.pkl")

print("Model loaded successfully!")


# ==========================================
# 2. CREATE A NEW HOUSE
# ==========================================

new_house = pd.DataFrame({
    "Area_sqft": [2500],
    "Lot_Area_sqft": [3000],
    "Bedrooms": [4],
    "Bathrooms": [3],
    "Rooms": [8],
    "Stories": [2],
    "Year_Built": [2015],
    "Garage_Cars": [2],
    "Parking_Spaces": [2],
    "Basement_sqft": [1000],
    "Overall_Quality": [8],
    "Distance_to_City_km": [5],
    "Energy_Efficiency_Score": [75],

    # Categorical features
    "Location": ["Peshawar"],
    "Property_Type": ["House"],
    "Parking_Type": ["Covered"],
    "Condition": ["Good"],
    "Furnishing": ["Furnished"],
    "Renovated": ["Yes"],
    "Security": ["24/7"],
    "Water_Supply": ["Municipal"],
    "Electricity_Backup": ["Solar"],
    "Has_Pool": ["No"],
    "Has_Garden": ["Yes"]
})


# ==========================================
# 3. MAKE PREDICTION
# ==========================================

prediction = model.predict(new_house)

predicted_price = prediction[0]


# ==========================================
# 4. DISPLAY RESULT
# ==========================================

print("\n==============================")
print("HOUSE PRICE PREDICTION")
print("==============================")

print(f"Predicted Price: {predicted_price:,.2f} PKR")
