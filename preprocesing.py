
import pandas as pd

# ============================================================
# 1. LOAD DATASET
# ============================================================

# Change this to your actual CSV file name
df = pd.read_csv("house_prices.csv")

print("Original Shape:")
print(df.shape)

print("\nOriginal Data Types:")
print(df.dtypes)


# ============================================================
# 2. CREATE A COPY
# ============================================================

clean_df = df.copy()


# ============================================================
# 3. CLEAN TEXT / CATEGORICAL COLUMNS
# ============================================================

text_cols = [
    "Location",
    "Property_Type",
    "Parking_Type",
    "Condition",
    "Furnishing",
    "Renovated",
    "Security",
    "Water_Supply",
    "Electricity_Backup",
    "Has_Pool",
    "Has_Garden"
]

for col in text_cols:
    clean_df[col] = clean_df[col].str.strip()


# Normalize Location
clean_df["Location"] = clean_df["Location"].str.title()

# Normalize Furnishing
clean_df["Furnishing"] = clean_df["Furnishing"].str.title()


# ============================================================
# 4. CONVERT NUMERIC COLUMNS
# ============================================================

numeric_cols = [
    "Area_sqft",
    "Lot_Area_sqft",
    "Bedrooms",
    "Garage_Cars",
    "Basement_sqft",
    "Energy_Efficiency_Score"
]

for col in numeric_cols:
    clean_df[col] = pd.to_numeric(
        clean_df[col],
        errors="coerce"
    )


# ============================================================
# 5. HANDLE MISSING NUMERICAL VALUES
# ============================================================

# Area
clean_df["Area_sqft"] = clean_df["Area_sqft"].fillna(
    clean_df["Area_sqft"].median()
)

# Lot Area
clean_df["Lot_Area_sqft"] = clean_df["Lot_Area_sqft"].fillna(
    clean_df["Lot_Area_sqft"].median()
)

# Bedrooms
clean_df["Bedrooms"] = clean_df["Bedrooms"].fillna(
    clean_df["Bedrooms"].median()
)

# Garage Cars
clean_df["Garage_Cars"] = clean_df["Garage_Cars"].fillna(
    clean_df["Garage_Cars"].median()
)

# Basement
# 0 means the property has no basement
clean_df["Basement_sqft"] = clean_df["Basement_sqft"].fillna(0)

# Energy Efficiency
clean_df["Energy_Efficiency_Score"] = clean_df[
    "Energy_Efficiency_Score"
].fillna(
    clean_df["Energy_Efficiency_Score"].median()
)


# ============================================================
# 6. HANDLE MISSING CATEGORICAL VALUES
# ============================================================

clean_df["Parking_Type"] = clean_df[
    "Parking_Type"
].fillna("Unknown")

clean_df["Furnishing"] = clean_df[
    "Furnishing"
].fillna("Unknown")

clean_df["Security"] = clean_df[
    "Security"
].fillna("Unknown")

clean_df["Water_Supply"] = clean_df[
    "Water_Supply"
].fillna("Unknown")


# Electricity Backup
#
# If NaN means the house has NO backup:
clean_df["Electricity_Backup"] = clean_df[
    "Electricity_Backup"
].fillna("None")

# If NaN means information was simply not provided,
# use "Unknown" instead of "None".


# ============================================================
# 7. CHECK FOR DUPLICATE ROWS
# ============================================================

print("\nDuplicate Rows:")
print(clean_df.duplicated().sum())

# Remove completely duplicated rows
clean_df = clean_df.drop_duplicates()


# ============================================================
# 8. CHECK DUPLICATE PROPERTY IDs
# ============================================================

print("\nDuplicate Property IDs:")
print(clean_df["Property_ID"].duplicated().sum())


# ============================================================
# 9. CHECK INVALID NUMERICAL VALUES
# ============================================================

print("\nInvalid Area:")
print(clean_df[clean_df["Area_sqft"] <= 0])

print("\nInvalid Lot Area:")
print(clean_df[clean_df["Lot_Area_sqft"] <= 0])

print("\nInvalid Bedrooms:")
print(clean_df[clean_df["Bedrooms"] <= 0])

print("\nInvalid Bathrooms:")
print(clean_df[clean_df["Bathrooms"] <= 0])

print("\nInvalid Rooms:")
print(clean_df[clean_df["Rooms"] <= 0])

print("\nInvalid Stories:")
print(clean_df[clean_df["Stories"] <= 0])

print("\nInvalid Distance:")
print(clean_df[clean_df["Distance_to_City_km"] < 0])

print("\nInvalid Sale Price:")
print(clean_df[clean_df["SalePrice_PKR"] <= 0])


# ============================================================
# 10. CHECK YEAR BUILT
# ============================================================

print("\nYear Built Summary:")
print(clean_df["Year_Built"].describe())

# Houses should normally have a reasonable construction year.
# We are only checking here — NOT deleting anything automatically.


# ============================================================
# 11. CHECK ENERGY EFFICIENCY RANGE
# ============================================================

print("\nEnergy Efficiency Summary:")
print(clean_df["Energy_Efficiency_Score"].describe())


# ============================================================
# 12. FINAL MISSING VALUE CHECK
# ============================================================

print("\n==============================")
print("FINAL MISSING VALUES")
print("==============================")

print(clean_df.isnull().sum())


# ============================================================
# 13. FINAL DATA TYPES
# ============================================================

print("\n==============================")
print("FINAL DATA TYPES")
print("==============================")

print(clean_df.dtypes)


# ============================================================
# 14. FINAL DATASET INFORMATION
# ============================================================

print("\n==============================")
print("FINAL DATASET SHAPE")
print("==============================")

print(clean_df.shape)


# ============================================================
# 15. FINAL NUMERICAL SUMMARY
# ============================================================

print("\n==============================")
print("NUMERICAL SUMMARY")
print("==============================")

print(clean_df.describe())


# ============================================================
# 16. SAVE CLEAN DATASET
# ============================================================

clean_df.to_csv(
    "house_price_clean.csv",
    index=False
)

print("\n===================================")
print("CLEAN DATASET SAVED SUCCESSFULLY")
print("===================================")

print("File: house_price_clean.csv")
