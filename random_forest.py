
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("house_price_clean.csv")

print("Dataset shape:", df.shape)


# ==========================================
# 2. REMOVE ID COLUMN
# ==========================================

df = df.drop(columns=["Property_ID"])


# ==========================================
# 3. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop(columns=["SalePrice_PKR"])
y = df["SalePrice_PKR"]


# ==========================================
# 4. IDENTIFY FEATURE TYPES
# ==========================================

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "bool"]
).columns.tolist()


print("\nNumerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)


# ==========================================
# 5. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))


# ==========================================
# 6. NUMERICAL PREPROCESSING
# ==========================================

numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)


# ==========================================
# 7. CATEGORICAL PREPROCESSING
# ==========================================

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                drop="first"
            )
        )
    ]
)


# ==========================================
# 8. COMBINE PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ==========================================
# 9. RANDOM FOREST MODEL
# ==========================================

model_pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),

        (
            "model",
            RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


# ==========================================
# 10. TRAIN MODEL
# ==========================================

print("\nTraining Random Forest model...")

model_pipeline.fit(X_train, y_train)


# ==========================================
# 11. MAKE PREDICTIONS
# ==========================================

predictions = model_pipeline.predict(X_test)


# ==========================================
# 12. EVALUATE MODEL
# ==========================================

r2 = r2_score(y_test, predictions)

mae = mean_absolute_error(
    y_test,
    predictions
)


print("\n==============================")
print("RANDOM FOREST RESULTS")
print("==============================")

print(f"R2 Score: {r2:.4f}")
print(f"Average Error (PKR): {mae:,.2f}")


# ==========================================
# 13. SAVE MODEL
# ==========================================

joblib.dump(
    model_pipeline,
    "random_forest_pipeline.pkl"
)


print("\n==============================")
print("PIPELINE SAVED SUCCESSFULLY")
print("==============================")

print("File: random_forest_pipeline.pkl")