import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("house_price_clean.csv")

print("Dataset shape:", df.shape)


# ============================================================
# 2. REMOVE ID COLUMN
# ============================================================

df = df.drop(columns=["Property_ID"])


# ============================================================
# 3. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=["SalePrice_PKR"])
y = df["SalePrice_PKR"]


# ============================================================
# 4. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numerical_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "string", "bool"]
).columns.tolist()


print("\nNumerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))


# ============================================================
# 6. NUMERICAL PREPROCESSING
# ============================================================

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


# ============================================================
# 7. CATEGORICAL PREPROCESSING
# ============================================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                drop="first"
            )
        )
    ]
)


# ============================================================
# 8. COMBINE PREPROCESSING
# ============================================================

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


# ============================================================
# 9. RANDOM FOREST MODEL
# ============================================================

random_forest = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 10. COMPLETE PIPELINE
# ============================================================

model_pipeline = Pipeline(
    steps=[
        (
            "preprocessing",
            preprocessor
        ),
        (
            "model",
            random_forest
        )
    ]
)


# ============================================================
# 11. PARAMETERS TO SEARCH
# ============================================================

param_distributions = {

    "model__n_estimators": [
        100,
        200,
        300,
        400
    ],

    "model__max_depth": [
        None,
        10,
        20,
        30,
        40
    ],

    "model__min_samples_split": [
        2,
        5,
        10
    ],

    "model__min_samples_leaf": [
        1,
        2,
        4
    ],

    "model__max_features": [
        "sqrt",
        "log2",
        1.0
    ]
}


# ============================================================
# 12. RANDOMIZED SEARCH
# ============================================================

random_search = RandomizedSearchCV(
    estimator=model_pipeline,
    param_distributions=param_distributions,
    n_iter=15,
    cv=3,
    scoring="neg_mean_absolute_error",
    random_state=42,
    n_jobs=-1,
    verbose=1
)


# ============================================================
# 13. TRAIN / TUNE MODEL
# ============================================================

print("\n==============================")
print("STARTING HYPERPARAMETER TUNING")
print("==============================")

print("\nThis may take some time...\n")

random_search.fit(
    X_train,
    y_train
)


# ============================================================
# 14. BEST PARAMETERS
# ============================================================

print("\n==============================")
print("BEST PARAMETERS")
print("==============================")

print(
    random_search.best_params_
)


# ============================================================
# 15. BEST CROSS-VALIDATION SCORE
# ============================================================

best_cv_mae = -random_search.best_score_

print("\nBest Cross-Validation MAE:")
print(f"{best_cv_mae:,.2f} PKR")


# ============================================================
# 16. FINAL TEST PREDICTIONS
# ============================================================

best_model = random_search.best_estimator_

predictions = best_model.predict(
    X_test
)


# ============================================================
# 17. EVALUATE MODEL
# ============================================================

r2 = r2_score(
    y_test,
    predictions
)

mae = mean_absolute_error(
    y_test,
    predictions
)


# ============================================================
# 18. DISPLAY RESULTS
# ============================================================

print("\n==============================")
print("TUNED RANDOM FOREST RESULTS")
print("==============================")

print(f"R2 Score: {r2:.4f}")

print(
    f"Average Error (PKR): {mae:,.2f}"
)


# ============================================================
# 19. SAVE BEST MODEL
# ============================================================

joblib.dump(
    best_model,
    "random_forest_tuned_pipeline.pkl"
)


print("\n==============================")
print("MODEL SAVED SUCCESSFULLY")
print("==============================")

print(
    "File: random_forest_tuned_pipeline.pkl"
)
# ============================================================
# ACTUAL VS PREDICTED PRICES
# ============================================================

comparison = pd.DataFrame({
    "Actual_Price": y_test.values,
    "Predicted_Price": predictions
})

comparison["Error"] = (
    comparison["Actual_Price"]
    - comparison["Predicted_Price"]
)

comparison["Absolute_Error"] = (
    comparison["Error"].abs()
)

print("\n==============================")
print("ACTUAL VS PREDICTED")
print("==============================")

print(
    comparison.head(20).to_string(index=False)
)

print("\nAverage Absolute Error:")
print(
    f"{comparison['Absolute_Error'].mean():,.2f} PKR"
)

print("\nLargest Errors:")
print(
    comparison
    .sort_values("Absolute_Error", ascending=False)
    .head(10)
    .to_string(index=False)
)