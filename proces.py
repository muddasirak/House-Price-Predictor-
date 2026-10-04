import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("house_price_clean.csv")

plt.figure(figsize=(10, 6))

plt.scatter(
    df["Area_sqft"],
    df["SalePrice_PKR"],
    alpha=0.3
)

plt.xlabel("Area (sqft)")
plt.ylabel("Sale Price (PKR)")
plt.title("Area vs House Price")

plt.show()