"""
QSkill Internship - Task 1 (Slab 1, Beginner)
Load a CSV with Pandas, perform basic analysis, and visualize with Matplotlib.
Dataset: sales_data.csv (Region, Product, UnitsSold, UnitPrice, Revenue)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("sales_data.csv")

print("=" * 60)
print("DATA PREVIEW")
print("=" * 60)
print(df.head())
print(f"\nShape: {df.shape[0]} rows, {df.shape[1]} columns")

print("\n" + "=" * 60)
print("BASIC ANALYSIS")
print("=" * 60)

avg_revenue = df["Revenue"].mean()
avg_units = df["UnitsSold"].mean()
avg_price = df["UnitPrice"].mean()

print(f"Average Revenue per order : ${avg_revenue:,.2f}")
print(f"Average Units Sold        : {avg_units:.1f}")
print(f"Average Unit Price        : ${avg_price:.2f}")
print(f"Total Revenue             : ${df['Revenue'].sum():,.2f}")
print(f"Highest single order      : ${df['Revenue'].max():,.2f}")
print(f"Lowest single order       : ${df['Revenue'].min():,.2f}")

revenue_by_region = df.groupby("Region")["Revenue"].sum().sort_values(ascending=False)
revenue_by_product = df.groupby("Product")["Revenue"].sum().sort_values(ascending=False)

print("\nRevenue by Region:")
print(revenue_by_region)
print("\nRevenue by Product:")
print(revenue_by_product)

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle("Sales Data Analysis Dashboard", fontsize=16, fontweight="bold")

axes[0, 0].bar(revenue_by_region.index, revenue_by_region.values, color="#4C72B0")
axes[0, 0].set_title("Total Revenue by Region")
axes[0, 0].set_xlabel("Region")
axes[0, 0].set_ylabel("Revenue ($)")

axes[0, 1].bar(revenue_by_product.index, revenue_by_product.values, color="#55A868")
axes[0, 1].set_title("Total Revenue by Product")
axes[0, 1].set_xlabel("Product")
axes[0, 1].set_ylabel("Revenue ($)")
axes[0, 1].tick_params(axis="x", rotation=20)

sc = axes[1, 0].scatter(
    df["UnitsSold"], df["UnitPrice"], c=df["Revenue"], cmap="viridis", alpha=0.8
)
axes[1, 0].set_title("Units Sold vs Unit Price")
axes[1, 0].set_xlabel("Units Sold")
axes[1, 0].set_ylabel("Unit Price ($)")
fig.colorbar(sc, ax=axes[1, 0], label="Revenue ($)")

pivot = df.pivot_table(values="Revenue", index="Region", columns="Product", aggfunc="mean")
im = axes[1, 1].imshow(pivot.values, cmap="YlOrRd", aspect="auto")
axes[1, 1].set_xticks(range(len(pivot.columns)))
axes[1, 1].set_xticklabels(pivot.columns, rotation=20)
axes[1, 1].set_yticks(range(len(pivot.index)))
axes[1, 1].set_yticklabels(pivot.index)
axes[1, 1].set_title("Avg Revenue Heatmap (Region x Product)")

for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        axes[1, 1].text(
            j, i, f"{pivot.values[i, j]:.0f}",
            ha="center", va="center", color="black", fontsize=8
        )

fig.colorbar(im, ax=axes[1, 1], label="Avg Revenue ($)")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("task1_dashboard.png", dpi=150)
print("\nSaved visualization to task1_dashboard.png")

print("\n" + "=" * 60)
print("INSIGHTS & OBSERVATIONS")
print("=" * 60)

top_region = revenue_by_region.index[0]
top_product = revenue_by_product.index[0]
corr = df["UnitsSold"].corr(df["Revenue"])

print(f"1. '{top_region}' region generates the highest total revenue "
      f"(${revenue_by_region.iloc[0]:,.2f}).")
print(f"2. '{top_product}' is the top-performing product by revenue "
      f"(${revenue_by_product.iloc[0]:,.2f}).")
print(f"3. Units Sold and Revenue have a correlation of {corr:.2f}, "
      f"indicating {'a strong' if corr > 0.5 else 'a moderate'} positive relationship.")

best_cell = pivot.stack().idxmax()
print(f"4. The best Region-Product combination by average revenue is "
      f"{best_cell[0]} / {best_cell[1]} (${pivot.stack().max():,.2f} avg).")
