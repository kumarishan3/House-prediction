import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
#  Loading data from CSV
# -----------------------------
df = pd.read_csv("data.csv")

print("📊 Basic Stats:\n", df.describe(), "\n")

# -----------------------------
# Correlation Analysis
# -----------------------------
print("🔗 Correlation:\n", df.corr(), "\n")

# -----------------------------
#  Visualization
# -----------------------------
plt.figure()
plt.scatter(df['area'], df['price'])
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Area vs Price")
plt.show()

# -----------------------------
#  Features & Target
# -----------------------------
X = df[['area', 'rooms']]
y = df['price']

# -----------------------------
#  Scaling
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
#  Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42
)

# -----------------------------
#  Model Training
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
#  Prediction
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
#  Evaluation
# -----------------------------
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("📉 MAE:", mae)
print("📉 RMSE:", rmse)
print("📊 R² Score:", r2)

# -----------------------------
#  Feature Importance
# -----------------------------
print("📌 Model Coefficients:", model.coef_)

# -----------------------------
#  Custom Prediction
# -----------------------------
sample = scaler.transform([[1000, 3]])
pred_price = model.predict(sample)[0]

print("\n🏠 Predicted price for (1000 sqft, 3 rooms):", round(pred_price))