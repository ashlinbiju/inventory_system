import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os
from zenml import step

@step
def train_model(df: pd.DataFrame) -> LinearRegression:
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df.sort_values('Date', inplace=True)
    df['Week'] = df['Date'].dt.isocalendar().week
    df['Year'] = df['Date'].dt.year

    # ✅ Add Prev_Week_Sales
    df['Prev_Week_Sales'] = df['Weekly_Sales'].shift(1)
    df.dropna(inplace=True)

    features = ['Year', 'Week', 'Holiday_Flag', 'Fuel_Price', 'CPI', 'Temperature', 'Unemployment', 'Prev_Week_Sales']
    target = 'Weekly_Sales'

    X = df[features]
    y = df[target]

    model = LinearRegression()
    model.fit(X, y)

    # Save model to disk
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/sales_model.pkl")
    print("✅ Model saved at models/sales_model.pkl")

    return model
