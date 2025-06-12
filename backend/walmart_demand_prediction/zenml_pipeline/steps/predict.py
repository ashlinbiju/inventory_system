from zenml import step
from sklearn.linear_model import LinearRegression
import pandas as pd
from datetime import datetime, timedelta

@step
def predict(model: LinearRegression, df: pd.DataFrame) -> pd.DataFrame:
    df['Week'] = df['Date'].dt.isocalendar().week
    df['Year'] = df['Date'].dt.year
    df_grouped = df.groupby(['Year', 'Week'])['Weekly_Sales'].sum().reset_index()
    
    # Get latest year and week
    last_row = df_grouped.iloc[-1]
    year, week = int(last_row['Year']), int(last_row['Week'])

    predictions = []
    for _ in range(2):  # Predict next 2 weeks
        week += 1
        if week > 52:
            week = 1
            year += 1
        pred = model.predict([[year, week]])[0]
        predictions.append({'Year': year, 'Week': week, 'Predicted_Sales': pred})

    return pd.DataFrame(predictions)
