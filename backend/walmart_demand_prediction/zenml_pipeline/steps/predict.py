import pandas as pd
from sklearn.linear_model import LinearRegression
from zenml import step

@step
def predict(model: LinearRegression, df: pd.DataFrame, n: int = 2) -> pd.DataFrame:
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df.sort_values('Date', inplace=True)

    last_row = df.iloc[-1].copy()
    last_date = last_row['Date']
    prev_sales = last_row['Weekly_Sales']

    predictions = []

    for i in range(1, n + 1):
        next_date = last_date + pd.DateOffset(weeks=1)
        year = next_date.isocalendar().year
        week = next_date.isocalendar().week

        input_data = pd.DataFrame([{
            'Year': year,
            'Week': week,
            'Holiday_Flag': last_row['Holiday_Flag'],
            'Fuel_Price': last_row['Fuel_Price'],
            'CPI': last_row['CPI'],
            'Temperature': last_row['Temperature'],
            'Unemployment': last_row['Unemployment'],
            'Prev_Week_Sales': prev_sales
        }])

        pred = model.predict(input_data)[0]
        predictions.append({
            'Year': year,
            'Week': week,
            'Predicted_Sales': round(pred, 2)
        })

        # Update for next iteration
        prev_sales = pred
        last_date = next_date

    return pd.DataFrame(predictions)
