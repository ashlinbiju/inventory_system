from flask import Blueprint, render_template
import pandas as pd

main = Blueprint("main", __name__)

@main.route("/")
def home():
    return render_template("index.html")

@main.route("/demand-prediction")
def demand_prediction():
    df = pd.read_csv("data/walmart_sales.csv", parse_dates=['Date'])
    current_week = df['Date'].dt.isocalendar().week.max()
    current_year = df['Date'].dt.year.max()

    # Dummy loading for now - in real case, this should read ZenML prediction outputs
    predictions = pd.DataFrame({
        'Year': [current_year]*2,
        'Week': [current_week + 1, current_week + 2],
        'Predicted_Sales': [15000, 16000]
    })

    current_sales = df[df['Date'].dt.isocalendar().week == current_week]['Weekly_Sales'].sum()

    return render_template("demand_prediction.html", current_sales=current_sales, predictions=predictions.to_dict(orient="records"))
