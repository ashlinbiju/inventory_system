from flask import Blueprint, render_template, request
import pandas as pd
import joblib
from datetime import datetime
import sys
import os

# Ensure parent directory is in the path so 'steps' can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from zenml_pipeline.steps.predict import predict  # Import the ZenML predict step

routes = Blueprint('routes', __name__)

@routes.route('/')
def home():
    return render_template('index.html')

@routes.route('/demand-prediction', methods=['GET', 'POST'])

def demand_prediction():
    # Load data and model
    df = pd.read_csv('data/walmart_sales.csv', parse_dates=['Date'], dayfirst=True)
    model = joblib.load('models/sales_model.pkl')

    # Default number of weeks to predict
    n = 2
    if request.method == 'POST':
        try:
            n = int(request.form['weeks'])
        except ValueError:
            n = 2

    # Get current week's sales
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    current_date = df['Date'].max()
    current_week = current_date.isocalendar().week
    current_year = current_date.year
    current_sales = df[(df['Date'].dt.isocalendar().week == current_week) &
                       (df['Date'].dt.year == current_year)]['Weekly_Sales'].sum()

    # Get predictions using ZenML step
    predictions_df = predict(model=model, df=df, n=n)

    return render_template('demand_prediction.html',
                           current_sales=round(current_sales, 2),
                           predictions=predictions_df.to_dict(orient='records'))
