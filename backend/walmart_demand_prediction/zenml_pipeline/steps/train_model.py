from zenml import step
from sklearn.linear_model import LinearRegression
import pandas as pd

@step
def train_model(df: pd.DataFrame) -> LinearRegression:
    df['Week'] = df['Date'].dt.isocalendar().week
    df['Year'] = df['Date'].dt.year
    df_grouped = df.groupby(['Year', 'Week'])['Weekly_Sales'].sum().reset_index()
    
    X = df_grouped[['Year', 'Week']]
    y = df_grouped['Weekly_Sales']
    
    model = LinearRegression()
    model.fit(X, y)
    return model
