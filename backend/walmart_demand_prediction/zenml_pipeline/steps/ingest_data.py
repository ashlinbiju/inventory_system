import pandas as pd
from zenml import step

@step
def ingest_data() -> pd.DataFrame:
    df = pd.read_csv("data/walmart_sales.csv", parse_dates=['Date'])
    df = df.sort_values("Date")
    return df
