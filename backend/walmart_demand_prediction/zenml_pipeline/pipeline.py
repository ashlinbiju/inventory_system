from zenml import pipeline
from zenml_pipeline.steps.ingest_data import ingest_data
from zenml_pipeline.steps.train_model import train_model
from zenml_pipeline.steps.predict import predict

@pipeline
def demand_prediction_pipeline():
    df = ingest_data()
    model = train_model(df=df)
    predict(model=model, df=df)
