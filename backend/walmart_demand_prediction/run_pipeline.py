from zenml_pipeline.pipeline import demand_prediction_pipeline

if __name__ == "__main__":
    pipeline = demand_prediction_pipeline()
    pipeline.run()
