# test_latency.py
import pandas as pd
import time
from solution import preprocess, load_model, predict

# Load test data
df = pd.read_csv("../Data/test.csv")

# These run before the clock (same as judge)
df_processed = preprocess(df)
model = load_model()

# Time only predict() — exactly like the judge does
start = time.perf_counter()
predictions = predict(df_processed, model)
duration = time.perf_counter() - start

print(f"Predict time:    {duration:.4f}s")
print(f"Latency penalty: {max(0.5, 1 - duration/10):.4f}")
print(f"Predictions shape: {predictions.shape}")
print(predictions.head())