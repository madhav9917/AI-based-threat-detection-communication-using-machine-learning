import joblib
import pandas as pd

# Load model
data = joblib.load("model/model.pkl")
model = data["model"]

print("✅ Model loaded")

# -------------------------------
# TEST NSL-LIKE DATA
# -------------------------------
nsl_sample = pd.DataFrame([{
    "length": 100,
    "proto": 0,
    "src_port": 0,
    "dst_port": 0,
    "tcp": 0,
    "udp": 0,
    "icmp": 0
}])

pred_nsl = model.predict(nsl_sample)[0]
print("NSL Test Prediction:", pred_nsl)

# -------------------------------
# TEST UNSW-LIKE DATA
# -------------------------------
unsw_sample = pd.DataFrame([{
    "length": 1500,
    "proto": 1,
    "src_port": 0,
    "dst_port": 0,
    "tcp": 1,
    "udp": 0,
    "icmp": 0
}])

pred_unsw = model.predict(unsw_sample)[0]
print("UNSW Test Prediction:", pred_unsw)