# recreate_label_encoder.py
import json, joblib
from sklearn.preprocessing import LabelEncoder

# Load class order list
with open("class_order.json") as f:
    classes = json.load(f)
print("Classes found:", classes)

# Fit and save encoder
le = LabelEncoder()
le.fit(classes)
joblib.dump(le, "label_encoder.pkl")
print("Saved label_encoder.pkl ✅")
