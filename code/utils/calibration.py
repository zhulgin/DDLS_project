# calibration.py
import json, numpy as np

def softmax_from_logits(L, T=1.0):
    z = L / T
    z -= z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)

class CalibratedPredictor:
    def __init__(self, keras_model, temperature_json="models/calibrated_v1/temperature.json"):
        self.model = keras_model
        self.T = float(json.load(open(temperature_json))["T"])

    def predict_proba(self, X):
        p_raw = self.model.predict(X)
        logits = np.log(np.clip(p_raw, 1e-12, 1.0))
        return softmax_from_logits(logits, self.T)

    def predict(self, X, tau=0.60):
        p = self.predict_proba(X)
        top = p.argmax(1)
        conf = p.max(1)
        abstain = conf < tau
        return {"probs": p, "top": top, "conf": conf, "abstain": abstain}
