import json, numpy as np, hashlib, os, sys
def h(p): 
    m=hashlib.sha256(open(p,'rb').read()).hexdigest()[:10] if os.path.exists(p) else 'MISSING'
    t=os.path.getmtime(p) if os.path.exists(p) else 0
    return m, t
for f in ["bins.npy","config.json","model.keras","label_encoder.pkl","class_order.json"]:
    print(f"{f:20s}", h(f))
bins=np.load("bins.npy"); print("bins.shape:", bins.shape, "ppm_min/max:", bins.min(), bins.max())
cfg=json.load(open("config.json")); print("config:", cfg)

