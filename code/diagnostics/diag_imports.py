# diagnose why webapp wont work, problems with environment etc

import os
os.environ["OMP_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
os.environ["VECLIB_MAXIMUM_THREADS"]="1"

def try_import(name, expr=""):
    print(f"-> importing {name} {expr}")
    __import__(name)
    print(f"OK: {name}")

try_import("numpy")
try_import("scipy")
try_import("sklearn")
try_import("pandas")
try_import("matplotlib")
try_import("joblib")
try_import("tensorflow")   # <- likely crash point on macOS
print("All imports succeeded.")
