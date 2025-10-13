import json, pandas as pd
# adjust the path/column name if needed; the column should be the 5-class label you trained on
df = pd.read_csv("processed_data/hmdb_subset_super_groups.csv")  # has a 'group' column with your 5 classes
classes = sorted(df["group"].dropna().unique().tolist())
print("Alphabetical class order inferred from training labels:")
for i,c in enumerate(classes): print(i, c)
json.dump(classes, open("class_order.json","w"), indent=2, ensure_ascii=False)
print("\nWrote class_order.json")
