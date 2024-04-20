import pandas as pd
import json

with open("data/news/train.json", "r") as file:
    data = json.load(file)


df = pd.DataFrame(data)
print(df.shape)

df = df[["headline", "section"]]

with open("data/news/classes.txt", "w") as file:
    for i in df["section"].unique():
        file.write(i + "\n")

df.columns = ["text", "label"]

df.to_parquet("data/news/data.parquet")
