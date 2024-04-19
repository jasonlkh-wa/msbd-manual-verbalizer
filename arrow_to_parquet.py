import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# with open("data/imdb/dataset.arrow", "rb") as f:
#     table = pa.ipc.RecordBatchStreamReader(f).read_all()
#     table: pd.DataFrame = table.to_pandas()
#     table.to_parquet("data/imdb/raw_data.parquet")

df = pd.read_parquet("data/imdb/raw_data.parquet")
print(df[["movie title - year", "genre"]])
df = df[["movie title - year", "genre"]]

df.columns = ["text", "label"]
print(df.head())
print(df["label"].value_counts())
# df.to_parquet("data/imdb/data.parquet")
