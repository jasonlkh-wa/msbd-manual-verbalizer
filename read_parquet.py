import pandas as pd

print(pd.read_parquet("soft_news_8_15.parquet").pred.value_counts())
