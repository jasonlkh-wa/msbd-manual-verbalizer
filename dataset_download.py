import os
import pandas as pd
from datasets import load_dataset, list_datasets

root = os.path.abspath(os.path.dirname(__file__))
# Load twitter financial news topic
dataset = load_dataset("zeroshot/twitter-financial-news-topic")
train_path = os.path.join(root, "data/twitter-financial-news-topic/train.parquet")
validation_path = os.path.join(
    root, "data/twitter-financial-news-topic/validation.parquet"
)
dataset["train"].to_parquet(train_path)
dataset["validation"].to_parquet(validation_path)

df_train = pd.read_parquet(train_path)
df_validation = pd.read_parquet(validation_path)
df = pd.concat([df_train, df_validation])
df.to_parquet(os.path.join(root, "data/twitter-financial-news-topic/data.parquet"))
