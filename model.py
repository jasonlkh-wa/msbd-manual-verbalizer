import pandas as pd
import os
from datasets import load_from_disk
from common_utils import printls
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import (
    ManualTemplate,
    ManualVerbalizer,
    SoftVerbalizer,
    AutomaticVerbalizer,
)
from openprompt import PromptForClassification, PromptDataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
from tqdm import tqdm
import argparse

if torch.cuda.is_available():
    DEFAULT_DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEFAULT_DEVICE = "mps"
else:
    DEFAULT_DEVICE = "cpu"


def load_args_setup():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="twitter-financial-news-topic",
        help="dataset name",
    )

    parser.add_argument("--k-shot", type=int, default=2)
    parser.add_argument("--epoch", type=int, default=3)

    parser.add_argument(
        "--verbalizer", type=str, default="manual", choices=["manual", "soft", "avs"]
    )

    return parser.parse_args()


news_revert_map = {
    "Art & Design": "Arts & Culture",
    "Theater": "Arts & Culture",
    "Music": "Arts & Culture",
    "Fashion & Style": "Lifestyle & Wellness",
    "Well": "Lifestyle & Wellness",
    "Food": "Lifestyle & Wellness",
    "Science": "Science & Education",
    "Education": "Science & Education",
    "Books": "Science & Education",
    "Television": "Media & Entertainment",
    "Dance": "Media & Entertainment",
    "Movies": "Media & Entertainment",
    "Real Estate": "Business & Economy",
    "Economy": "Business & Economy",
    "Global Business": "Business & Economy",
    "Automobiles": "Technology & Innovation",
    "Media": "Technology & Innovation",
    "Technology": "Technology & Innovation",
    "Style": "Social & Opinion",
    "Opinion": "Social & Opinion",
    "Your Money": "Social & Opinion",
    "Health": "Sports & Leisure",
    "Sports": "Sports & Leisure",
    "Travel": "Sports & Leisure",
}


def read_data(name: str, class_map) -> pd.DataFrame:
    """Return the dataframe for the given dataset name."""
    if name == "twitter-financial-news-topic":
        path = os.path.join(ROOT, "data/twitter-financial-news-topic/data.parquet")
        return pd.read_parquet(path)
    elif name == "imdb":
        path = os.path.join(ROOT, "data/imdb/data.parquet")
        class_map = {v: k for k, v in class_map.items()}
        df = pd.read_parquet(path)
        df["label"] = df["label"].map(lambda x: class_map[x])
        return df
    elif name == "news":
        path = os.path.join(ROOT, "data/news/data.parquet")
        class_map = {v: k for k, v in class_map.items()}
        df = pd.read_parquet(path)
        df["label"] = df["label"].map(lambda x: news_revert_map[x])
        df["label"] = df["label"].map(lambda x: class_map[x])
        return df


def load_class_map(name: str) -> list:
    """Return the class map for the given dataset name."""
    if name == "twitter-financial-news-topic":
        path = os.path.join(ROOT, "data/twitter-financial-news-topic/classes.txt")
    if name == "imdb":
        path = os.path.join(ROOT, "data/imdb/classes.txt")
    if name == "news":
        path = os.path.join(ROOT, "data/news/classes.txt")

    with open(path, "r") as f:
        classes = f.readlines()

    if name == "twitter-financial-news-topic" or name == "imdb":
        return {i: c.strip() for i, c in enumerate(classes)}
    elif name == "news":
        return {i: c.strip() for i, c in enumerate(classes)}


def get_data_split(data, k_shot, class_map) -> tuple:
    """Return the train and test data for the given dataset."""

    def check_k_shot_fulfilled(class_counter, k_shot):
        for cnt in class_counter.values():
            if cnt < k_shot:
                return False
        return True

    class_counter = {c: 0 for c in class_map.keys()}
    train_df = pd.DataFrame(columns=data.columns)
    test_df = pd.DataFrame(columns=data.columns)

    i = 0
    while not check_k_shot_fulfilled(class_counter, k_shot):
        row = data.iloc[i]
        if class_counter[row["label"]] < k_shot:
            class_counter[row["label"]] += 1
            train_df.loc[len(train_df)] = row
        else:
            test_df.loc[len(test_df)] = row
        i += 1

    test_df = pd.concat([test_df, data.iloc[i:]])
    return train_df, test_df


def get_manual_template(name: str, tokenizer) -> ManualTemplate:
    """Return the manual template for the given dataset name."""
    if name == "twitter-financial-news-topic":
        return ManualTemplate(
            text='A {"mask"} news: {"placeholder":"text_a"}', tokenizer=tokenizer
        )
    if name == "imdb":
        return ManualTemplate(
            text='A {"mask"} review: {"placeholder":"text_a"}', tokenizer=tokenizer
        )
    if name == "news":
        return ManualTemplate(
            text='A {"mask"} news: {"placeholder":"text_a"}', tokenizer=tokenizer
        )


def get_manual_verbalizer(name: str, classes: list, tokenizer) -> ManualVerbalizer:
    """Return the verbalizer for the given dataset name."""
    if name == "twitter-financial-news-topic":
        return ManualVerbalizer(
            classes=classes,
            label_words={
                "Analyst Update": ["analyst update"],
                "Fed | Central Banks": ["fed"],
                "Company | Product News": ["company"],
                "Treasuries | Corporate Debt": ["treasury"],
                "Dividend": ["dividend"],
                "Earnings": ["earnings"],
                "Energy | Oil": ["energy"],
                "Financials": ["financial"],
                "Currencies": ["currency"],
                "General News | Opinion": ["general"],
                "Gold | Metals | Materials": ["gold"],
                "IPO": ["ipo"],
                "Legal | Regulation": ["legal"],
                "M&A | Investments": ["m&a"],
                "Macro": ["macro"],
                "Markets": ["market"],
                "Politics": ["politics"],
                "Personnel Change": ["personnel change"],
                "Stock Commentary": ["stock commentary"],
                "Stock Movement": ["stock movement"],
            },
            tokenizer=tokenizer,
        )


def get_soft_verbalizer(classes: list, tokenizer, plm) -> SoftVerbalizer:
    """Return the soft verbalizer for the given dataset name."""
    return SoftVerbalizer(
        tokenizer=tokenizer,
        model=plm,
        classes=classes,
    )


def get_avs_verbalizer(classes: list, tokenizer, plm) -> AutomaticVerbalizer:
    """Return the avs for the given dataset name."""
    return AutomaticVerbalizer(
        tokenizer=tokenizer,
        model=plm,
        classes=classes,
    )


def get_dataloader(df, template, tokenizer, WrapperClass):
    return PromptDataLoader(
        dataset=df,
        template=template,
        tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass,
        max_seq_length=256,
        decoder_max_length=3,
        batch_size=8,
        shuffle=False,
        teacher_forcing=False,
        predict_eos_token=False,
        truncate_method="head",
        DEFAULT_DEVICE=DEFAULT_DEVICE,
    )


def manual_verb_model_training(prompt_model, args, train_dataloader):
    # Training
    loss_function = torch.nn.CrossEntropyLoss()
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in prompt_model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in prompt_model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4)

    for epoch in range(args.epoch):
        tot_loss = 0
        for step, input in enumerate(train_dataloader):
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            logits = prompt_model(input, DEFAULT_DEVICE=DEFAULT_DEVICE)
            labels = input["label"]
            loss = loss_function(logits, labels)
            loss.backward()
            tot_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            if step % 100 == 1:
                print(f"Epoch {epoch}, step {step}, loss {loss.item()}")

    return prompt_model


def soft_verb_model_training(prompt_model, args, train_dataloader):
    # Training
    loss_func = torch.nn.CrossEntropyLoss()

    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters1 = [
        {
            "params": [
                p
                for n, p in prompt_model.plm.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in prompt_model.plm.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer_grouped_parameters2 = [
        {"params": prompt_model.verbalizer.group_parameters_1, "lr": 3e-5},
        {"params": prompt_model.verbalizer.group_parameters_2, "lr": 3e-4},
    ]

    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=3e-5)
    optimizer2 = AdamW(optimizer_grouped_parameters2)

    for epoch in range(args.epoch):
        tot_loss = 0
        for step, inputs in enumerate(train_dataloader):
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            logits = prompt_model(inputs)
            labels = inputs["label"]
            loss = loss_func(logits, labels)
            loss.backward()
            tot_loss += loss.item()
            optimizer1.step()
            optimizer1.zero_grad()
            optimizer2.step()
            optimizer2.zero_grad()
            print(tot_loss / (step + 1))
    return prompt_model


def avs_verb_model_training(verbalizer, prompt_model, args, train_dataloader):
    # Training
    with torch.no_grad():
        for batch in tqdm(train_dataloader, desc="Init_using_{}".format("train")):
            batch = batch.cuda()
            logits = prompt_model(batch)
        verbalizer.optimize_to_initialize()
    loss_func = torch.nn.CrossEntropyLoss()

    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters1 = [
        {
            "params": [
                p
                for n, p in prompt_model.plm.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in prompt_model.plm.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=3e-5)

    for epoch in range(args.epoch):
        tot_loss = 0
        for step, inputs in enumerate(train_dataloader):
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            logits = prompt_model(inputs)
            labels = inputs["label"]
            loss = loss_func(logits, labels)
            loss.backward()
            tot_loss += loss.item()
            optimizer1.step()
            optimizer1.zero_grad()
            print(tot_loss / (step + 1))
    return prompt_model


def main():
    args = load_args_setup()
    global ROOT
    ROOT = os.path.dirname(__file__)

    class_map = load_class_map(args.dataset)
    raw_data = read_data(args.dataset, class_map)
    printls(raw_data.head())
    printls("Data shape:\n" + str(raw_data.shape))

    # Get k_shot data
    df_train, df_test = get_data_split(
        raw_data, k_shot=args.k_shot, class_map=class_map
    )
    df_test_raw = df_test  # as a copy for output

    printls(f"{df_train.shape=}, {df_test.shape=}")

    # Convert data to openpromopt InputExample type
    convert_to_input_example = lambda data: [
        InputExample(text_a=row["text"], label=int(row["label"]))
        for _, row in data.iterrows()
    ]
    df_train = convert_to_input_example(df_train)
    df_test = convert_to_input_example(df_test)

    # Load PLM, template and verbalizer
    plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-uncased")

    manual_template = get_manual_template(args.dataset, tokenizer)
    # Data loader
    train_dataloader = get_dataloader(
        df_train, manual_template, tokenizer, WrapperClass
    )

    # Load verbalizer
    if args.verbalizer == "manual":
        verbalizer = get_manual_verbalizer(args.dataset, class_map.values(), tokenizer)
    elif args.verbalizer == "soft":
        verbalizer = get_soft_verbalizer(class_map.values(), tokenizer, plm)
    elif args.verbalizer == "avs":
        verbalizer = get_avs_verbalizer(class_map.values(), tokenizer, plm)

    prompt_model = PromptForClassification(
        template=manual_template,
        plm=plm,
        verbalizer=verbalizer,
        freeze_plm=False,
    )
    if torch.cuda.is_available():
        prompt_model = prompt_model.cuda()

    if args.verbalizer == "verbalizer":
        prompt_model = manual_verb_model_training(prompt_model, args, train_dataloader)
    elif args.verbalizer == "soft":
        prompt_model = soft_verb_model_training(prompt_model, args, train_dataloader)
    elif args.verbalizer == "avs":
        prompt_model = avs_verb_model_training(
            verbalizer,prompt_model, args, train_dataloader
        )

    # Evaluate model result
    validation_dataloader = get_dataloader(
        df_test, manual_template, tokenizer, WrapperClass
    )

    all_preds = []
    all_labels = []
    for step, inputs in tqdm(enumerate(validation_dataloader)):
        if torch.cuda.is_available():
            inputs = inputs.cuda()

        logits = prompt_model(inputs)
        labels = inputs["label"]
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

    val_accuracy = sum([int(i == j) for i, j in zip(all_preds, all_labels)]) / len(
        all_labels
    )
    print(val_accuracy)

    # Output test accuracy and labels
    # file naming would be [dataset]_[k_shot]_[epoch].parquet
    label_output_path = os.path.join(
        ROOT,
        "output",
        f"{args.verbalizer}_{args.dataset}_{args.k_shot}_{args.epoch}.parquet",
    )
    accuracy_output_path = os.path.join(
        ROOT,
        "output",
        f"{args.verbalizer}_{args.dataset}_{args.k_shot}_{args.epoch}_accuracy.txt",
    )
    with open(accuracy_output_path, "w") as f:
        f.write(str(val_accuracy))

    output = pd.concat(
        [
            df_test_raw.reset_index(drop=True),
            pd.Series(all_labels, name="label_"),
            pd.Series(all_preds, name="pred"),
        ],
        axis=1,
        join="outer",
    )
    output.to_parquet(label_output_path)


if __name__ == "__main__":
    main()
