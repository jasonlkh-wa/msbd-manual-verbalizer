import pandas as pd
import os
from datasets import load_from_disk
from common_utils import printls
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, ManualVerbalizer
from openprompt import PromptForClassification, PromptDataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
from tqdm import tqdm
import argparse


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

    return parser.parse_args()


def read_data(name: str) -> pd.DataFrame:
    """Return the dataframe for the given dataset name."""
    if name == "twitter-financial-news-topic":
        path = os.path.join(ROOT, "data/twitter-financial-news-topic/data.parquet")

    return pd.read_parquet(path)


def load_class_map(name: str) -> list:
    """Return the class map for the given dataset name."""
    if name == "twitter-financial-news-topic":
        path = os.path.join(ROOT, "data/twitter-financial-news-topic/classes.txt")

    with open(path, "r") as f:
        classes = f.readlines()

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


def get_verbalizer(name: str, classes: list, tokenizer) -> ManualVerbalizer:
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
    )


def model_training(prompt_model, args, train_dataloader):
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
            logits = prompt_model(input)
            labels = input["label"]
            loss = loss_function(logits, labels)
            loss.backward()
            tot_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            if step % 100 == 1:
                print(f"Epoch {epoch}, step {step}, loss {loss.item()}")

    return prompt_model


def main():
    args = load_args_setup()
    global ROOT
    ROOT = os.path.dirname(__file__)

    raw_data = read_data(args.dataset)
    class_map = load_class_map(args.dataset)
    printls(raw_data.head())
    printls("Data shape:\n" + str(raw_data.shape))

    # Get k_shot data
    df_train, df_test = get_data_split(
        raw_data, k_shot=args.k_shot, class_map=class_map
    )
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

    manaul_template = get_manual_template(args.dataset, tokenizer)
    manual_verbalizer = get_verbalizer(args.dataset, class_map.values(), tokenizer)

    # Data loader
    train_dataloader = get_dataloader(
        df_train, manaul_template, tokenizer, WrapperClass
    )

    prompt_model = PromptForClassification(
        template=manaul_template,
        plm=plm,
        verbalizer=manual_verbalizer,
        freeze_plm=False,
    )
    if torch.cuda.is_available():
        prompt_model = prompt_model.cuda()

    prompt_model = model_training(prompt_model, args, train_dataloader)

    # Evaluate model result
    validation_dataloader = get_dataloader(
        df_test, manaul_template, tokenizer, WrapperClass
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

    acc = sum([int(i == j) for i, j in zip(all_preds, all_labels)]) / len(all_labels)
    print(acc)


if __name__ == "__main__":
    main()
