from datasets import load_dataset


def load_ag_news():
    print("Loading AG News dataset...")
    try:
        dataset = load_dataset("ag_news")
    except Exception:
        # Fallback for newer huggingface_hub versions requiring namespace format
        dataset = load_dataset("fancyzhx/ag_news")

    train = dataset["train"].to_pandas()
    test = dataset["test"].to_pandas()

    # AG News label map
    label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
    train["label_name"] = train["label"].map(label_map)
    test["label_name"] = test["label"].map(label_map)

    print(f"\nTrain size: {len(train):,}")
    print(f"Test size:  {len(test):,}")

    print("\nClass distribution (train):")
    print(train["label_name"].value_counts())

    print("\nSample rows:")
    print(train[["text", "label_name"]].head(3).to_string())

    return train, test


if __name__ == "__main__":
    train_df, test_df = load_ag_news()
