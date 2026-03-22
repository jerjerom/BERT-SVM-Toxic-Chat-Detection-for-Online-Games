import pandas as pd


def load_and_prepare_data(train_path, val_path, test_path):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    def clean_df(df):
        df = df.rename(columns={"message": "text", "target": "label"})
        df = df[["text", "label"]].copy()
        df["label"] = df["label"].apply(lambda x: 0 if x == 0 else 1)
        df["text"] = df["text"].astype(str).str.strip()
        df = df.dropna(subset=["text", "label"])
        df = df[df["text"] != ""]
        return df

    return clean_df(train_df), clean_df(val_df), clean_df(test_df)
