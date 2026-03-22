from src.preprocessing import load_and_prepare_data
from src.bert_svm_model import BERTSVMModel
from src.evaluation import evaluate_model


def run_bert_experiment():
    train_df, val_df, test_df = load_and_prepare_data(
        "data/raw/train.csv",
        "data/raw/validation.csv",
        "data/raw/test.csv"
    )

    model = BERTSVMModel()
    model.train(train_df["text"], train_df["label"])

    val_preds = model.predict(val_df["text"])
    test_preds = model.predict(test_df["text"])

    evaluate_model(val_df["label"], val_preds, "BERT + SVM (Validation)")
    evaluate_model(test_df["label"], test_preds, "BERT + SVM (Test)")