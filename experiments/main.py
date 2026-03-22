from experiments.run_tfidf_svm import run_tfidf_experiment
from experiments.run_bert_svm import run_bert_experiment

def main():
    print("Running TF-IDF + SVM...")
    run_tfidf_experiment()

    print("\nRunning BERT + SVM...")
    run_bert_experiment()

if __name__ == "__main__":
    main()
    