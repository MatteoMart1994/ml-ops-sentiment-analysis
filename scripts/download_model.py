"""Pre-download Hugging Face weights during Docker build (cache-friendly layer)."""

from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def main() -> None:
    AutoTokenizer.from_pretrained(MODEL_ID)
    AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    print(f"Cached {MODEL_ID}")


if __name__ == "__main__":
    main()
