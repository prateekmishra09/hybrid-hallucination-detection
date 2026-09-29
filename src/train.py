import sys
from pathlib import Path

# Allow importing dataset.py from src
sys.path.append(str(Path(__file__).parent.parent))

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)

from src.dataset import (
    load_raw_qa,
    create_qa_splits,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "microsoft/deberta-v3-base"

DATA_PATH = Path("src/data/qa_data.json")

MAX_LENGTH = 256

NUM_LABELS = 2

# Small test run first
TRAIN_LIMIT = 1000
VAL_LIMIT = 300

BATCH_SIZE = 2

EPOCHS = 1

LEARNING_RATE = 2e-5


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 60)
print("LOADING DATA")
print("=" * 60)

raw_data = load_raw_qa(DATA_PATH)

train_data, val_data, test_data = create_qa_splits(
    raw_data
)

# Small subset for the first test
train_data = train_data[:TRAIN_LIMIT]
val_data = val_data[:VAL_LIMIT]

print("Original QA pairs:", len(raw_data))
print("Training examples:", len(train_data))
print("Validation examples:", len(val_data))
print("Test examples:", len(test_data))


# ============================================================
# 2. FORMAT INPUT
# ============================================================

def format_example(example):

    text = (
        "Knowledge: "
        + example["knowledge"]
        + "\nQuestion: "
        + example["question"]
        + "\nAnswer: "
        + example["answer"]
    )

    return text


train_texts = [
    format_example(example)
    for example in train_data
]

val_texts = [
    format_example(example)
    for example in val_data
]

train_labels = [
    example["label"]
    for example in train_data
]

val_labels = [
    example["label"]
    for example in val_data
]


# ============================================================
# 3. LOAD TOKENIZER
# ============================================================

print("\n" + "=" * 60)
print("LOADING DeBERTa TOKENIZER")
print("=" * 60)

# IMPORTANT:
# use_fast=False forces the SentencePiece tokenizer
# instead of trying to convert the tokenizer to tiktoken.

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=False
)

print("Tokenizer loaded successfully.")


# ============================================================
# 4. TOKENIZE DATA
# ============================================================

print("\nTokenizing training data...")

train_encodings = tokenizer(
    train_texts,
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH,
)

print("Tokenizing validation data...")

val_encodings = tokenizer(
    val_texts,
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH,
)

print("Tokenization completed.")


# ============================================================
# 5. PYTORCH DATASET
# ============================================================

class HallucinationDataset(torch.utils.data.Dataset):

    def __init__(self, encodings, labels):

        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):

        item = {
            key: torch.tensor(value[idx])
            for key, value in self.encodings.items()
        }

        item["labels"] = torch.tensor(
            self.labels[idx],
            dtype=torch.long
        )

        return item

    def __len__(self):

        return len(self.labels)


train_dataset = HallucinationDataset(
    train_encodings,
    train_labels
)

val_dataset = HallucinationDataset(
    val_encodings,
    val_labels
)


# ============================================================
# 6. LOAD DEBERTA MODEL
# ============================================================

print("\n" + "=" * 60)
print("LOADING DeBERTa-v3-base MODEL")
print("=" * 60)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS
)

print("Model loaded successfully.")


# ============================================================
# 7. EVALUATION METRICS
# ============================================================

def compute_metrics(eval_pred):

    predictions, labels = eval_pred

    # Convert logits into probabilities
    probabilities = torch.softmax(
        torch.tensor(predictions),
        dim=1
    )[:, 1].numpy()

    # Probability >= 0.5 → hallucinated
    predicted_labels = (
        probabilities >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        labels,
        predicted_labels
    )

    f1 = f1_score(
        labels,
        predicted_labels
    )

    auroc = roc_auc_score(
        labels,
        probabilities
    )

    return {
        "accuracy": accuracy,
        "f1": f1,
        "auroc": auroc,
    }


# ============================================================
# 8. TRAINING ARGUMENTS
# ============================================================

training_args = TrainingArguments(

    output_dir="./checkpoints/deberta_test",

    num_train_epochs=EPOCHS,

    learning_rate=LEARNING_RATE,

    per_device_train_batch_size=BATCH_SIZE,

    per_device_eval_batch_size=BATCH_SIZE,

    eval_strategy="epoch",

    save_strategy="no",

    logging_steps=50,

    report_to="none",

    use_cpu=True,
)


# ============================================================
# 9. TRAINER
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=val_dataset,

    compute_metrics=compute_metrics,
)


# ============================================================
# 10. TRAIN MODEL
# ============================================================

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

trainer.train()


# ============================================================
# 11. VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)

results = trainer.evaluate()

for key, value in results.items():

    if isinstance(value, float):

        print(
            f"{key}: {value:.4f}"
        )