import json
from pathlib import Path


def load_raw_qa(file_path):
    """Load the 10,000 original HaluEval QA pairs."""

    data = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    return data


def expand_qa_data(data):
    """
    Convert each QA pair into two classification examples.

    label 0 = factual/right answer
    label 1 = hallucinated answer
    """

    examples = []

    for item in data:

        # Factual answer
        examples.append({
            "knowledge": item["knowledge"],
            "question": item["question"],
            "answer": item["right_answer"],
            "label": 0
        })

        # Hallucinated answer
        examples.append({
            "knowledge": item["knowledge"],
            "question": item["question"],
            "answer": item["hallucinated_answer"],
            "label": 1
        })

    return examples


def create_qa_splits(data):
    """
    Split original QA pairs first, then expand them.

    7,000 pairs -> train
    1,500 pairs -> validation
    1,500 pairs -> test
    """

    train_pairs = data[:7000]
    val_pairs = data[7000:8500]
    test_pairs = data[8500:10000]

    train = expand_qa_data(train_pairs)
    validation = expand_qa_data(val_pairs)
    test = expand_qa_data(test_pairs)

    return train, validation, test


if __name__ == "__main__":

    path = Path(__file__).parent / "data" / "qa_data.json"

    # Load original paired records
    raw_data = load_raw_qa(path)

    print("Original QA pairs:", len(raw_data))

    # Create splits
    train, validation, test = create_qa_splits(raw_data)

    print("\nDataset split:")
    print("Train:", len(train))
    print("Validation:", len(validation))
    print("Test:", len(test))

    print("\nLabel distribution:")
    print("Train factual:",
          sum(x["label"] == 0 for x in train))
    print("Train hallucinated:",
          sum(x["label"] == 1 for x in train))

    print("\nFirst training example:")
    print(train[0])