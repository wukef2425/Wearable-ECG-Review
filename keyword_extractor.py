import pandas as pd
import re
import numpy as np

YELLOW = "\033[93m"
RESET = "\033[0m"

df = pd.read_csv("ECG_ML_SLR_merged_deduplicated.csv")

task_keywords = [
    "atrial fibrillation", "arrhythmia", "R-peak detection", "QRS detection",
    "PVC detection", "AF detection"
]

metric_keywords = [
    "accuracy", "precision", "recall", "f1[- ]?score", "sensitivity",
    "specificity", "auc", "detection rate"
]

model_keywords = {
    r"convolutional[- ]neural[- ]networks?": "CNN",
    r"cnn": "CNN",
    r"recurrent[- ]neural[- ]networks?": "RNN",
    r"rnn": "RNN",
    r"long short[- ]?term memory": "LSTM",
    r"lstm": "LSTM",
    r"gru": "GRU",
    r"transformer[- ]?based": "Transformer",
    r"transformer": "Transformer",
    r"deep residual network|resnet": "ResNet",
    r"mlp": "MLP",
    r"support vector machine|svm": "SVM",
    r"random forest": "Random Forest",
    r"knn": "KNN",
    r"decision tree": "Decision Tree",
    r"deep neural network": "DNN",
}

dataset_keywords = {
    r"mit[-\s]?bih": "MIT-BIH",
    r"ptb[-\s]?xl": "PTB-XL",
    r"physionet": "PhysioNet",
    r"afdb": "AFDB",
    r"incart": "INCART",
    r"ludb": "LUDB",
    r"cudb": "CUDB",
    r"nsrdb": "NSRDB"
}

def normalize_task(task):
    alias_map = {
        "af detection": "atrial fibrillation",
        "atrial fibrillation detection": "atrial fibrillation",
        "detection of af": "atrial fibrillation",
        "arrhythmia classification": "arrhythmia",
        "heartbeat categorization": "ECG classification",
        "pvc classification": "PVC detection",
    }
    return alias_map.get(task.lower(), task)

task_pattern = re.compile(r'|'.join(task_keywords), re.IGNORECASE)
metric_pattern = re.compile(r'|'.join(metric_keywords), re.IGNORECASE)
model_pattern = re.compile(r'|'.join(model_keywords.keys()), re.IGNORECASE)
dataset_pattern = re.compile(r'|'.join(dataset_keywords), re.IGNORECASE)

# Value pattern: 60%, 97.7%, .89 etc.
value_pattern = re.compile(r'\b\d{1,3}(?:\.\d+)?\s?%|(?<!\d)\.\d+(?!\d)')

def extract_structured_results(abstract):
    sentences = re.split(r'(?<=[.!?]) +', abstract)
    results = []

    for s in sentences:
        tasks = set([normalize_task(t.lower()) for t in task_pattern.findall(s)])
        if not tasks:
            continue

        metrics = metric_pattern.findall(s)
        values = value_pattern.findall(s)

        if not metrics or not values:
            continue

        if len(metrics) != len(values):
            print(f"{YELLOW}WARNING: Mismatch between metric and value count: {len(metrics)} vs {len(values)}{RESET}")
            print(f"{YELLOW}Sentence: {s}{RESET}\n")
            continue

        cleaned_values = []
        for value in values:
            value_cleaned = value.strip()
            if value_cleaned.startswith("."):
                value_cleaned = str(round(float(value_cleaned) * 100, 1)) + "%"
            elif "%" not in value_cleaned:
                value_cleaned += "%"
            cleaned_values.append(value_cleaned)

        for task in tasks:
            for metric, value in zip(metrics, cleaned_values):
                results.append({
                    "Sentence": s.strip(),
                    "Task": task,
                    "Metric": metric.lower(),
                    "Value": value
                })

    return results if results else np.nan

def extract_models(abstract):
    found_models = []
    for pattern, alias in model_keywords.items():
        if re.search(pattern, abstract, re.IGNORECASE):
            found_models.append(alias)
    return list(set(found_models)) if found_models else np.nan

def extract_datasets(abstract):
    found_datasets = []
    for pattern, name in dataset_keywords.items():
        if re.search(pattern, abstract, re.IGNORECASE):
            found_datasets.append(name)
    return list(set(found_datasets)) if found_datasets else np.nan

df["Structured_Results"] = df["Abstract"].apply(extract_structured_results)
df["Models"] = df["Abstract"].apply(extract_models)
df["Datasets"] = df["Abstract"].apply(extract_datasets)

structured_rows = []
for idx, row in df.iterrows():
    if isinstance(row["Structured_Results"], list):
        for item in row["Structured_Results"]:
            structured_rows.append({
                "Title": row["Title"],
                "Link": row["Link"],
                "Year": row["Year"],
                "Abstract": row["Abstract"],
                "Models": ", ".join(row["Models"]) if isinstance(row["Models"], list) else row["Models"],
                "Task": item["Task"],
                "Metric": item["Metric"],
                "Value": item["Value"],
                "Datasets": ", ".join(row["Datasets"]) if isinstance(row["Datasets"], list) else row["Datasets"],
                "Result Sentence": item["Sentence"]
            })

structured_df = pd.DataFrame(structured_rows)
structured_df.to_csv("structured_ecg_results.csv", index=False)

print("fin")