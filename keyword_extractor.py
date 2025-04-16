import pandas as pd
import re
import numpy as np

YELLOW = "\033[93m"
RESET = "\033[0m"

df = pd.read_csv("ECG_ML_SLR_merged_deduplicated.csv")

task_keywords = [
    r"atrial fibrillation",
    r"arrhythmias?",
    r"r[- ]?peak detection",
    r"qrs detection",
    r"pvc detection",
    r"af detection",
    r"\baf\b"
]

metric_keywords = [
    r"accuracy", r"precision", r"recall", r"f1[- ]?score", r"sensitivity",
    r"specificity", r"auc", r"detection rate"
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
        "atrial fibrillation": "atrial fibrillation",
        "atrial fibrillation detection": "atrial fibrillation",
        "detection of af": "atrial fibrillation",
        "af detection": "atrial fibrillation",
        "af": "atrial fibrillation",
        "arrhythmia": "arrhythmia",
        "arrhythmias": "arrhythmia",
        "arrhythmia classification": "arrhythmia",
        "pvc detection": "PVC detection",
        "pvc classification": "PVC detection",
        "qrs detection": "QRS detection",
        "r-peak detection": "R-peak detection",
        "r peak detection": "R-peak detection",
    }
    return alias_map.get(task.lower(), task)

task_pattern = re.compile(r'|'.join(task_keywords), re.IGNORECASE)
metric_pattern = re.compile(r'|'.join(metric_keywords), re.IGNORECASE)
model_pattern = re.compile(r'|'.join(model_keywords.keys()), re.IGNORECASE)
dataset_pattern = re.compile(r'|'.join(dataset_keywords), re.IGNORECASE)

value_pattern = re.compile(r'\b\d{1,3}(?:\.\d+)?\s?%|\b\d?\.\d+\b')

def extract_structured_results(abstract):
    sentences = re.split(r'(?<=[.!?]) +', abstract)
    results = []

    for s in sentences:
        tasks = set([normalize_task(t) for t in task_pattern.findall(s)])
        if not tasks:
            continue

        metrics = metric_pattern.findall(s)
        values = value_pattern.findall(s)

        if not metrics or not values:
            for task in tasks:
                results.append({
                    "Sentence": s.strip(),
                    "Task": task,
                    "Metric": np.nan,
                    "Value": np.nan,
                    "Note": "No metric/value matched"
                })
            continue

        if len(metrics) != len(values):
            print(f"{YELLOW}WARNING: Mismatch between metric and value count: {len(metrics)} vs {len(values)}{RESET}")
            print(f"{YELLOW}Sentence: {s}{RESET}\n")
            for task in tasks:
                results.append({
                    "Sentence": s.strip(),
                    "Task": task,
                    "Metric": np.nan,
                    "Value": np.nan,
                    "Note": "Metric/value count mismatch"
                })
            continue

        cleaned_values = []
        for value in values:
            value_cleaned = value.strip()
            if re.fullmatch(r"\.\d+", value_cleaned):
                # ".89" → "89.0%"
                value_cleaned = str(round(float(value_cleaned) * 100, 1)) + "%"
            elif "%" in value_cleaned:
                # already percent, keep as is
                pass
            else:
                # e.g. 0.988 → keep as is
                value_cleaned = value_cleaned
            cleaned_values.append(value_cleaned)

        for task in tasks:
            for metric, value in zip(metrics, cleaned_values):
                results.append({
                    "Sentence": s.strip(),
                    "Task": task,
                    "Metric": metric.lower(),
                    "Value": value,
                    "Note": ""
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

def format_row(row, item=None):
    return {
        "Title": row["Title"],
        "Link": row["Link"],
        "Year": row["Year"],
        "Abstract": row["Abstract"],
        "Models": ", ".join(row["Models"]) if isinstance(row["Models"], list) else (row["Models"] or ""),
        "Datasets": ", ".join(row["Datasets"]) if isinstance(row["Datasets"], list) else (row["Datasets"] or ""),
        "Task": item["Task"] if item else np.nan,
        "Metric": item["Metric"] if item else np.nan,
        "Value": item["Value"] if item else np.nan,
        "Result Sentence": item["Sentence"] if item else "",
        "Note": item.get("Note", "") if item else "No task matched"
    }

structured_rows = []

for _, row in df.iterrows():
    if isinstance(row["Structured_Results"], list):
        structured_rows.extend([format_row(row, item) for item in row["Structured_Results"]])
    else:
        structured_rows.append(format_row(row))

structured_df = pd.DataFrame(structured_rows)
structured_df.to_csv("structured_ecg_results.csv", index=False)

print("fin")