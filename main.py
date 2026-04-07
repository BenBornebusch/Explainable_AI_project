import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
from lime.lime_text import LimeTextExplainer
from sklearn.metrics import f1_score
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


'''
---------------------------------------------------------------------------------------------------
Configuration
---------------------------------------------------------------------------------------------------
'''
'''
SEED ensures reproducibility --> without it LIME would give slightly different results each run
'''
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "Hate-speech-CNERG/bert-base-uncased-hatexplain"
N_SAMPLES = 20
ONLY_OFFENSIVE_AND_HATE = True
SHAP_MAX_EVALS = 100 #reduced for CPU speed
LIME_NUM_FEATURES = 20 #this is the number of words LIME will consider
LIME_NUM_SAMPLES = 200
MAX_LENGTH = 512 #the maximum length of BERT's token

ID2LABEL = {0: "normal", 1: "offensive", 2: "hate"}
LABEL_NAME_TO_ID = {
    "normal": 0,
    "offensive": 1,
    "hatespeech": 2,
    "hate": 2,
}

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True) #creating output folder if not exisring


'''
Utility funcitons
'''
def resolve_data_path(filename: str) -> Path:
    '''Tries multiple common locations for the data files and returns the first one found.'''
    candidates = [Path("data") / filename, Path(filename), Path("/mnt/data") / filename,]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find '{filename}'. Checked: "
        + ", ".join(str(p) for p in candidates)
    )


def normalize_scores(scores: List[float]) -> List[float]:
    '''Normalises a list of scores to [0, 1] by dividing by the maximum value.'''
    arr = np.array(scores, dtype=float)
    if arr.size == 0:
        return []
    max_val = float(arr.max())
    if max_val <= 0:
        return [0.0] * len(arr) #this avoids division by zero
    return (arr / max_val).tolist()



'''
---------------------------------------------------------------------------------------------------
1) Loading the Data --> from the data file
---------------------------------------------------------------------------------------------------
'''
def load_hatexplain(n_samples = N_SAMPLES, only_offensive_and_hate= ONLY_OFFENSIVE_AND_HATE,):
    dataset_path = resolve_data_path("dataset.json")
    split_path = resolve_data_path("post_id_divisions.json")

    with open(dataset_path, "r", encoding="utf-8") as f:
        full_data = json.load(f)

    with open(split_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    test_ids = set(splits["test"]) #only keeping the posts that belong to the test split
    samples = []

    for post_id, row in full_data.items():
        if post_id not in test_ids:
            continue #skipping training and validation posts

        tokens = row.get("post_tokens", [])
        if not tokens:
            continue #skipping posts with no tokens

        text = " ".join(tokens) #joining the tokens back into one single string for the model

        annotators = row.get("annotators", [])
        if not annotators:
            continue #skip the posts with no annotators

        label_votes = [ann["label"] for ann in annotators if "label" in ann] #majority vote across annotators
        if not label_votes:
            continue

        majority_label_name = max(set(label_votes), key=label_votes.count) #the most common label wins
        label_id = LABEL_NAME_TO_ID.get(majority_label_name, 0)

        rationales = row.get("rationales", [])
        valid_rationales = [r for r in rationales if len(r) == len(tokens)]
        if not valid_rationales:
            continue

        rationale_sum = np.sum(valid_rationales, axis=0)
        human_rationale = (rationale_sum >= 2).astype(int).tolist()

        if only_offensive_and_hate and label_id == 0:
            continue #skipping normal posts

        samples.append(
            {
                "post_id": post_id,
                "text": text,
                "tokens": tokens,
                "label": int(label_id),
                "label_name": ID2LABEL[label_id],
                "human_rationale": human_rationale,
            }
        )

        if len(samples) >= n_samples:
            break

    print(f"Loaded {len(samples)} samples from HateXplain test split.")
    return samples


'''
---------------------------------------------------------------------------------------------------
2) Loading the Model --> using pre-trained Bert model
---------------------------------------------------------------------------------------------------
'''
def load_model(model_name: str = MODEL_NAME):
    """Downloads and loads a BERT model pre-trained on HateXplain from HuggingFace."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name,output_attentions=True,).to(DEVICE)
    model.eval()
    return tokenizer, model

'''
---------------------------------------------------------------------------------------------------
3) prediction function for LIME and SHAP
---------------------------------------------------------------------------------------------------
'''
def build_predict_fn(tokenizer, model):
    """
    Wraps BERT into a predict_proba(texts) function that returns a probability
    array --> the interface both LIME and SHAP need
    """
    def predict_proba(texts: List[str]) -> np.ndarray:
        probs = []
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH, padding=True).to(DEVICE)

            with torch.no_grad(): #no gradient needed --> this is going to save memory and speeds up inference
                logits = model(**inputs).logits

            prob = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            probs.append(prob)

        return np.array(probs)

    return predict_proba


'''
---------------------------------------------------------------------------------------------------
4) Method 1: Attention Maps
---------------------------------------------------------------------------------------------------
'''
def get_attention_scores(text: str, tokenizer, model) -> List[float]:
    '''
    Extracts importance scores directly from BERT by averaging the CLS token's
    outgoing attention across all heads in the last layer, then maps scores back to whole words.
    '''
    words = text.split()
    
    inputs = tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    last_attn = outputs.attentions[-1]  
    cls_attn = last_attn[0, :, 0, :] 
    mean_attn = cls_attn.mean(dim=0).cpu().numpy()
    #word_ids() maps all of the sub-word tokens back to their original word index
    word_ids = inputs.word_ids(batch_index=0)
    word_scores = np.zeros(len(words), dtype=float)

    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue #skipping special tokens
        if 0 <= word_idx < len(word_scores):
            word_scores[word_idx] += float(mean_attn[token_idx]) #aacumulating the sub-word scores

    return normalize_scores(word_scores.tolist())

'''
---------------------------------------------------------------------------------------------------
5) Method 2: LIME 
---------------------------------------------------------------------------------------------------
'''
def get_lime_scores(text: str, predict_fn, explainer: LimeTextExplainer, num_features: int = LIME_NUM_FEATURES, num_samples: int = LIME_NUM_SAMPLES) -> List[float]:
    '''
    Runs LIME on one text by randomly masking words, re-predicting, and fitting
    a local linear model --> returns absolute coefficient values as importance scores.
    '''
    predicted_class = int(np.argmax(predict_fn([text])[0])) #gets the models predicted class

    explanation = explainer.explain_instance(text_instance=text, classifier_fn=predict_fn, labels=(predicted_class,), num_features=num_features, num_samples=num_samples)

    score_map = {
        feature: abs(weight)
        for feature, weight in explanation.as_list(label=predicted_class)
    }

    words = text.split()
    scores = [float(score_map.get(word, 0.0)) for word in words]
    return normalize_scores(scores)

'''
---------------------------------------------------------------------------------------------------
6) Method 3: SHAP
---------------------------------------------------------------------------------------------------
'''
def get_shap_scores(text: str, predict_fn, explainer,) -> List[float]:
    '''
    Runs SHAP PartitionExplainer on one text using word-level masking to estimate
    Shapley values --> returns absolute values for the predicted class.
    '''
    shap_values = explainer([text], max_evals=SHAP_MAX_EVALS) #max evals reduced for CPU speed
    predicted_class = int(np.argmax(predict_fn([text])[0])) #getting the models predicted class

    shap_tokens = list(shap_values.data[0]) #this is the actual words corresponding to each Shapley value
    shap_importances = shap_values.values[0, :, predicted_class] #these are the shapley values for the predicted class


    score_map: Dict[str, float] = {}
    for token, value in zip(shap_tokens, shap_importances):
        key = str(token).strip()
        if key:
            score_map[key] = abs(float(value))

    words = text.split()
    scores = [float(score_map.get(word, 0.0)) for word in words]
    return normalize_scores(scores)


'''
---------------------------------------------------------------------------------------------------
7) Evaluation: Token f1 and IoU vs the human rationales
---------------------------------------------------------------------------------------------------
'''
def binarise_scores(scores: List[float], threshold: float = None) -> List[int]:
    '''converts the continuous scores to binary by flagging the words above the mean as importnatn'''
    arr = np.array(scores, dtype=float)
    if arr.size == 0:
        return []
    if threshold is None:
        threshold = float(arr.mean())
    return (arr >= threshold).astype(int).tolist()


def token_f1(pred, gold):
    '''Computes token-level F1 score between predicted and human-annotated important tokens.'''
    
    if sum(gold) == 0:
        return float("nan") #skip samples with no human annotation
    return f1_score(gold, pred, zero_division=0)


def token_iou(pred, gold):
    '''Computes Intersection over Union between predicted and human-annotated token sets.'''
    pred_set = set(i for i, v in enumerate(pred) if v == 1)
    gold_set = set(i for i, v in enumerate(gold) if v == 1)
    if not gold_set:
        return float("nan") #skipping sample with no human annotation
    intersection = len(pred_set & gold_set)
    union = len(pred_set | gold_set)
    return intersection / union if union > 0 else 0.0


def evaluate_all(samples, attn_list, lime_list, shap_list):
    results = {m: {"f1": [], "iou": []} for m in ("attention", "lime", "shap")}

    for i, sample in enumerate(samples):
        gold = sample["human_rationale"] #this is the ground truth from human annotators --> called golden standard 

        if sum(gold) == 0:
            continue #no annotation available --> skip

        for method, scores in [
            ("attention", attn_list[i]),
            ("lime", lime_list[i]),
            ("shap", shap_list[i]),
        ]:
            pred = binarise_scores(scores) #converting continuous scores to binary
            results[method]["f1"].append(token_f1(pred, gold))
            results[method]["iou"].append(token_iou(pred, gold))

    summary = {}
    for m, vals in results.items():
        f1s = [v for v in vals["f1"] if not np.isnan(v)] #filtering out skipped samples
        ious = [v for v in vals["iou"] if not np.isnan(v)]

        summary[m] = {
            "mean_f1": round(np.mean(f1s), 4) if f1s else 0.0,
            "mean_iou": round(np.mean(ious), 4) if ious else 0.0,
            "n": len(f1s), #this is the number of valid samples used
        }

    return summary


'''
---------------------------------------------------------------------------------------------------
8) Visualization
---------------------------------------------------------------------------------------------------
'''
def plot_metric_comparison(summary, save_path=None):
    methods = list(summary.keys())
    f1s = [summary[m]["mean_f1"] for m in methods]
    ious = [summary[m]["mean_iou"] for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))

    b1 = ax.bar(x - width / 2, f1s, width, label="Token F1")
    b2 = ax.bar(x + width / 2, ious, width, label="Token IoU")

    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in methods])
    ax.set_ylabel("Score")
    ax.set_title("XAI Method Alignment with Human Rationales")
    ax.set_ylim(0, 1)
    ax.legend()

    ax.bar_label(b1, fmt="%.2f", padding=3, fontsize=8)
    ax.bar_label(b2, fmt="%.2f", padding=3, fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)

    plt.show()
    plt.close()

'''
---------------------------------------------------------------------------------------------------
9) Main Pipeline
---------------------------------------------------------------------------------------------------
'''
def main():
    # 1. Load data + model
    samples = load_hatexplain()
    tokenizer, model = load_model()
    predict_fn = build_predict_fn(tokenizer, model)

    # Create explainers ONCE
    lime_explainer = LimeTextExplainer(class_names=list(ID2LABEL.values()),random_state=SEED)
    shap_masker = shap.maskers.Text(r"\W+")
    shap_explainer = shap.PartitionExplainer(predict_fn, shap_masker)

    attn_scores_all = []
    lime_scores_all = []
    shap_scores_all = []

    # 2. Generate explanations
    for sample in tqdm(samples, desc="Computing explanations"):
        text = sample["text"]
        attn_scores_all.append(get_attention_scores(text, tokenizer, model))
        lime_scores_all.append(get_lime_scores(text, predict_fn, lime_explainer))
        shap_scores_all.append(get_shap_scores(text, predict_fn, shap_explainer))

    # Step 3: Evaluate all methods against human rationales
    summary = evaluate_all(samples, attn_scores_all, lime_scores_all, shap_scores_all)

    print("\nEvaluation Summary:")
    for method, metrics in summary.items():
        print(f"  {method.upper():12s}  F1={metrics['mean_f1']:.3f}  IoU={metrics['mean_iou']:.3f}  (n={metrics['n']})")

    # Step 4: Save bar chart
    plot_metric_comparison(summary, save_path=OUTPUT_DIR / "metric_comparison.png")

    print("\nSaved outputs to:", OUTPUT_DIR.resolve())
    print("Done.")


if __name__ == "__main__":
    main()