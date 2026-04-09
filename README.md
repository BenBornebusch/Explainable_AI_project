**Research Question:** How does an attention-based explanation compare to post-hoc methods (LIME and SHAP) in identifying important features that drive hateful predictions in LLMs?

---

## Project Structure

```
Explainable_AI_project/
├── main.py                     
├── requirements.txt           
├── README.md                   
├── data/
│   ├── dataset.json            
│   └── post_id_divisions.json  
└── outputs/
    └── metric_comparison.png  
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/BenBornebusch/Explainable_AI_project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Running the code

```bash
python main.py
```

This will:
1. Load 20 offensive and hate speech posts from the HateXplain test split
2. Load `bert-base-uncased-hatexplain` from HuggingFace (cached after first run)
3. Compute **Attention**, **LIME**, and **SHAP** scores for each post
4. Evaluate all three methods against human-annotated rationales (Token F1, IoU)
5. Save a bar chart to `outputs/metric_comparison.png`

---

## Methods

| Method | Type | Description |
|---|---|---|
| Attention Maps | Intrinsic | CLS token attention from last BERT layer, averaged over heads |
| LIME | Post-hoc | Local linear surrogate via random word masking |
| SHAP | Post-hoc | Shapley value attribution via PartitionExplainer |

---

## Evaluation

Each method's scores are binarised using a mean threshold and compared against the human rationales using:
- **Token F1** — harmonic mean of precision and recall at the token level
- **Token IoU** — intersection over union of flagged token sets
