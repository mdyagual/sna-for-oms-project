# Detection Taxonomy Evaluation

Author: Mishell Yagual Mendoza

Co-Author: Dinis Perpétuo Amado

Versión: 1.0

## Introduction
The general approach to detect intentions in plain text it is common to find that "one model to rule them all" is possible but the architecture you choose depends on your latency and accuracy requirements. This document pretends to show an analysis about the best options to consider to build a detection taxonomy solution around 7 specific types of speech:

- Emotional speech

- Extremist speech

- Hate speech

- Radicalism speech

- Sentimental speech

- Threat speech

- Violated-scaled speech

## NLP academic and Engineering Standars

### The Encoder approach

Models like RoBERTa-large or DeBERTa-v3 are the "Gold Standard" for pure classification. They are smaller, faster, and cheaper to run.

- Best for: Hate speech, Sentiment, Threats, and Violated-scaled speech.

- Recommended Model: facebook/roberta-hate-speech-dynabench-r4-target or any fine-tuned DeBERTa-v3 from the Hugging Face Hub.


### The Generative approach

Larger models like Llama 3.1 (8B/70B) or Mistral excel at "Radicalism" and "Extremist" speech because these often require understanding intent, subtext, and political nuance that smaller models miss.

- Best for: Emotional speech (subtle), Radicalism, and Extremist speech.

- Recommended Model: meta-llama/Llama-3.1-8B-Instruct.

In general terms, based on the speeches that will be evaluated:

| Category | Recommended Model Type | Why? |
| :--- | :--- | :--- |
| **Emotional / Sentimental** | **RoBERTa-base** | Emotions are well-mapped in standard NLP; small models handle this with high F1 scores. |
| **Hate / Threat Speech** | **DeBERTa-v3 / Detoxify** | Highly optimized for specific toxic keywords and syntax patterns. |
| **Extremist / Radicalism** | **Llama 3.1 (Instruct)** | Requires "reasoning" to detect dog-whistles or ideological recruitment language. |
| **Violated-scaled** | **Regression-based BERT** | If "scaled" means a 1–10 severity, you need a model trained for *regression*, not just classification. |

## Mapping Selected Models per script
It was done some research theorically about possible models to be used. The next table illustrate the model recomended for category:

| Taxonomy Category | Model Chosen in Previous Code | Architecture Type | Rationale for Selection |
| :--- | :--- | :--- | :--- |
| **Sentimental** | `twitter-roberta-base-sentiment` | **Encoder** (Path A) | This is a classic, lightweight, highly accurate model pre-trained specifically on social media sentiment. It is fast and reliable for standard categorization. |
| **Emotional** | `Meta-Llama-3-8B-Instruct` | **Generative** (Path B) | Emotion can be complex (e.g., sarcasm, layered frustration). A generative model understands context and tone better than a simple encoder. |
| **Hate / Threat** | `Llama-Guard-3-8B` | **Generative** (Path B, Specialized) | Llama Guard is explicitly designed by Meta for Trust and Safety. It acts like a classifier but uses generative architecture to robustly output safety categories based on strict policy guidelines. |
| **Radicalism** | `DeepSeek-R1-Distill-Llama-8B` | **Generative** (Path B, Reasoning) | Radicalism is highly subtle and ideological. The `R1` models use "Chain of Thought" reasoning, allowing the model to "think" through political subtext before outputting a classification. |
| **Extremist** | `DeepSeek-R1-Distill-Llama-70B` | **Generative** (Path B, Heavy) | Extremist speech requires deep geopolitical and historical context. A massive 70B parameter model handles complex nuance and dog-whistles, though it requires significant computational resources. |
| **Violated-scaled** | `Llama-3.2-3B-Instruct` | **Generative** (Path B, Lightweight) | A smaller, highly efficient generative model that is likely used to output a scaled number (e.g., "Rate severity 1-5") via structured prompt engineering. |


## Implementation Options to consider

### Single layer: Zero-Shot Classification

A zero-shot pipeline where labels are provided and the selected model decides which fits the best. This is an Encoder that just knows **English** so well that if you give it the label "hate speech" and a hateful sentence, it can guess they match. It is highly flexible but often less accurate.

```python
from transformers import pipeline

# Use a strong Zero-Shot model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "We must take back our country by any means necessary."
candidate_labels = ["emotional", "extremist", "hate speech", "radicalism", "threat"]

result = classifier(text, candidate_labels, multi_label=True)

print(f"Top Label: {result['labels'][0]} (Score: {result['scores'][0]:.4f})")

```

### Multilayer Specialized Model
For categories like hate speech and threats, specialized models are more accurate than general LLMs. This is also an Encoder. But researchers already did the hard work of training it specifically on millions of sentimental tweets. You just download it and use it. It is rigid (it only does sentiment), but highly accurate.


```python
#### For hate speech and threats 
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "Llama-Guard-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def detect_violation(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    # Logic to map logits to your taxonomy categories
    return torch.softmax(logits, dim=1)

```


```python
#### For sentiment
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def detect_violation(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    # Logic to map logits to your taxonomy categories
    return torch.softmax(logits, dim=1)

```



