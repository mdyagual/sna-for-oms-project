# Two-Stage Open-Source Ensemble: Intention Detection for SNA

This script (`detect.py`) implements a high-performance, two-stage Natural Language Processing (NLP) pipeline designed to extract Knowledge Graphs and classify intentions from raw text. The architecture is entirely **Open-Source**, utilizing the Hugging Face Inference API to eliminate reliance on proprietary models.

---

## System Architecture

The pipeline follows a **Segmented Model Architecture** to ensure high accuracy for specific speech categories while maintaining computational efficiency.

### Stage 1: Base Graph Extraction
A generative model (**Llama 3.1 8B Instruct**) serves as the "Intelligence Analyst".
* **Task**: Performs Named Entity Recognition (NER) and Relation Extraction (RE).
* **Logic**: Extracts socially relevant entities (Person, Group, Institution, Location) and their core interactions.

### Stage 2: Specialized Taxonomy Routing
Extracted relationships are routed through three specialized model groups to determine the final intention classification:

| Group | Model | Taxonomy Focus |
| :--- | :--- | :--- |
| **Group A (Encoders)** | `twitter-roberta-base-sentiment` | **Sentimental** and **Emotional** speech. |
| **Group B (Safety)** | `roberta-hate-speech-dynabench` | **Hate** and **Threat** detection. |
| **Group C (Reasoners)** | `Meta-Llama-3-8B-Instruct` | **Extremist**, **Radicalism**, and **Violated-scaled** speech. |

---

## Technical Features

* **Parallel Processing**: Uses `ThreadPoolExecutor` to process multiple text entries simultaneously, significantly reducing execution time.
* **Exponential Backoff**: Implements a robust `retry_call` wrapper to handle API rate limits and server instability (Errors 500/402).
* **Taxonomy Normalization**: Automatically maps "Neutral" interactions to the `SENTIMENTAL` category to maintain 100% compliance with the required 7-category taxonomy.
* **Dynamic Confidence Labels**: Generates real-time accuracy percentages for each classification to meet academic reporting standards.

---

## Installation & Usage

### 1. Requirements
* Python 3.10+
* Libraries: `huggingface_hub`, `python-dotenv`

### 2. Environment Setup
Create a `.env` file in the root directory:
```env
HF_TOKEN=your_hugging_face_token
```

## Execution
The script processes examples.txt and outputs the final graph to results/extraction_results.json.

```bash
python detect.py
``` 

## Official Taxonomy

The script classifies all interactions into one of these 7 mandatory categories:

- EMOTIONAL
- EXTREMIST
- HATE
- RADICALISM
- SENTIMENTAL (includes mundane/neutral actions)
- THREAT
- VIOLATED