# Unified behavioral Framework

This repository implements an end-to-end pipeline for analyzing offender-management and incident-related text logs through a combination of natural language processing, knowledge-graph construction, social network analysis (SNA), and graph neural networks (GNNs). The system extracts entities and relationships from unstructured text, normalizes them into a graph schema, stores the graph in Neo4j, computes network metrics, and trains a GNN to predict escalation risk.

## What the project does

The project is designed to transform raw textual incident or observation reports into actionable intelligence. In practice, it can:

- ingest text logs and extract structured entities and relationships;
- classify the report into a taxonomy of risk-related categories;
- build and maintain a knowledge graph in Neo4j;
- calculate SNA metrics such as tie strength, reputation, betweenness, eigenvector centrality, and community structure;
- train a GNN-based risk escalation model over the graph topology and relational features;
- expose the functionality through a FastAPI service and a notebook-based evaluation workflow.

The core idea is to move from free-form log text to a graph representation that supports both semantic reasoning and quantitative network analysis.

## Project architecture

The codebase is organized into four layers:

### 1. NLP extraction layer
Location: [core/nlp_layer/extract_detect.py](core/nlp_layer/extract_detect.py)

This layer is responsible for:

- calling Hugging Face inference models to extract entities and relations;
- classifying each log into a taxonomy category;
- cleaning and normalizing the LLM output;
- storing processed results in the local cache.

Key functions:

- `safe_json_load` – parses model output into JSON and recovers from malformed responses.
- `heal_and_deduplicate_graph` – removes garbage entities/relations, repairs orphan nodes, and deduplicates edges.
- `normalize_confidence` – standardizes confidence scores into the $[0,1]$ range.
- `process_single_log_safely` – processes one log with retry logic and concurrency control.
- `async_retry_call` – retries Hugging Face requests when they fail or hit rate limits.
- `get_master_taxonomy` – routes a report through multiple classification models and applies a decision policy.
- `process_entry` – runs extraction and taxonomy classification for a single input text.
- `process_entry_file` – processes a batch of lines from a text file.

### 2. Knowledge graph and SNA layer
Location: [core/kg_sna_layer/graph_builder.py](core/kg_sna_layer/graph_builder.py)

This layer connects the NLP output to Neo4j and performs social network analysis.

Key responsibilities:

- ingesting processed entities and relations into a graph database;
- creating nodes and edges with risk-weighted metadata;
- computing relational metrics such as tie strength and reputation;
- computing structural metrics such as betweenness, eigenvector centrality, and community assignments;
- providing topology features used by the GNN model.

Main class and functions:

- `AsyncKnowledgeGraphManager` – manages Neo4j connections, ingestion, and metric calculation.
- `ingest_log_data` – upserts graph entities and relationships from processed logs.
- `calculate_sna_metrics` – computes and stores SNA metrics on the graph.
- `calculate_topological_edge_metrics` – computes topology-based edge features such as common neighbors, Jaccard similarity, and Adamic-Adar.
- `normalize_confidence` – confidence normalization for graph property values.
- `wipe_sna_metrics` – removes calculated SNA metrics from the graph.
- `calculate_strict_fold_topology` – constructs topology features for model training and evaluation.

### 3. GNN risk prediction layer
Location: [core/gnn_layer/risk_predict.py](core/gnn_layer/risk_predict.py)

This layer focuses on learning from the graph structure and relationship features to predict which edges indicate likely escalation risk.

Key components:

- `RiskEscalationGNN` – a Graph Attention Network (GATv2-based) model for edge-level risk prediction.
- `FocalLoss` – a class-balanced loss function designed to deal with imbalanced edge labels.
- `fetch_graph_from_neo4j` – retrieves nodes, edges, taxonomy information, and topology features from Neo4j.
- `graph_edge_oversample` – balances the training data by oversampling minority-class edges.
- `train_model` – trains the GNN and stores training history.
- `evaluate_model` – evaluates the model using ROC-AUC and recall metrics.
- `evaluate_with_kfold` – performs repeated stratified K-fold evaluation.
- `generate_threat_report` – converts model predictions into ranked alerts.

### 4. API layer
Location: [core/api_server.py](core/api_server.py)

The API layer exposes the pipeline through FastAPI.

Available endpoints:

- `GET /` – redirects to the Swagger UI.
- `GET /health` – health check endpoint.
- `POST /api/v1/extract` – extracts graph data from a supplied text payload.
- `POST /api/v1/sna-metrics` – triggers SNA metric recalculation in Neo4j.
- `GET /api/v1/test-project` – processes a demo dataset and populates the graph.

### 5. Data and persistence layer
Location: [data/db_storage.py](data/db_storage.py)

The project uses:

- TinyDB for local caching of processed logs;
- Neo4j for persistent graph storage and graph analytics;
- JSON and text datasets located under [data/files](data/files).

## Main workflow

The repository supports two main usage paths:

### A. API-driven workflow

1. Start the FastAPI service.
2. Send a text payload to the extraction endpoint.
3. The NLP layer extracts entities and relations and assigns taxonomy.
4. The graph layer ingests the output into Neo4j.
5. SNA metrics are calculated and stored.
6. The resulting graph can be used for risk analysis or downstream GNN training.

### B. Batch and notebook workflow

1. Prepare a text file containing log samples in [data/files](data/files).
2. Run the batch processor to process many logs and cache them locally.
3. Ingest the processed results into Neo4j.
4. Use the notebook [notebooks/risk_final_evaluation.ipynb](notebooks/risk_final_evaluation.ipynb) to:
   - fetch the graph from Neo4j;
   - build topology features;
   - train and evaluate the model;
   - visualize convergence and threshold behavior;
   - produce threat alerts.

## Notebook usage

The notebook [notebooks/risk_final_evaluation.ipynb](notebooks/risk_final_evaluation.ipynb) is the main analysis and evaluation entry point for the GNN component. It covers:

- environment setup and module import;
- graph loading from Neo4j;
- feature engineering and distribution visualization;
- model convergence plots;
- threshold calibration for risk classification;
- evaluation of the trained graph model.

## Supported data and assets

The repository ships with sample datasets and cached results in:

- [data/files](data/files)
- [data/extract_detect_cache.json](data/extract_detect_cache.json)
- [data/extract_detect_paper.json](data/extract_detect_paper.json)

These files are useful for testing the pipeline without immediately needing a fresh external dataset.

## Getting started

### Prerequisites

Before reproducing the project, make sure you have:

- Python 3.10 or 3.11
- a running Neo4j instance
- a Hugging Face access token with permission to use the models referenced in [core/config.py](core/config.py)
- optional: a CUDA-enabled GPU for faster PyTorch training, though CPU execution is possible

### 1. Clone and enter the repository

```bash
git clone <repository-url>
cd sna-for-oms-project
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If you want the complete environment used for experimentation, you may also use:

```bash
pip install -r requirements_full.txt
```

### 4. Configure environment variables

Create a `.env` file in the repository root with the following values:

```env
HF_TOKEN=your_huggingface_token
NEO4J_USER=neo4j
NEO4J_PASS=your_password
NEO4J_URL_PAPER=bolt://localhost:7687
```

The project loads these values through [core/config.py](core/config.py).

### 5. Start Neo4j

Ensure that your Neo4j instance is reachable at the URL specified in the environment variables.

### 6. Run the API

```bash
uvicorn core.api_server:app --reload
```

Then open the Swagger UI at:

- http://127.0.0.1:8000/docs

### 7. Run the batch processor

You can process a batch of demo data with:

```bash
python -c "import asyncio; from core.nlp_layer.extract_detect import process_entry_file; asyncio.run(process_entry_file())"
```

### 8. Launch the notebook

Start Jupyter from the repository root:

```bash
jupyter notebook
```

Then open [notebooks/risk_final_evaluation.ipynb](notebooks/risk_final_evaluation.ipynb).

## Requirements to replicate the project

To reproduce the behavior of this repository, the following are needed:

- Python environment with the packages listed in [requirements.txt](requirements.txt)
- access to Hugging Face inference endpoints for the LLMs configured in [core/config.py](core/config.py)
- a Neo4j database with sufficient permissions to create nodes, relationships, and properties
- sample text data in the format used by [data/files](data/files)
- optionally, Jupyter and notebook dependencies for running the evaluation notebook

## Notes and practical considerations

- The NLP stage depends on external LLM inference and may be rate-limited.
- The graph ingestion stage assumes that the Neo4j database is reachable and properly configured.
- Caching is enabled through TinyDB to avoid recomputing already-processed logs.
- The GNN workflow is intended for experimentation and evaluation rather than being a production-ready deployment by itself.

## Summary

This project brings together three complementary technologies:

- large language models for semantic extraction;
- knowledge graphs for structured representation and network analysis;
- graph neural networks for predictive risk modeling.

Together, they form a research-oriented pipeline for turning offense-related text into graph-based intelligence.
