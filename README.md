# Airline Feedback Triage System

### AI-Powered Airline Customer Feedback Analysis

An AI-powered system for automatically analyzing, classifying, and storing airline customer feedback using Large Language Models (LLMs).

The project processes customer tweets individually through a FastAPI backend. Each tweet is analyzed using the OpenAI API to identify its sentiment and one or more feedback categories.

The system supports three classification labels:

- **Complaint:** Customer complaints or negative experiences.
- **Compliment:** Positive feedback, appreciation, or praise.
- **Request:** Questions, assistance requests, or specific customer needs.

Sentiment analysis is performed independently using three labels: **positive, negative, and neutral**.

Additionally, the system generates a 1536-dimensional embedding for each tweet using OpenAI's `text-embedding-3-small` model.

The tweet, its embedding, and the AI analysis results are stored in a PostgreSQL database with the pgvector extension, running inside Docker.

An interactive Streamlit dashboard provides visualizations of customer feedback, comparisons of different AI models and prompt versions, semantic search, CSV upload and batch processing, and classification filtering.

The project uses a dataset containing **3,822 United Airlines tweets**.

## Features

- **Multi-label Classification:** Classifies each tweet into one or more categories: `complaint`, `compliment`, and `request`.
- **Sentiment Analysis:** Identifies customer sentiment as `positive`, `negative`, or `neutral`.
- **Tweet-by-Tweet Processing:** Processes each tweet independently through a FastAPI endpoint.
- **Structured AI Output:** Uses Pydantic to validate classification and sentiment results.
- **Embedding Generation:** Generates a 1536-dimensional embedding for each tweet using OpenAI's `text-embedding-3-small` model.
- **Vector Storage:** Stores embeddings in PostgreSQL using the pgvector extension.
- **Analysis History:** Stores AI analysis results separately from the original tweets.
- **Duplicate Prevention:** Prevents duplicate tweets using unique tweet identifiers.
- **Error Handling:** Handles invalid input, duplicate tweets, AI failures, embedding failures, and database errors.
- **Transaction Management:** Uses database transactions to prevent incomplete records.
- **CSV Batch Processing:** Processes tweets sequentially through the existing FastAPI workflow, including uploads from the Streamlit dashboard.
- **CSV Upload & Progress:** Accepts a local CSV file in Streamlit, previews its records, displays processing progress, and reports successful, duplicate, and failed submissions.
- **Semantic Search:** Uses query embeddings and pgvector cosine distance to rank stored tweets by semantic similarity, including when the exact search words do not appear in the tweet.
- **Classification Filtering:** Filters analyzed tweets by `All`, `Complaint`, `Compliment`, or `Request`, supporting multiple labels per tweet.
- **Interactive Dashboard:** Visualizes analyzed tweets, classification and sentiment distributions, semantic-search results, and model evaluation results.
- **Model Comparison:** Compares six model/prompt combinations using accuracy, Macro F1, latency, token usage, and estimated API costs.

## Technologies

| Component | Technology |
|---|---|
| Programming Language | Python |
| Backend Framework | FastAPI |
| AI Models | OpenAI (`gpt-4o-mini`, `gpt-4.1-mini`, `gpt-5-mini`) |
| Current Classification Model | `gpt-4o-mini` |
| Embedding Model | `text-embedding-3-small` |
| Structured Output | Pydantic |
| Database | PostgreSQL 17 |
| Vector Extension | pgvector |
| Containerization | Docker Desktop |
| ORM | SQLAlchemy |
| PostgreSQL Driver | psycopg2 |
| Dashboard | Streamlit |
| Data Processing | Pandas |
| Evaluation Metrics | Scikit-learn |
| Database Management | pgAdmin 4 |
| Version Control | Git & GitHub |

## System Architecture

The system consists of three main components:

1. **FastAPI Backend:** Receives customer tweets, performs AI analysis, generates embeddings, and stores the results.
2. **PostgreSQL + pgvector:** Stores tweet information, AI analysis results, and vector embeddings.
3. **Streamlit Dashboard:** Provides CSV upload, classification filters, semantic search via FastAPI, visualizations, and model evaluation results.

### Processing Workflow

Each tweet is processed independently through the `POST /comments` endpoint.

```text
                  New Tweet
                      |
                      v
                FastAPI Backend
                POST /comments
                      |
                      v
              Input Validation
              & Duplicate Check
                      |
                      v
                 OpenAI API
                      |
             +--------+--------+
             |                 |
             v                 v
       Classification      Sentiment
        (Multi-label)      (Single-label)
             |                 |
             +--------+--------+
                      |
                      v
              Embedding Generation
             text-embedding-3-small
                 (1536 dimensions)
                      |
                      v
             PostgreSQL + pgvector
                (Docker)
                      |
             +--------+--------+
             |                 |
             v                 v
          comments          analyses
        Tweet Data        Classification[]
        Embedding         Sentiment
                          Model
                          Prompt Version
                          Token Usage
                      |
                      v
                API Response
```

### Database Structure

The PostgreSQL database contains two main tables.

**1. `comments`**

Stores the original tweet information and its generated embedding.

The embedding is stored using the PostgreSQL `vector(1536)` data type provided by pgvector.

**2. `analyses`**

Stores the AI-generated classification and sentiment results, including the model name, prompt version, and token usage.

Multi-label classification is stored as a PostgreSQL `VARCHAR[]` array.

The two tables are connected through the foreign key `analyses.comment_id`.

This creates a one-to-many relationship: one comment can have multiple analysis records.

### Data Processing

The system supports two processing methods:

- **Single Tweet Processing:** A tweet is submitted directly through the FastAPI endpoint.
- **CSV Batch Processing:** Tweets are read from a CSV file and submitted sequentially to the same API endpoint. Files can be processed by the existing Python script or uploaded through the Streamlit dashboard.

In both cases, each tweet is analyzed before its results are saved.

Tweet information and analysis results are stored within a database transaction to prevent incomplete records.

The system also checks for duplicate tweet identifiers before processing.

## Installation & Setup

The following instructions describe how to set up the project locally.

### 1. Prerequisites

Before running the application, make sure the following tools are available:

- Python and pip
- Git
- Docker Desktop
- PostgreSQL client tools (optional)
- pgAdmin 4 (optional, for database inspection)
- An OpenAI API key

The project was developed on Windows using PyCharm and Docker Desktop.

### 2. Clone the Repository

Clone the GitHub repository:

```bash
git clone https://github.com/zhaleh-salehi/airline-feedback-triage.git
```

Navigate to the project directory:

```bash
cd airline-feedback-triage
```

### 3. PostgreSQL + pgvector with Docker

The project uses PostgreSQL 17 with the pgvector extension.

**Database configuration used during development:**

| Setting | Value |
|---|---|
| Docker Image | `pgvector/pgvector:pg17` |
| Container Name | `airline-pgvector` |
| Database Name | `airline_feedback` |
| PostgreSQL User | `airline_user` |
| Host Port | `5433` |
| Container Port | `5432` |
| pgvector Extension | `0.8.7` (verified environment) |

Make sure Docker Desktop is running.

Download the required Docker image:

```bash
docker pull pgvector/pgvector:pg17
```

Create a persistent Docker volume for PostgreSQL data:

```powershell
docker volume create airline_pgdata
```

Create and start the PostgreSQL container:

```powershell
docker run -d `
  --name airline-pgvector `
  -e POSTGRES_USER=airline_user `
  -e POSTGRES_PASSWORD=YOUR_SECURE_PASSWORD `
  -e POSTGRES_DB=airline_feedback `
  -p 127.0.0.1:5433:5432 `
  -v airline_pgdata:/var/lib/postgresql/data `
  pgvector/pgvector:pg17
```

Replace `YOUR_SECURE_PASSWORD` with a private PostgreSQL password.

The container uses a named Docker volume (`airline_pgdata`) to persist database files independently of the container lifecycle.

**Important:** These commands are intended for a new installation. Do not run `docker run` again if a container named `airline-pgvector` already exists.

The container image includes pgvector, but the extension must also be enabled in the application database:

```powershell
docker exec airline-pgvector psql -U airline_user -d airline_feedback -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

The Docker container exposes PostgreSQL on `localhost:5433`, which can be used by the FastAPI backend and pgAdmin 4.

Verify that the container is running:

```bash
docker ps
```

The PostgreSQL container should appear in the list of running containers.

### 4. Verify PostgreSQL Connection

For the development configuration, run:

```bash
docker exec airline-pgvector psql -U airline_user -d airline_feedback -c "SELECT current_database(), current_user, version();"
```

This verifies the PostgreSQL connection and displays the database name, current user, and server version.

### 5. Verify pgvector

Check whether the pgvector extension is enabled:

```bash
docker exec airline-pgvector psql -U airline_user -d airline_feedback -c "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';"
```

If the extension is enabled, PostgreSQL returns a row with the extension name `vector`.

The development environment was verified using PostgreSQL `17.12` and pgvector `0.8.7`.

### 6. Environment Variables

Create a `.env` file in the root directory of the project:

```text
airline-feedback-triage/
├── backend/
├── frontend/
├── data/
└── .env
```

Add the following environment variables:

```dotenv
OPENAI_API_KEY=YOUR_OPENAI_API_KEY
OPENAI_MODEL=gpt-4o-mini
DATABASE_URL=postgresql+psycopg2://airline_user:YOUR_SECURE_PASSWORD@localhost:5433/airline_feedback
```

Replace the placeholders with your own OpenAI API key and PostgreSQL password.

The `DATABASE_URL` connects the Python backend to PostgreSQL running in Docker.

**Important:** If your PostgreSQL password contains special characters such as `@`, `:`, `/`, or `#`, URL-encode those characters in the connection string.

Never commit your `.env` file or API credentials to GitHub. Make sure `.env` is listed in `.gitignore`.

### 7. Python Environment & Dependencies

The project uses a Python virtual environment to manage dependencies.

The following commands are intended for Windows PowerShell.

**Create a virtual environment**

From the project root directory, run:

```powershell
python -m venv .venv
```

**Activate the virtual environment**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Install the required packages**

```powershell
python -m pip install -r backend/requirements.txt
```

The requirements file includes the dependencies needed for the FastAPI backend, OpenAI integration, PostgreSQL connection, embedding storage, Streamlit dashboard, and evaluation scripts.

**Check installed dependencies**

```powershell
python -m pip check
```

A successful dependency check should return:

```text
No broken requirements found.
```

**Note:** The dependency versions were verified in the development environment. Installation from scratch on a separate machine has not yet been tested.

### 8. Running the FastAPI Backend

Before starting the backend, make sure:

- Docker Desktop is running.
- The PostgreSQL container `airline-pgvector` is running.
- The pgvector extension is enabled.
- The `.env` file contains the required environment variables.
- The Python virtual environment is activated.
- The required Python packages are installed.

**Start the FastAPI server**

Open a terminal in the project root directory and run:

```powershell
python -m uvicorn backend.main:app --reload
```

The backend will be available at:

`http://127.0.0.1:8000`

**Open Swagger UI**

FastAPI provides interactive API documentation at:

`http://127.0.0.1:8000/docs`

Swagger UI allows users to explore the API endpoints, submit test requests, and inspect the responses.

**Verify the backend**

Open the following URL in your browser:

`http://127.0.0.1:8000/`

The root endpoint should return:

```json
{
  "status": "ok"
}
```

**Important:** Submitting a new tweet through `POST /comments` triggers OpenAI API requests for classification and embedding generation. These requests may incur API costs.

The `--reload` option is intended for local development.

## API Endpoints

The FastAPI backend provides six endpoints for processing, retrieving, and semantically searching airline customer feedback.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Check the API status |
| POST | `/comments` | Analyze a tweet, generate its embedding, and save the results |
| GET | `/comments` | Retrieve stored tweets and their analyses |
| GET | `/comments/search` | Rank stored tweets by semantic similarity to a search query |
| GET | `/comments/{comment_id}` | Retrieve a specific tweet and its analyses |
| GET | `/analyses` | Retrieve stored AI analysis results |

Interactive API documentation is available through Swagger UI:

`http://127.0.0.1:8000/docs`

### Semantic Search API

The endpoint `GET /comments/search` accepts a free-text query and returns the nearest stored tweets based on semantic similarity.

Example:

```text
GET /comments/search?query=lost%20baggage&limit=5
```

- `query`: Required search text (at least two characters; whitespace-only queries are rejected).
- `limit`: Maximum number of results (default: 5, minimum: 1, maximum: 50).
- The backend generates an embedding for the search phrase using `text-embedding-3-small`.
- PostgreSQL with pgvector ranks existing tweet embeddings by **cosine distance**.
- The response returns stored tweet details and related analyses, ordered from nearest to more distant matches.
- The search does **not** create new tweets or modify stored records.

**Limitations:** The current version has no minimum similarity threshold and does not return a similarity score. Therefore, the nearest results are not guaranteed to be relevant. Searches require an OpenAI embedding API call, which may incur usage costs. Only tweets with stored embeddings can be returned.

**Observed local example (November 18, 2026):** A search for `lost baggage` ranked the synthetic test tweet `semantic_test_001` first. The tweet stated, “My suitcase never arrived at the airport after my flight. I have no idea where my belongings are.” This demonstrates one successful cross-wording match; it is not a formal retrieval-quality evaluation.

### Example: Analyze a New Tweet

The main endpoint is:

`POST /comments`

It receives one tweet and automatically performs the following operations:

1. Validates the incoming data.
2. Checks whether the tweet already exists.
3. Sends the tweet to OpenAI for multi-label classification and sentiment analysis.
4. Generates a 1536-dimensional embedding.
5. Stores the tweet and embedding in the `comments` table.
6. Stores the AI analysis in the `analyses` table.
7. Returns the saved tweet and its analysis results.

### Example Request

The following JSON represents an illustrative tweet:

```json
{
  "tweet_id": "demo_tweet_001",
  "airline_sentiment": "negative",
  "airline": "United",
  "name": "Demo User",
  "text": "My flight was delayed and I need help changing my booking.",
  "tweet_created": "2026-11-16T10:30:00"
}
```

The fields `tweet_id` and `text` are required.

The remaining fields are optional.

Classification and sentiment predictions are generated by the AI model and are not supplied by the API user.

The `airline_sentiment` field represents an existing dataset label, not the AI-generated sentiment.

### Example Response Structure

A successful request returns HTTP `201 Created`.

The response contains the stored tweet information and an `analyses` array.

The following example is illustrative and does not represent a recorded API response:

```json
{
  "id": 1,
  "tweet_id": "demo_tweet_001",
  "airline_sentiment": "negative",
  "airline": "United",
  "name": "Demo User",
  "text": "My flight was delayed and I need help changing my booking.",
  "tweet_created": "2026-11-16T10:30:00",
  "created_at": "2026-11-16T10:30:05",
  "analyses": [
    {
      "id": 1,
      "comment_id": 1,
      "model": "gpt-4o-mini",
      "prompt_version": "v4",
      "classification": [
        "complaint",
        "request"
      ],
      "sentiment": "negative",
      "tokens_used": 472,
      "created_at": "2026-11-16T10:30:05"
    }
  ]
}
```

The IDs, timestamps, token count, and classification results above are examples. Actual results depend on the request and model output.

The embedding is stored in PostgreSQL but is not included in the API response.

### Error Handling

The API uses the following HTTP status codes:

| Status Code | Description |
|---|---|
| `201` | Tweet successfully analyzed and stored |
| `409` | Duplicate tweet or database constraint violation |
| `422` | Invalid input data |
| `502` | AI analysis or embedding generation failure |
| `500` | Database error during storage |
| `404` | Requested comment not found |

The system uses PostgreSQL transactions to prevent incomplete tweet and analysis records from being saved when a database operation fails.

### Pagination

The endpoints `GET /comments` and `GET /analyses` support pagination through two parameters:

- `skip`: Number of records to skip (default: 0).
- `limit`: Maximum number of records to return (default: 50, maximum: 100).

Example:

`GET /comments?skip=0&limit=10`

## Running the Streamlit Dashboard

The project includes an interactive dashboard built with Streamlit.

The dashboard provides semantic search, CSV upload and batch processing, classification filtering, visualizations of customer feedback stored in PostgreSQL, and historical evaluation results from CSV files.

### 1. Start the Dashboard

Make sure that:

- Docker Desktop and the PostgreSQL container are running.
- The FastAPI backend is running at `http://127.0.0.1:8000` for semantic search and CSV processing.
- The `.env` file contains the correct database connection settings.
- The Python virtual environment is activated.
- The required dependencies are installed.

Open a terminal in the project root directory and run:

```powershell
python -m streamlit run frontend/dashboard.py
```

Streamlit normally opens the dashboard automatically in your web browser.

The default local address is:

`http://localhost:8501`

### 2. Dashboard Features

The dashboard includes the following components:

**Operational Overview**

- Total number of analyzed tweets
- Classification distribution
- Sentiment distribution
- Interactive table of analyzed tweets
- Filter by `All`, `Complaint`, `Compliment`, or `Request`
- Support for multi-label classification (a tweet may appear under multiple filters)

**Semantic Search**

- Enter a natural-language search phrase and click **Search**.
- Choose the maximum number of results (3, 5, 10, or 20 in the dashboard).
- View ranked matching tweets with their classification and sentiment.
- Search requests are sent to the FastAPI `GET /comments/search` endpoint.

**CSV Upload & Batch Processing**

- Upload a `.csv` file using **Browse files**.
- Preview the first ten rows and click **Start Processing**.
- Each row is submitted individually to `POST /comments` for classification, sentiment analysis, embedding generation, and database storage.
- View a progress bar during processing and a summary of processed, successful, duplicate, and failed rows.
- Processing stops early on serious backend errors or connection failures; progress reaching 100% indicates that all rows were attempted, not necessarily that they all succeeded.

The CSV requires `tweet_id` and `text` columns. Optional columns include `airline_sentiment`, `airline`, `name`, and `tweet_created`. The uploader reads the file in memory; it does not require saving a permanent copy into the project. Keep input batches small for demonstrations because each new tweet may trigger paid OpenAI API calls.

**Model & Prompt Evaluation**

- Comparison of six model/prompt combinations
- Sentiment Accuracy
- Classification Accuracy
- Macro F1-score for classification and sentiment
- Average API latency
- Input and output token usage
- Estimated API costs

### 3. Data Sources

The dashboard uses two data sources:

**PostgreSQL Database**

The operational dashboard retrieves analyzed tweets and their AI-generated results from PostgreSQL.

Classification and sentiment are retrieved from the `analyses` table.

The system supports multiple classification labels for a single tweet.

**CSV Evaluation Files**

Model comparison results are loaded from CSV files in the project's `data/` directory. These historical evaluation inputs are separate from the CSV files uploaded for operational processing.

These files contain evaluation results, latency measurements, token usage, and estimated costs.

### 4. Evaluation Limitations

The evaluation dashboard displays results from experiments conducted using Prompt Versions `v2` and `v3`.

These experiments were performed before the introduction of multi-label classification with Prompt Version `v4`.

The evaluation dataset contains 100 tweets.

Classification evaluation was conducted on 84 tweets with preliminary reference labels. The remaining 16 tweets were marked as `needs_review`.

Latency, token usage, and cost comparisons were conducted using the same 20 tweets for each of the six model/prompt combinations.

Therefore, the displayed evaluation results should not be interpreted as a direct performance evaluation of the current multi-label Prompt Version `v4`.

### 5. Dashboard Purpose

The Streamlit dashboard provides a visual overview of customer feedback analysis, interactive classification filtering, semantic search, CSV ingestion, and comparison of different AI models and prompt engineering approaches.

It separates operational data stored in PostgreSQL from experimental evaluation results stored in CSV files.

## Model Evaluation & Experimental Results

To evaluate the performance of the AI classification system, six combinations of OpenAI models and prompt versions were tested.

### Evaluation Setup

The evaluation was conducted using a dataset of **100 United Airlines tweets**.

Three OpenAI models were evaluated:

- `gpt-4o-mini`
- `gpt-4.1-mini`
- `gpt-5-mini`

Each model was tested with two prompt versions: `v2` and `v3`.

The evaluation focused on two tasks:

**Sentiment Analysis**

Sentiment predictions were evaluated against the existing sentiment labels in the dataset.

All 100 tweets were used for sentiment evaluation.

**Classification**

Classification predictions were compared against manually assigned preliminary reference labels.

Of the 100 tweets, 84 were used for classification evaluation. The remaining 16 tweets were marked as `needs_review` and excluded from classification metric calculations.

### Evaluation Metrics

The following metrics were calculated:

- **Accuracy:** Percentage of correct predictions.
- **Macro F1-score:** Unweighted average of the F1-scores across the three classes.
- **Token Usage:** Total number of tokens consumed during evaluation.

### Experimental Results

| Model | Prompt | Sentiment Accuracy | Sentiment Macro F1 | Classification Accuracy | Classification Macro F1 |
|---|---|---|---|---|---|
| gpt-4o-mini | v2 | 89.00% | 0.8824 | 89.29% | 0.8920 |
| gpt-4o-mini | v3 | 91.00% | 0.9055 | 90.48% | 0.9037 |
| gpt-4.1-mini | v2 | 86.00% | 0.8479 | 82.14% | 0.8203 |
| gpt-4.1-mini | v3 | 89.00% | 0.8821 | 91.67% | 0.9157 |
| gpt-5-mini | v2 | 86.00% | 0.8515 | 92.86% | 0.9289 |
| gpt-5-mini | v3 | 91.00% | 0.9064 | 91.67% | 0.9170 |

### Token Usage

The following token consumption was recorded for 100 tweets per model/prompt combination.

| Model | Prompt | Total Tokens |
|---|---|---:|
| gpt-4o-mini | v2 | 38,885 |
| gpt-4o-mini | v3 | 44,784 |
| gpt-4.1-mini | v2 | 38,882 |
| gpt-4.1-mini | v3 | 44,791 |
| gpt-5-mini | v2 | 55,312 |
| gpt-5-mini | v3 | 61,275 |

### Additional Performance Experiments

A separate experiment was conducted to compare API latency and token usage.

For this experiment, the same 20 tweets were processed using each of the six model/prompt combinations.

A total of 120 successful requests were recorded.

The Streamlit dashboard visualizes:

- Average API latency
- Input token usage
- Output token usage
- Estimated API costs

### Evaluation Limitations

The classification results should be considered **preliminary**, because the reference labels have not undergone a complete independent validation process.

The evaluation dataset contains only 100 tweets from United Airlines and may not represent the full diversity of airline customer feedback.

Furthermore, these experiments were conducted using single-label classification with Prompt Versions `v2` and `v3`.

The current backend uses Prompt Version `v4`, which supports multi-label classification.

Therefore, the experimental results presented above do not measure the performance of the current multi-label classification system.

A separate multi-label evaluation would be required to assess Prompt Version `v4`.

### Evaluation Summary

The experiments demonstrate how different OpenAI models and prompt versions can produce different classification results, sentiment predictions, and token consumption.

The findings provide a basis for comparing model behavior, prompt engineering approaches, and computational costs.

All reported results are based on the evaluation datasets and experiment outputs generated during project development.

## Project Structure

The project is organized into separate directories for backend development, frontend visualization, datasets, experiments, evaluation, and testing.

```text
airline-feedback-triage/
│
├── backend/
│   ├── main.py
│   ├── analyzer.py
│   ├── analyzer_v3.py
│   ├── embedding.py
│   ├── database.py
│   ├── models.py
│   ├── schemas.py
│   ├── requirements.txt
│   │
│   ├── evaluation/
│   │   ├── evaluate.py
│   │   ├── evaluate_41.py
│   │   ├── evaluate_41_v2.py
│   │   ├── evaluate_5.py
│   │   ├── evaluate_5_v2.py
│   │   ├── evaluate_v3.py
│   │   ├── evaluate_latency.py
│   │   ├── evaluate_parameters.py
│   │   ├── evaluation_metrics.py
│   │   └── prepare_dataset.py
│   │
│   ├── experiments/
│   │   ├── analyzer_41.py
│   │   ├── analyzer_41_v2.py
│   │   ├── analyzer_5.py
│   │   ├── analyzer_5_v2.py
│   │   └── cost_comparison.py
│   │
│   ├── scripts/
│   │   └── load_data.py
│   │
│   └── tests/
│       ├── test_analyzer.py
│       ├── test_embedding_failure.py
│       ├── test_failure.py
│       └── test_rollback.py
│
├── frontend/
│   └── dashboard.py
│
├── data/
│   ├── Tweets_clean.csv
│   ├── evaluation_dataset.csv
│   ├── evaluation_dataset_labeled_01_100_draft.csv
│   ├── evaluation_latency_results.csv
│   ├── evaluation_parameter_results.csv
│   ├── evaluation_results_41_v2.csv
│   ├── evaluation_results_41_v3.csv
│   ├── evaluation_results_5_v2.csv
│   ├── evaluation_results_5_v3.csv
│   ├── evaluation_results_v2.csv
│   ├── evaluation_results_v3.csv
│   └── original_Tweets.csv
│
├── .gitignore
├── .env                  # Local configuration (not committed)
└── README.md
```

### Directory Overview

| Directory | Purpose |
|---|---|
| `backend/` | FastAPI application, AI analysis, embeddings, and database integration |
| `backend/evaluation/` | Model evaluation and performance measurement scripts |
| `backend/experiments/` | Alternative model implementations and cost experiments |
| `backend/scripts/` | CSV batch-processing utility |
| `backend/tests/` | Backend validation and regression tests |
| `frontend/` | Streamlit dashboard |
| `data/` | Tweet datasets and experimental results |

## Future Improvements

The current system provides an end-to-end workflow for processing airline customer feedback, generating AI classifications and embeddings, and storing the results in PostgreSQL.

Potential future improvements include:

**1. Semantic Search Enhancements**

Extend the implemented embedding-based nearest-neighbor search with similarity scores, configurable relevance thresholds, and systematic retrieval-quality evaluation.

**2. Retrieval-Augmented Generation (RAG)**

Extend semantic search with a RAG workflow that retrieves relevant customer feedback and uses an LLM to generate context-aware summaries or answers.

RAG is a potential extension and is **not part of the current implementation**.

**3. Multi-label Evaluation**

Create a manually validated multi-label reference dataset and evaluate the current Prompt Version `v4` using appropriate multi-label metrics.

**4. Automated Testing and CI**

Integrate the existing regression tests into an automated continuous integration workflow.

**5. Deployment**

Prepare the application for deployment outside the local development environment, including secure configuration management and database backups.

## Project Status

The core application has been implemented and tested locally.

The current version supports:

- FastAPI-based tweet processing
- OpenAI-powered multi-label classification and sentiment analysis
- Embedding generation and PostgreSQL/pgvector storage
- Duplicate prevention and error handling
- CSV batch processing through Python and the Streamlit uploader
- Semantic search through FastAPI and Streamlit (tested locally with a synthetic example)
- Classification filtering in the Streamlit dashboard
- Streamlit dashboard visualizations
- Model and prompt evaluation
- Backend regression testing

During the November 18, 2026 local demonstration, a nine-row CSV upload reported 9 processed, 9 successful, 0 duplicate, and 0 failed requests. The classification filter returned 13 tweets for `All`, 12 for `Complaint`, 1 for `Compliment`, and 6 for `Request` at that time. These counts describe the test database, not the complete 3,822-tweet source dataset.

The system is currently configured for local development using Docker Desktop and PostgreSQL. Screen recording, presentation material, and additional documentation are project-delivery activities rather than API features.