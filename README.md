# InteroperAI-ResQUE
## Intelligent Disaster Data Interoperability and Emergency Response Coordination Platform

**Component:** Member 2 — Intelligent Disaster Assessment and Emergency Response Intelligence  
**Team:** Salva Soumya and Gilledi Santhosh

---

## 🌐 Live Deployment

| Component | Link |
|---|---|
| **Member 2 Frontend** | [Open Dashboard](https://interoper-ai-res-que-member2.vercel.app/) |
| **Member 2 Backend API** | [Open API](https://interoperai-member2-api.onrender.com) |
| **Member 2 API Documentation** | [Swagger UI](https://interoperai-member2-api.onrender.com/docs) |
| Member 1 Frontend | [Open Data Integration Platform](https://interoper-ai-res-que-member1-29p4.vercel.app/) |
| Member 1 Backend API | [Open API](https://interoperai-member1-api.onrender.com) |
| Member 1 API Documentation | [Swagger UI](https://interoperai-member1-api.onrender.com/docs) |

---

## 📌 Project Overview

InteroperAI-ResQUE is an intelligent disaster-response platform designed to improve the interoperability of disaster-related data and support emergency-response assessment.

Member 2 is responsible for receiving standardized datasets from Member 1 and presenting disaster-assessment and operational information through an interactive dashboard. Its backend supports file ingestion, incident assessment, anomaly-detection workflows, shelter and resource information, and ingestion traceability.

The system combines API-based data exchange, data processing, machine-learning components, and operational visualization to help users understand incoming disaster information and identify potential response priorities.

## 🎯 Objectives

- Receive standardized disaster data from Member 1.
- Maintain traceability of received files and ingestion jobs.
- Assess incident impact using available disaster-related information.
- Identify unusual records through machine-learning-based anomaly detection.
- Present shelter capacity, occupancy, resource availability, and shortages.
- Support operational analysis through dashboard views and assessment results.
- Provide meaningful source, job, and ingestion information for auditing.
- Integrate with Member 1 through a REST API.

## ✨ Key Features

### 1. Disaster Assessment Dashboard
Displays incident-assessment information, impact indicators, priority summaries, and operational statistics available from the backend.

### 2. API-Based Data Ingestion
Receives standardized disaster datasets from Member 1 through a file-upload API.

### 3. Machine Learning Integration
Includes anomaly-detection and supervised-classification work developed during the project. The live backend uses only models that are successfully loaded and integrated into its processing workflow.

### 4. Shelter Management
Displays shelter capacity and occupancy information available to the application.

### 5. Resource Management
Presents available resources, tracked resource categories, and reported shortages.

### 6. Analytics and Recommendations
Provides operational views and recommendations supported by the application's available backend data and logic.

### 7. Assessment Provenance
Displays information such as the source file, source system, assessment identifier, job identifier, and record-processing details.

### 8. Ingestion and Activity Tracking
Provides access to recent ingestion transactions and available operational activity logs.

---

## 🏗️ System Architecture

```text
                  MEMBER 1
       Disaster Data Integration Platform
                      |
                      v
           Standardized Disaster Dataset
                      |
                      v
          HTTP POST /api/receive-file
                      |
                      v
             MEMBER 2 BACKEND
                 FastAPI
                 (Render)
                      |
           +----------+-----------+
           |          |           |
           v          v           v
        Receive    Data and      Load
         File      Assessment   Available
                   Processing   ML Models
           |          |           |
           +----------+-----------+
                      |
                      v
          Assessment and Operational APIs
                      |
                      v
            MEMBER 2 FRONTEND
                React + Vite
                 (Vercel)
                      |
           +----------+-----------+
           |          |           |
           v          v           v
        Incident    Shelters    Resources
        Dashboard   and Beds    and Shortages
                      |
                      v
           Analytics, Results and
           Ingestion Traceability
```

## 🛠️ Technology Stack

| Category | Technology | Purpose |
|---|---|---|
| Frontend | React | Interactive dashboard and application interface |
| Frontend tooling | Vite | Development server and production build |
| UI styling | Tailwind CSS | Responsive layouts and visual styling |
| Backend language | Python | Application logic and data processing |
| Backend framework | FastAPI | REST API development |
| API server | Uvicorn | Runs the FastAPI application |
| Data processing | Pandas | Tabular dataset processing |
| Numerical computing | NumPy | Numerical operations |
| Machine learning | scikit-learn | Anomaly detection and supervised-learning experiments |
| Model persistence | Joblib | Saving and loading trained model artifacts |
| Database | SQLite | Application and ingestion data persistence |
| API communication | HTTP multipart upload | Transfers files from Member 1 to Member 2 |
| Frontend deployment | Vercel | Hosts the web frontend |
| Backend deployment | Render | Hosts the API service |

---

## 🧠 Machine Learning Models and Approaches

The project explores multiple machine-learning models and approaches for disaster-data assessment. Model training, artifact availability, deployment, and live execution are distinct stages; therefore, the dashboard should report results only from models that actually execute.

### 1. Isolation Forest — Anomaly Detection

Isolation Forest is an unsupervised machine-learning algorithm used to identify observations that differ from typical data patterns.

**Purpose in the project:**
- Identify potentially unusual disaster records.
- Generate anomaly predictions and scores when supported by the active pipeline.
- Supplement rule-based impact indicators with a data-driven anomaly signal.

**Artifact:** `isolation_forest_model.joblib`

The deployed backend has reported a missing-artifact warning for this file in its expected location in the past. The model must be available at the configured path and load successfully for live anomaly detection to run.

### 2. Hybrid Isolation Forest — Hybrid Anomaly Detection

The project also contains a hybrid anomaly-detection implementation and associated model artifact.

**Artifact:** `hybrid_isolation_forest.joblib`

This approach represents additional anomaly-detection work beyond the primary Isolation Forest pipeline. Its active use depends on whether the current backend loads and executes the artifact.

### 3. Random Forest Classifier — Supervised Learning Experiment

Random Forest was explored as a supervised classification approach for disaster-priority assessment.

The earlier benchmark did not outperform the majority-class baseline. Consequently, its performance should be evaluated further before it is selected for production use.

### 4. HistGradientBoosting — Supervised Learning Experiment

HistGradientBoosting was explored as another classification candidate. Its suitability depends on evaluation results, feature preparation, and whether it has been integrated into the active backend.

### 5. Priority Classification Candidate

The project also includes work on a saved priority-classification candidate.

**Artifact previously identified:** `priority_classifier_candidate.joblib`

The presence of a saved candidate does not establish that it is currently deployed or used to produce dashboard results. Its loading path and inference workflow must be verified before it is described as active.

### Model Lifecycle and Reliability

The project distinguishes the following model states:

| State | Meaning |
|---|---|
| Trained | The model was fitted during experimentation. |
| Saved | A serialized model artifact was generated. |
| Deployed | The artifact is available in the deployed environment. |
| Loaded | The backend successfully loaded the model and its dependencies. |
| Active | The backend actually executes the model in its processing workflow. |
| Experimental | The model was explored or evaluated but is not confirmed as active in production. |

For reproducible inference, the expected preprocessing steps, feature ordering, and compatible library versions must be preserved alongside each model.

---

## 🔄 Implementation and Working

### Step 1: Receive the Dataset

Member 1 sends a standardized disaster dataset to the Member 2 backend using an HTTP `POST` request.

The receiving endpoint accepts a file as multipart form data.

### Step 2: Save and Track the Incoming File

The backend saves the received dataset through its ingestion workflow. The associated filename and job information help identify the transaction.

### Step 3: Process the Incoming Data

The backend processes the received dataset using the data-handling and assessment logic available in the current implementation.

### Step 4: Execute Available ML Components

The backend can execute its integrated machine-learning components when their artifacts, preprocessing dependencies, and expected input features are available.

The application should distinguish between an ML prediction, a rule-based assessment, and a model that could not be executed.

### Step 5: Prepare Assessment Information

The backend exposes available incident-assessment results and operational information through API endpoints.

### Step 6: Display the Dashboard

The Member 2 frontend requests backend data and displays information such as:

- Assessed records
- High-impact incidents
- Incident priority summaries
- Anomaly indicators
- Shelter capacity and occupancy
- Resource shortages
- Recent ingestion transactions
- Assessment provenance and data-quality information

The displayed values depend on the current dataset and the results returned by the backend.

### Step 7: Verify the End-to-End Workflow

After sending data from Member 1, check Member 2's ingestion history and assessment information. If the upload succeeds but model processing fails, inspect the backend logs and model-loading status.

---

## 🔗 API Documentation and Integration

**Member 2 Swagger UI:**

[https://interoperai-member2-api.onrender.com/docs](https://interoperai-member2-api.onrender.com/docs)

### File Reception Endpoint

| Property | Value |
|---|---|
| HTTP method | `POST` |
| Endpoint | `/api/receive-file` |
| Content type | `multipart/form-data` |
| Required form field | `file` |
| Purpose | Receive a dataset from Member 1 |

**Full endpoint:**

```text
https://interoperai-member2-api.onrender.com/api/receive-file
```

The API documentation also exposes optional headers for request identification:

- `X-Source-System`
- `X-Client-Id`
- `X-Correlation-Id`

These identifiers can help trace where a request originated and correlate related processing operations when supplied and recorded by the backend.

---

## 🚀 Local Installation and Setup

### Prerequisites

Install the following before running the application locally:

- Python
- Node.js and npm
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/SalvaSoumya/InteroperAI-ResQUE-Member2.git
cd InteroperAI-ResQUE-Member2
```

### 2. Set Up the Backend

Open PowerShell in the repository root:

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Start the backend:

```powershell
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8001
```

Local backend URLs:

- API: `http://127.0.0.1:8001`
- Swagger UI: `http://127.0.0.1:8001/docs`

### 3. Set Up the Frontend

Open a separate terminal:

```powershell
cd frontend
npm install
npm run dev
```

Vite usually starts at `http://localhost:5173`. If that port is already in use, Vite may choose another available port.

### 4. Configure the Frontend API URL

The frontend reads the backend URL from `VITE_API_BASE_URL`, with a local fallback of `http://127.0.0.1:8001`.

For the deployed frontend, configure the following environment variable in Vercel:

```text
VITE_API_BASE_URL=https://interoperai-member2-api.onrender.com
```

Redeploy the frontend after changing environment variables.

---

## ☁️ Deployment Configuration

### Frontend — Vercel

| Setting | Value |
|---|---|
| Root Directory | `frontend` |
| Framework Preset | Vite |
| Build Command | `npm run build` |
| Output Directory | `dist` |

**Environment variable:**

```text
VITE_API_BASE_URL=https://interoperai-member2-api.onrender.com
```

### Backend — Render

| Setting | Value |
|---|---|
| Repository | `InteroperAI-ResQUE-Member2` |
| Branch | `main` |
| Root Directory | `backend` |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn main:app --host 0.0.0.0 --port $PORT` |

Required model artifacts must be available at the paths expected by the backend. Files on a developer's local computer are not automatically available to Render.

---

## 🧪 Testing and Verification

1. Open the Member 1 frontend.
2. Upload a supported disaster dataset.
3. Complete the validation and standardization workflow.
4. Select **Standardize & Transfer to Member 2**.
5. Confirm that Member 2 accepts the file.
6. Open the Member 2 dashboard.
7. Inspect the recent ingestion transaction and source-file details.
8. Check available incident-assessment and resource information.
9. Review Render application logs if the upload or downstream processing fails.

A successful upload response confirms receipt by the endpoint, but it does not by itself prove that every downstream model completed successfully.

---

## ⚠️ Reliability and Operational Considerations

- **Cold starts:** Free Render services may sleep after inactivity, delaying the first request.
- **Model availability:** Ensure model artifacts exist at the expected deployed paths.
- **Preprocessing compatibility:** Model inference must use the same expected features and preprocessing used during training.
- **Error handling:** Log model-loading and inference failures clearly.
- **Storage persistence:** SQLite files stored on an ephemeral filesystem may not survive service restarts or redeployments. Use persistent storage or a managed database when durable production storage is required.
- **Security:** Do not commit credentials, secrets, local `.env` files, virtual environments, or unnecessary backup files.
- **Auditability:** Record source-system and correlation identifiers when available to improve request traceability.

---

## 🔮 Future Enhancements

- Add a detailed API request-monitoring view.
- Display received file previews and schema information.
- Show model execution status and results for each ingestion job.
- Improve anomaly explanations and incident-priority interpretability.
- Add durable database storage and stronger audit-log retention.
- Introduce more detailed validation and processing error reports.
- Expand monitoring for end-to-end transfer and assessment status.

---

## 👥 Team

**Salva Soumya and Gilledi Santhosh**

**Project:** InteroperAI-ResQUE — Intelligent Disaster Data Interoperability and Emergency Response Coordination Platform

**Member 2 focus:** Disaster data reception, assessment, machine-learning integration, and emergency-response operational intelligence.
