# InteroperAI-ResQUE
## Intelligent Disaster Data Interoperability and Emergency Response Coordination Platform

**Member 2 — Intelligent Disaster Assessment and Emergency Response Intelligence**

**Developed by:** Salva Soumya and Gilledi Santhosh

---

## 🌐 Live Deployment

| Component | Deployment Link |
|---|---|
| Member 2 Frontend | [Open Application](https://interoper-ai-res-que-member2.vercel.app/) |
| Member 2 Backend API | [Open Backend](https://interoperai-member2-api.onrender.com) |
| Member 2 API Documentation | [Swagger UI](https://interoperai-member2-api.onrender.com/docs) |
| Member 1 Frontend | [Open Application](https://interoper-ai-res-que-member1-29p4.vercel.app/) |
| Member 1 Backend API | [Open Backend](https://interoperai-member1-api.onrender.com) |
| Member 1 API Documentation | [Swagger UI](https://interoperai-member1-api.onrender.com/docs) |

## 📌 Project Overview

Member 2 is the disaster-assessment and operational-intelligence component of InteroperAI-ResQUE.

It receives standardized disaster-related datasets from Member 1, records incoming files, and provides a dashboard for viewing incident assessments, impact indicators, shelter information, resource availability, anomalies, and ingestion history supported by the backend.

The application combines API-based data ingestion, backend processing, saved machine-learning artifacts, and an interactive frontend to support disaster-response data analysis.

## 🎯 Objectives

- Receive standardized disaster datasets from Member 1.
- Save and track incoming files and ingestion transactions.
- Provide access to incident-assessment and operational information.
- Identify unusual records using Isolation Forest when the model is available and successfully loaded.
- Present shelter capacity, resource availability, and supply shortages.
- Provide provenance and data-quality traceability.
- Expose API endpoints for the frontend and Member 1 integration.

## ✨ Key Features

- **API-Based File Reception:** Receive files through a REST API.
- **Ingestion Tracking:** Record received files and associated job information.
- **Disaster Assessment:** Present incident impact and priority information supported by the backend.
- **Anomaly Detection:** Use the configured Isolation Forest model when available.
- **Shelter Management:** Display shelter capacity and occupancy information.
- **Resource Management:** Present resource availability and supply shortages.
- **Analytics and Recommendations:** Provide operational views supported by backend endpoints.
- **Assessment Provenance:** Show source-file, source-system, job, and assessment information.
- **Operational Audit Trail:** Present available ingestion and activity information.

## 🏗️ System Architecture

```text
                 MEMBER 1
       Data Integration Application
                      |
                      v
            Standardized Dataset
                      |
           HTTP POST / File Upload
                      |
                      v
             Member 2 FastAPI
                  (Render)
                      |
           +----------+----------+
           |          |          |
           v          v          v
        Receive    Process     Load ML
         File       Data       Artifacts
           |          |          |
           +----------+----------+
                      |
                      v
           Assessment and Operational
                  Data APIs
                      |
                      v
           React + Vite Frontend
                  (Vercel)
                      |
                      v
              User Dashboard
```

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| React | Interactive frontend and dashboard |
| Vite | Frontend development and production build |
| Tailwind CSS | UI styling |
| Python | Backend processing and application logic |
| FastAPI | REST API endpoints |
| Uvicorn | ASGI server |
| Pandas | Tabular data ingestion and processing |
| NumPy | Numerical operations |
| scikit-learn | Machine-learning functionality, including Isolation Forest |
| Joblib | Loading and saving serialized model artifacts |
| SQLite | Application and ingestion persistence |
| Vercel | Frontend deployment |
| Render | Backend deployment |

## 🔄 Implementation and Working

### 1. Receiving Data from Member 1

Member 2 exposes a file-receiving API endpoint:

`POST https://interoperai-member2-api.onrender.com/api/receive-file`

The endpoint accepts a multipart form request with a required `file` field. Optional headers include source-system, client, and correlation identifiers.

### 2. File Ingestion

The backend receives the file and saves it into its ingestion workflow. The received filename and job information can be used to trace the incoming transaction.

### 3. Backend Processing

After receiving a file, the backend performs the processing supported by its current configuration. The resulting information is exposed through API endpoints for the frontend.

### 4. Machine-Learning Integration

The project includes an Isolation Forest model artifact for anomaly detection. The model must be available at the path expected by the backend and load successfully before its results can be used.

Other model artifacts should only be reported as active when the backend actually loads and uses them. Model availability, execution status, and output should be based on the application's actual runtime results.

### 5. Dashboard Visualization

The React frontend requests data from the backend and displays operational information such as incident impact, shelter capacity, resource shortages, assessment information, and recent ingestion transactions.

### 6. Provenance and Traceability

The dashboard includes assessment provenance and data-quality information. Fields such as source file, source system, assessment ID, job ID, and ingestion timestamp help users trace the origin and processing of a dataset.

### 7. End-to-End Integration

The complete workflow is:

1. Member 1 uploads and validates a dataset.
2. Member 1 standardizes the dataset.
3. Member 1 transfers the standardized file to Member 2.
4. Member 2 receives and saves the file.
5. Member 2 runs the processing available in its backend.
6. The Member 2 dashboard retrieves and displays available results.

## 🔗 API Documentation

**Member 2 Swagger UI:**

https://interoperai-member2-api.onrender.com/docs

### File Reception

| Property | Value |
|---|---|
| HTTP Method | `POST` |
| Endpoint | `/api/receive-file` |
| Content Type | `multipart/form-data` |
| Required field | `file` |
| Purpose | Receive a dataset from Member 1 |

Optional request headers documented by the API:

- `X-Source-System`
- `X-Client-Id`
- `X-Correlation-Id`

Use the Swagger documentation to inspect the current API schema and response structure.

## 🚀 Local Setup

### Prerequisites

- Python
- Node.js and npm
- Git

### Backend Setup

Open PowerShell in the repository root:

```powershell
cd backend
.\venv\Scripts\Activate.ps1
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8001
```

If the virtual environment does not exist, create it and install the backend dependencies:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Local backend URLs:

- API: `http://127.0.0.1:8001`
- Swagger documentation: `http://127.0.0.1:8001/docs`

### Frontend Setup

Open a separate terminal:

```powershell
cd frontend
npm install
npm run dev
```

Vite usually serves the frontend at `http://localhost:5173`. If that port is occupied, it may select another available port.

### Frontend Environment Variable

The frontend reads the API base URL from `VITE_API_BASE_URL`. For the deployed frontend, configure:

```text
VITE_API_BASE_URL=https://interoperai-member2-api.onrender.com
```

The fallback URL `http://127.0.0.1:8001` is intended for local development.

## ☁️ Deployment Configuration

### Frontend — Vercel

| Setting | Value |
|---|---|
| Root Directory | `frontend` |
| Framework | Vite |
| Build Command | `npm run build` |
| Output Directory | `dist` |

Environment variable:

```text
VITE_API_BASE_URL=https://interoperai-member2-api.onrender.com
```

### Backend — Render

| Setting | Value |
|---|---|
| Root Directory | `backend` |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
| Branch | `main` |

Ensure that required model artifacts are included in the deployment or loaded from a configured artifact store. A file on a developer's computer is not automatically available to Render.

## 🧠 Model Artifacts and Reliability

The project contains saved machine-learning artifacts used by its assessment workflow.

- **Isolation Forest:** Used for anomaly detection when the expected artifact is available and loads successfully.
- **Other model artifacts:** Their usage depends on the implemented backend workflow and successful loading.
- **Artifact paths:** Must match the paths expected by the backend code.
- **Model compatibility:** Saved models should be compatible with the installed scikit-learn and related library versions.

If an artifact is missing or fails to load, the application should report the failure rather than claim that the model processed the data.

## 🧪 Testing and Verification

1. Open the Member 1 frontend.
2. Upload a supported disaster dataset.
3. Complete validation and standardization.
4. Send the dataset to Member 2.
5. Confirm that the receiving endpoint accepts the file.
6. Open the Member 2 dashboard.
7. Check the latest ingestion transaction, source-file information, and available assessment results.
8. Review the Render logs if a request fails or processing does not complete.

## ⚠️ Operational Notes

- Free Render services may sleep after inactivity, delaying the first request.
- A successful file upload does not by itself guarantee that all downstream ML processing completed.
- Check Render logs for model-loading and processing errors.
- SQLite data stored on an ephemeral filesystem may not persist through restarts or redeployments. A persistent disk or managed database is recommended if durable production storage is required.
- Keep secrets, local `.env` files, virtual environments, backups, and unnecessary generated artifacts out of Git.

## 👥 Team

**Salva Soumya and Gilledi Santhosh**

*InteroperAI-ResQUE — Intelligent Disaster Data Interoperability and Emergency Response Coordination Platform*
