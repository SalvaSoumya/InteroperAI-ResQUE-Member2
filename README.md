# Intelligent Disaster Assessment and Resource Management System

Member 2 module for the InteroperAI-ResQ project.

## Run backend
cd backend
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn main:app --reload

Backend: http://127.0.0.1:8000
Swagger: http://127.0.0.1:8000/docs

## Run frontend
Open a second terminal:
cd frontend
npm install
npm run dev

Frontend: http://localhost:5173
