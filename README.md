# MoJin RAG Project

This project provides a Retrieval-Augmented Generation (RAG) system with both backend and frontend components.

## Features
- Backend API for document processing and retrieval
- Frontend interface for querying and viewing results
- Containerized with Docker for easy deployment

## Getting Started

### Prerequisites
- Docker
- Docker Compose

### Installation
1. Clone the repository
2. Run the system: `docker-compose up --build`
3. Access the frontend at http://localhost:3000

### Project Structure
```
mojin_rag_project/
├── backend/          # Python backend service
│   ├── app.py        # Main application
│   ├── Dockerfile    # Backend container configuration
│   └── requirements.txt
├── frontend/         # React frontend
│   ├── public/
│   ├── src/
│   ├── Dockerfile    # Frontend container configuration
│   └── package.json
├── docker-compose.yml # Orchestration configuration
└── README.md         # This file
```

## Configuration
Modify environment variables in `docker-compose.yml` for your specific setup.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
