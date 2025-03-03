# api.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import os
from dotenv import load_dotenv

# Importer notre système RAG
from rag_system import create_rag_system, BudgetRAG

# Charger les variables d'environnement
load_dotenv()

# Initialiser le système RAG
csv_path = os.getenv("CSV_PATH", "./data/transactions.csv")
persist_directory = os.getenv("PERSIST_DIRECTORY", "./db")

# Créer l'application FastAPI
app = FastAPI(title="BudgetRAG API", description="API pour l'assistant budgétaire RAG")

# Configurer CORS pour permettre les requêtes du frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Origine du frontend React
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modèles de données
class QuestionRequest(BaseModel):
    user_uuid: str
    question: str

class QuestionResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

class UserExistsRequest(BaseModel):
    user_uuid: str

class UserExistsResponse(BaseModel):
    exists: bool

# Singleton pour le système RAG
rag_system: Optional[BudgetRAG] = None

def get_rag_system() -> BudgetRAG:
    """
    Retourne l'instance du système RAG, l'initialise si nécessaire
    """
    global rag_system
    if rag_system is None:
        print("Initialisation du système RAG...")
        rag_system = create_rag_system(csv_path, persist_directory)
    return rag_system

# Routes API
@app.get("/")
def read_root():
    """
    Route racine pour l'API
    """
    return {
        "message": "Bienvenue sur l'API BudgetRAG",
        "version": "1.0.0",
        "documentation": "/docs"
    }

@app.post("/api/check-user", response_model=UserExistsResponse)
def check_user(request: UserExistsRequest, rag: BudgetRAG = Depends(get_rag_system)):
    """
    Vérifie si un utilisateur existe dans la base de données
    """
    exists = request.user_uuid in rag.user_ids
    return UserExistsResponse(exists=exists)

@app.post("/api/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest, rag: BudgetRAG = Depends(get_rag_system)):
    """
    Pose une question au système RAG pour un utilisateur spécifique
    """
    # Vérifier si l'utilisateur existe
    if request.user_uuid not in rag.user_ids:
        raise HTTPException(
            status_code=404,
            detail=f"Utilisateur {request.user_uuid} non trouvé dans la base de données."
        )
    
    # Initialiser le retriever pour l'utilisateur
    if not rag.initialize_retriever_for_user(request.user_uuid):
        raise HTTPException(
            status_code=500,
            detail="Erreur lors de l'initialisation du retriever pour l'utilisateur."
        )
    
    # Poser la question
    result = rag.ask_question(request.question)
    
    return QuestionResponse(
        answer=result["answer"],
        sources=result["sources"]
    )

@app.get("/api/users", response_model=List[str])
def get_users(rag: BudgetRAG = Depends(get_rag_system)):
    """
    Retourne la liste des UUID utilisateurs disponibles
    Note: Dans un environnement de production, cette route devrait être sécurisée
    """
    return list(rag.user_ids)

if __name__ == "__main__":
    # Lancer le serveur API
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)