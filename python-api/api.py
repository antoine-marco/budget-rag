# api.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import os
import traceback
from dotenv import load_dotenv

# Importer notre système RAG mis à jour
from rag_system import create_rag_system, BudgetRAG

# Charger les variables d'environnement
load_dotenv()

# Initialiser le système RAG
csv_path = os.getenv("CSV_PATH", "../data/transactions.csv")
persist_directory = os.getenv("PERSIST_DIRECTORY", "../db")

# Vérifier les chemins
print(f"Chemin CSV configuré: {csv_path}")
print(f"Chemin absolu CSV: {os.path.abspath(csv_path)}")
print(f"Le fichier CSV existe: {os.path.exists(csv_path)}")

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
        try:
            rag_system = create_rag_system(csv_path, persist_directory)
        except Exception as e:
            error_msg = f"Erreur lors de l'initialisation du système RAG: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())
            raise RuntimeError(error_msg)
    return rag_system

# Gestionnaire d'exceptions global
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    error_msg = f"Une erreur inattendue s'est produite: {str(exc)}"
    print(error_msg)
    print(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": error_msg},
    )

# Route racine
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

# Routes API
@app.post("/api/check-user", response_model=UserExistsResponse)
def check_user(request: UserExistsRequest, rag: BudgetRAG = Depends(get_rag_system)):
    """
    Vérifie si un utilisateur existe dans la base de données
    """
    try:
        exists = request.user_uuid in rag.user_ids
        print(f"Vérification de l'utilisateur {request.user_uuid}: {exists}")
        return UserExistsResponse(exists=exists)
    except Exception as e:
        error_msg = f"Erreur lors de la vérification de l'utilisateur: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)

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
    
    try:
        # Poser la question
        result = rag.ask_question(request.question)
        
        # Vérification supplémentaire
        if not result.get("answer"):
            raise HTTPException(
                status_code=500,
                detail="Le système RAG n'a pas pu générer de réponse."
            )
            
        return QuestionResponse(
            answer=result["answer"],
            sources=result["sources"]
        )
    except Exception as e:
        error_msg = f"Erreur lors du traitement de la question: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/api/users", response_model=List[str])
def get_users(rag: BudgetRAG = Depends(get_rag_system)):
    """
    Retourne la liste des UUID utilisateurs disponibles
    Note: Dans un environnement de production, cette route devrait être sécurisée
    """
    return list(rag.user_ids)

# Route de santé
@app.get("/api/health")
def health_check():
    """
    Vérifie l'état de santé de l'API
    """
    return {
        "status": "healthy",
        "csv_path": csv_path,
        "csv_exists": os.path.exists(csv_path),
        "db_path": persist_directory,
        "db_exists": os.path.exists(persist_directory)
    }

if __name__ == "__main__":
    # Lancer le serveur API
    uvicorn.run("api_fixed:app", host="0.0.0.0", port=8000, reload=True)