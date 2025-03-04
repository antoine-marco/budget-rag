# rag_system.py
import os
import pandas as pd
from typing import List, Dict, Any, Optional
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langsmith import Client
import numpy as np
from dotenv import load_dotenv
import traceback

# Charger les variables d'environnement
load_dotenv()

# Configuration LangSmith pour le debug
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "budget-rag-assistant"

# Vérification des chemins
print("\n==== VÉRIFICATION DES CHEMINS ====")
csv_path = os.getenv("CSV_PATH", "../data/transactions.csv")
persist_directory = os.getenv("PERSIST_DIRECTORY", "../db")

print(f"Chemin CSV configuré: {csv_path}")
print(f"Chemin absolu CSV: {os.path.abspath(csv_path)}")
print(f"Le fichier CSV existe: {os.path.exists(csv_path)}")

print(f"Répertoire de persistance configuré: {persist_directory}")
print(f"Chemin absolu du répertoire de persistance: {os.path.abspath(persist_directory)}")
print(f"Le répertoire de persistance existe: {os.path.exists(persist_directory)}")
print("================================\n")

client = Client()

class BudgetRAG:
    def __init__(self, csv_path: str, persist_directory: str = "db"):
        """
        Initialise le système RAG pour l'analyse budgétaire
        
        Args:
            csv_path: Chemin vers le fichier CSV des transactions
            persist_directory: Répertoire pour persister la base vectorielle
        """
        self.csv_path = csv_path
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None
        self.user_ids = set()
        
        print(f"BudgetRAG initialisé avec:")
        print(f"- CSV: {self.csv_path} (existe: {os.path.exists(self.csv_path)})")
        print(f"- Persistance: {self.persist_directory} (existe: {os.path.exists(self.persist_directory)})")
        
    def load_and_process_data(self, force_reload: bool = False) -> None:
        """
        Charge et prétraite les données de transactions
        
        Args:
            force_reload: Si True, force le rechargement des données même si la base vectorielle existe
        """
        # Vérifier si le fichier CSV existe
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Le fichier CSV {self.csv_path} n'existe pas! Chemin absolu: {os.path.abspath(self.csv_path)}")
        
        # Charger d'abord tous les UUID des utilisateurs depuis le CSV
        print(f"Chargement des utilisateurs depuis {self.csv_path}...")
        try:
            df = pd.read_csv(self.csv_path)
            print(f"Colonnes du CSV: {df.columns.tolist()}")
            
            # Vérifier la présence de la colonne user_uuid
            if 'user_uuid' not in df.columns:
                raise ValueError(f"La colonne 'user_uuid' n'existe pas dans le CSV! Colonnes disponibles: {df.columns.tolist()}")
            
            # Stocker tous les UUID des utilisateurs, même si on ne recharge pas la base vectorielle
            self.user_ids = set(df['user_uuid'].unique())
            print(f"CSV chargé avec succès. Nombre d'utilisateurs uniques: {len(self.user_ids)}")
            
            # Débogage - Vérifier quelques UUID
            if len(self.user_ids) > 0:
                print("Exemples d'UUIDs disponibles:")
                for i, uid in enumerate(list(self.user_ids)[:3]):
                    print(f"  {i+1}. {uid}")
        except Exception as e:
            raise RuntimeError(f"Erreur lors du chargement du CSV: {str(e)}")
        
        # Vérifier si la base vectorielle existe déjà et si on doit la recharger
        if os.path.exists(self.persist_directory) and not force_reload:
            print("Chargement de la base vectorielle existante...")
            try:
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                print(f"Base vectorielle chargée avec succès!")
                return
            except Exception as e:
                print(f"Erreur lors du chargement de la base vectorielle: {str(e)}")
                print("Recréation de la base vectorielle...")
        elif not os.path.exists(self.persist_directory):
            print(f"Le répertoire de persistance {self.persist_directory} n'existe pas. Création d'une nouvelle base vectorielle...")
        else:
            print("Force reload activé. Création d'une nouvelle base vectorielle...")
        
        print("Traitement des données de transactions pour la vectorisation...")
        # Convertir les montants en float (si nécessaire)
        if df['net_amount'].dtype == 'object':
            print("Conversion des montants en format numérique...")
            
            # Fonction de nettoyage qui gère différents formats de montants
            def clean_amount(amount_str):
                if pd.isna(amount_str):
                    return 0.0
                
                if isinstance(amount_str, (int, float)):
                    return float(amount_str)
                
                # Supprimer les espaces (séparateurs de milliers)
                cleaned = str(amount_str).replace(' ', '')
                # Remplacer les virgules par des points (séparateur décimal)
                cleaned = cleaned.replace(',', '.')
                
                try:
                    return float(cleaned)
                except ValueError:
                    print(f"AVERTISSEMENT: Impossible de convertir '{amount_str}' en nombre, utilisé 0.0 à la place")
                    return 0.0
            
            # Appliquer la fonction de nettoyage à la colonne
            df['net_amount'] = df['net_amount'].apply(clean_amount)
            print(f"Conversion terminée. Exemple de valeurs: {df['net_amount'].head(3).tolist()}")
        
        # Créer des documents pour chaque transaction avec métadonnées
        documents = []
        
        for user_id in self.user_ids:
            user_df = df[df['user_uuid'] == user_id]
            
            # Créer un document de résumé pour chaque utilisateur
            summary_stats = self._generate_user_summary(user_df)
            summary_text = f"Résumé des transactions pour l'utilisateur {user_id}:\n"
            summary_text += f"- Période: {summary_stats['start_date']} à {summary_stats['end_date']}\n"
            summary_text += f"- Nombre total de transactions: {summary_stats['total_transactions']}\n"
            summary_text += f"- Montant total dépensé: {summary_stats['total_spent']} €\n"
            summary_text += f"- Montant total reçu: {summary_stats['total_received']} €\n"
            summary_text += f"- Catégories les plus fréquentes: {', '.join(summary_stats['top_categories'])}\n"
            
            documents.append(Document(
                page_content=summary_text,
                metadata={
                    "user_uuid": user_id,
                    "doc_type": "summary"
                }
            ))
            
            # Traiter chaque transaction de l'utilisateur
            for _, row in user_df.iterrows():
                date = row['date']
                description = row['description_clean']
                amount = row['net_amount']
                category = row['category_name']
                parent_category = row['parent_name']
                
                # Créer le contenu de la transaction
                content = f"Transaction du {date}: {description}, Montant: {amount} €, "
                content += f"Catégorie: {category}, Catégorie parente: {parent_category}"
                
                # Ajouter le document avec métadonnées
                documents.append(Document(
                    page_content=content,
                    metadata={
                        "user_uuid": user_id,
                        "date": date,
                        "amount": amount,
                        "category": category,
                        "parent_category": parent_category,
                        "doc_type": "transaction"
                    }
                ))
                
            # Ajouter un document par catégorie pour cet utilisateur
            for category in user_df['category_name'].unique():
                category_df = user_df[user_df['category_name'] == category]
                total_amount = category_df['net_amount'].sum()
                avg_amount = category_df['net_amount'].mean()
                count = len(category_df)
                
                content = f"Résumé de la catégorie {category} pour l'utilisateur {user_id}:\n"
                content += f"- Nombre de transactions: {count}\n"
                content += f"- Montant total: {total_amount:.2f} €\n"
                content += f"- Montant moyen: {avg_amount:.2f} €\n"
                
                documents.append(Document(
                    page_content=content,
                    metadata={
                        "user_uuid": user_id,
                        "category": category,
                        "doc_type": "category_summary"
                    }
                ))
        
        # Diviser les documents si nécessaire (pour de grandes transactions)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        texts = text_splitter.split_documents(documents)
        
        print(f"Vectorisation de {len(texts)} documents...")
        
        # Créer la base vectorielle
        self.vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        # Persister la base vectorielle
        self.vectorstore.persist()
        print("Base vectorielle créée et persistée.")
    
    def _generate_user_summary(self, user_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Génère des statistiques résumées pour un utilisateur
        
        Args:
            user_df: DataFrame contenant les transactions d'un utilisateur
            
        Returns:
            Dict contenant les statistiques résumées
        """
        start_date = user_df['date'].min()
        end_date = user_df['date'].max()
        total_transactions = len(user_df)
        total_spent = user_df[user_df['net_amount'] < 0]['net_amount'].sum() * -1
        total_received = user_df[user_df['net_amount'] > 0]['net_amount'].sum()
        top_categories = user_df['category_name'].value_counts().head(5).index.tolist()
        
        return {
            'start_date': start_date,
            'end_date': end_date,
            'total_transactions': total_transactions,
            'total_spent': abs(total_spent),
            'total_received': total_received,
            'top_categories': top_categories
        }
    
    def initialize_retriever_for_user(self, user_uuid: str) -> bool:
        """
        Initialise le retriever pour un utilisateur spécifique
        
        Args:
            user_uuid: UUID de l'utilisateur
            
        Returns:
            bool: True si l'utilisateur existe, False sinon
        """
        # Vérifier si l'utilisateur existe
        if user_uuid not in self.user_ids:
            print(f"Utilisateur {user_uuid} non trouvé dans la base de données.")
            print(f"UUIDs disponibles: {list(self.user_ids)[:5]} (et {len(self.user_ids) - 5} autres)")
            return False
        
        # Créer un retriever spécifique à l'utilisateur
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "filter": {"user_uuid": user_uuid},
                "fetch_k": 20
            }
        )
        
        # Template de prompt pour le système RAG
        prompt_template = """
        Tu es un assistant financier spécialisé dans l'analyse des transactions bancaires.
        Tu as accès aux transactions de l'utilisateur et tu dois l'aider à mieux comprendre ses dépenses et à optimiser son budget.
        
        Réponds uniquement aux questions concernant ses transactions ou liées à la gestion de budget.
        Si la question sort de ce cadre, réponds poliment que tu n'es pas programmé pour répondre à ce genre de question.
        
        Bases tes réponses uniquement sur les données de transaction fournies.
        Si une information n'est pas disponible dans les données, indique-le clairement.
        
        Contexte des transactions:
        {context}
        
        Question: {input}
        
        Réponse:
        """
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "input"]
        )
        
        # Créer le LLM
        llm = ChatOpenAI(temperature=0, model="gpt-4")
        
        # Créer la chaîne de documents
        combine_docs_chain = create_stuff_documents_chain(llm, PROMPT)
        
        # Créer la chaîne de récupération
        self.qa_chain = create_retrieval_chain(self.retriever, combine_docs_chain)
        
        print(f"Retriever initialisé pour l'utilisateur {user_uuid}")
        return True
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Pose une question au système RAG
        
        Args:
            question: Question de l'utilisateur
            
        Returns:
            Dict contenant la réponse et les documents sources
        """
        if not self.retriever or not self.qa_chain:
            return {
                "answer": "Erreur: Aucun utilisateur n'a été sélectionné. Veuillez d'abord initialiser un utilisateur.",
                "sources": []
            }
        
        try:
            # Vérifier si la question est liée au budget ou aux transactions
            if not self._is_budget_related(question):
                return {
                    "answer": "Je suis désolé, je ne suis programmé que pour répondre aux questions concernant vos transactions bancaires et votre gestion de budget. Pourriez-vous reformuler votre question dans ce contexte?",
                    "sources": []
                }
            
            # Poser la question au système RAG
            print(f"Envoi de la question: '{question}' au système RAG")
            result = self.qa_chain.invoke({"input": question})
            print(f"Structure de réponse reçue: {list(result.keys())}")
            
            # Extraire la réponse
            answer = ""
            if "answer" in result:
                answer = result["answer"]
            else:
                # Chercher dans d'autres clés possibles
                print(f"Clés disponibles: {list(result.keys())}")
                if hasattr(result, "get"):
                    for key in result.keys():
                        if key != "context" and isinstance(result[key], str):
                            answer = result[key]
                            print(f"Réponse trouvée dans la clé: {key}")
                            break
            
            # Extraire les sources - nouvelle structure
            sources = []
            
            # Vérifier la structure pour localiser les documents
            if "context" in result:
                context_docs = result["context"]
                print(f"Nombre de documents de contexte: {len(context_docs)}")
                for doc in context_docs:
                    if hasattr(doc, "page_content") and hasattr(doc, "metadata"):
                        sources.append({
                            "content": doc.page_content,
                            "metadata": doc.metadata
                        })
            # Nouvelle alternative pour certaines versions de LangChain
            elif "retrieved_documents" in result:
                docs = result["retrieved_documents"]
                print(f"Nombre de documents récupérés: {len(docs)}")
                for doc in docs:
                    if hasattr(doc, "page_content") and hasattr(doc, "metadata"):
                        sources.append({
                            "content": doc.page_content,
                            "metadata": doc.metadata
                        })
            
            print(f"Réponse trouvée: {answer[:50]}... (tronquée)")
            print(f"Nombre de sources: {len(sources)}")
            
            return {
                "answer": answer,
                "sources": sources
            }
            
        except Exception as e:
            print(f"Erreur dans ask_question: {str(e)}")
            traceback.print_exc()
            return {
                "answer": f"Une erreur s'est produite: {str(e)}",
                "sources": []
            }
    
    def _is_budget_related(self, question: str) -> bool:
        """
        Vérifie si une question est liée au budget ou aux transactions
        
        Args:
            question: Question de l'utilisateur
            
        Returns:
            bool: True si la question est liée au budget, False sinon
        """
        # Liste de mots-clés liés au budget et aux transactions
        budget_keywords = [
            "dépense", "dépenses", "budget", "transaction", "transactions", "argent",
            "économie", "économies", "épargne", "compte", "comptes", "banque",
            "bancaire", "finance", "financier", "financière", "coût", "coûts",
            "prix", "montant", "montants", "euros", "€", "euro", "paiement",
            "paiements", "revenu", "revenus", "salaire", "salaires", "catégorie",
            "catégories", "achat", "achats", "facture", "factures", "abonnement",
            "abonnements", "mensuel", "mensuelle", "annuel", "annuelle", "quotidien",
            "quotidienne", "hebdomadaire", "loyer", "nourriture", "alimentation",
            "transport", "loisir", "loisirs", "économiser", "dépenser", "investir",
            "investissement", "investissements", "crédit", "débit", "virement"
        ]
        
        # Convertir la question en minuscules
        question_lower = question.lower()
        
        # Vérifier si un des mots-clés est présent dans la question
        for keyword in budget_keywords:
            if keyword in question_lower:
                return True
        
        return False

# Point d'entrée pour l'API
def create_rag_system(csv_path: str, persist_directory: str = "db"):
    """
    Crée et initialise le système RAG
    
    Args:
        csv_path: Chemin vers le fichier CSV des transactions
        persist_directory: Répertoire pour persister la base vectorielle
        
    Returns:
        BudgetRAG: Instance du système RAG
    """
    print("\n==== CRÉATION DU SYSTÈME RAG ====")
    print(f"CSV path: {csv_path} (absolu: {os.path.abspath(csv_path)})")
    print(f"Persist dir: {persist_directory} (absolu: {os.path.abspath(persist_directory)})")
    
    rag = BudgetRAG(csv_path, persist_directory)
    rag.load_and_process_data()
    
    print(f"Système RAG créé avec {len(rag.user_ids)} utilisateurs")
    return rag

if __name__ == "__main__":
    # Test du système RAG
    rag = create_rag_system(csv_path, persist_directory)
    
    # Afficher les UUIDs disponibles
    print("\nUtilisateurs disponibles:")
    for i, user_id in enumerate(list(rag.user_ids)[:5]):
        print(f"{i+1}. {user_id}")
    
    # Exemple d'utilisation (si des utilisateurs existent)
    if rag.user_ids:
        user_id = next(iter(rag.user_ids))
        print(f"\nTest avec l'utilisateur: {user_id}")
        if rag.initialize_retriever_for_user(user_id):
            result = rag.ask_question("Quelles sont mes dépenses les plus importantes ce mois-ci?")
            print("\nRéponse:")
            print(result["answer"])
    else:
        print("\nAucun utilisateur trouvé pour le test!")