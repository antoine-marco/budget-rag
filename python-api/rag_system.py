# rag_system.py
import os
import pandas as pd
from typing import List, Dict, Any
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
from langchain.callbacks.tracers import LangChainTracer
from langchain.prompts import PromptTemplate
from langsmith import Client
import numpy as np
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Configuration LangSmith pour le debug
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "budget-rag-assistant"

client = Client()
tracer = LangChainTracer(project_name="budget-rag-assistant")

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
        
    def load_and_process_data(self, force_reload: bool = False) -> None:
        """
        Charge et prétraite les données de transactions
        
        Args:
            force_reload: Si True, force le rechargement des données même si la base vectorielle existe
        """
        # Vérifier si la base vectorielle existe déjà
        if os.path.exists(self.persist_directory) and not force_reload:
            print("Chargement de la base vectorielle existante...")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            
            # Récupérer la liste des user_ids
            all_docs = self.vectorstore.get()
            metadata_list = all_docs.get('metadatas', [])
            self.user_ids = {m.get('user_uuid') for m in metadata_list if m.get('user_uuid')}
            print(f"Base chargée avec {len(self.user_ids)} utilisateurs uniques.")
            return
        
        print("Chargement et traitement des données de transactions...")
        # Charger les données CSV
        df = pd.read_csv(self.csv_path)
        
        # Convertir les montants en float (si nécessaire)
        if df['Net Amount'].dtype == 'object':
            df['Net Amount'] = df['Net Amount'].str.replace(',', '.').astype(float)
        
        # Collecter tous les user_ids
        self.user_ids = set(df['User UUID'].unique())
        
        # Créer des documents pour chaque transaction avec métadonnées
        documents = []
        
        for user_id in self.user_ids:
            user_df = df[df['User UUID'] == user_id]
            
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
                date = row['Date']
                description = row['Description Clean']
                amount = row['Net Amount']
                category = row['Category Name']
                parent_category = row['Parent Name']
                
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
            for category in user_df['Category Name'].unique():
                category_df = user_df[user_df['Category Name'] == category]
                total_amount = category_df['Net Amount'].sum()
                avg_amount = category_df['Net Amount'].mean()
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
        start_date = user_df['Date'].min()
        end_date = user_df['Date'].max()
        total_transactions = len(user_df)
        total_spent = user_df[user_df['Net Amount'] < 0]['Net Amount'].sum() * -1
        total_received = user_df[user_df['Net Amount'] > 0]['Net Amount'].sum()
        top_categories = user_df['Category Name'].value_counts().head(5).index.tolist()
        
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
        
        Historique de la conversation:
        {chat_history}
        
        Question: {question}
        
        Réponse:
        """
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "chat_history", "question"]
        )
        
        # Initialiser la mémoire de conversation
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Créer la chaîne de question-réponse
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(temperature=0, model="gpt-4"),
            retriever=self.retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": PROMPT},
            return_source_documents=True,
            callbacks=[tracer]
        )
        
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
            result = self.qa_chain({"question": question})
            
            # Extraire la réponse et les sources
            answer = result.get("answer", "")
            source_docs = result.get("source_documents", [])
            
            # Formater les sources pour le frontend
            sources = []
            for doc in source_docs:
                sources.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
            
            return {
                "answer": answer,
                "sources": sources
            }
            
        except Exception as e:
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
    rag = BudgetRAG(csv_path, persist_directory)
    rag.load_and_process_data()
    return rag

if __name__ == "__main__":
    # Test du système RAG
    rag = create_rag_system("./data/transactions.csv")
    
    # Exemple d'utilisation
    user_id = "some-user-uuid"
    if rag.initialize_retriever_for_user(user_id):
        result = rag.ask_question("Quelles sont mes dépenses les plus importantes ce mois-ci?")
        print(result["answer"])