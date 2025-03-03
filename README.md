# Guide de déploiement - BudgetRAG

Ce guide détaille les étapes pour déployer l'application BudgetRAG en local. L'application est composée de trois parties principales :
1. Un système RAG Python avec API FastAPI
2. Un backend Node.js
3. Un frontend React

## Prérequis

- **Python 3.9+**
- **Node.js 16+**
- **npm 8+**
- **Clé API OpenAI**
- **Clé API LangSmith** (optionnel, pour le debugging)

## Structure du projet

```
budget-rag/
├── data/
│   └── transactions.csv
├── python-api/
│   ├── rag_system.py
│   ├── api.py
│   └── requirements.txt
├── backend/
│   ├── server.js
│   └── package.json
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── App.js
│   │   ├── App.css
│   │   └── index.js
│   └── package.json
└── .env
```

## Étape 1 : Préparation de l'environnement

1. Clonez le dépôt ou créez la structure de dossiers ci-dessus

2. Créez un fichier `.env` à la racine du projet avec les variables suivantes :

```
# OpenAI API
OPENAI_API_KEY=votre_clé_api_openai

# LangSmith (optionnel)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=votre_clé_api_langsmith
LANGCHAIN_PROJECT=budget-rag-assistant

# Chemins des fichiers
CSV_PATH=./data/transactions.csv
PERSIST_DIRECTORY=./db

# Ports
PYTHON_API_PORT=8000
NODE_BACKEND_PORT=5000
REACT_FRONTEND_PORT=3000
```

3. Placez votre fichier de transactions dans `data/transactions.csv`

## Étape 2 : Installation et démarrage du système RAG Python

1. Créez un environnement virtuel Python

```bash
cd python-api
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
```

2. Installez les dépendances

```bash
pip install -r requirements.txt
```

Contenu du fichier `requirements.txt` :
```
fastapi
uvicorn
langchain
langchain-openai
langsmith
chromadb
pandas
pydantic
python-dotenv
```

3. Démarrez l'API Python

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Le premier démarrage peut prendre du temps car le système va charger et vectoriser toutes les transactions.

## Étape 3 : Installation et démarrage du backend Node.js

1. Installez les dépendances

```bash
cd backend
npm install
```

Contenu du fichier `package.json` :
```json
{
  "name": "budget-rag-backend",
  "version": "1.0.0",
  "description": "Backend pour l'application BudgetRAG",
  "main": "server.js",
  "scripts": {
    "start": "node server.js",
    "dev": "nodemon server.js"
  },
  "dependencies": {
    "axios": "^1.6.2",
    "cors": "^2.8.5",
    "dotenv": "^16.3.1",
    "express": "^4.18.2"
  },
  "devDependencies": {
    "nodemon": "^3.0.1"
  }
}
```

2. Démarrez le serveur Node.js

```bash
npm run dev
```

## Étape 4 : Installation et démarrage du frontend React

1. Créez l'application React si ce n'est pas déjà fait

```bash
npx create-react-app frontend
cd frontend
```

2. Installez les dépendances supplémentaires

```bash
npm install axios
```

3. Copiez les fichiers `App.js` et `App.css` dans le dossier `src`

4. Créez un fichier `.env.local` dans le dossier `frontend` avec la variable suivante :

```
REACT_APP_API_URL=http://localhost:5000
```

5. Démarrez l'application React

```bash
npm start
```

## Étape 5 : Utilisation de l'application

1. Accédez à l'application dans votre navigateur : http://localhost:3000

2. Sur la page de connexion, entrez un UUID d'utilisateur existant dans votre fichier de transactions. Pour le développement, l'application affiche automatiquement quelques UUIDs disponibles que vous pouvez utiliser.

3. Une fois connecté, vous pouvez poser des questions sur vos transactions et votre budget.

## Sécurité des données

Cette application a été conçue avec un focus sur la sécurité des données utilisateurs :

1. **Isolation des données par utilisateur** : Les requêtes sont filtrées par UUID utilisateur pour garantir que chaque utilisateur n'a accès qu'à ses propres données.

2. **Filtrage des questions hors sujet** : Le système RAG est configuré pour rejeter les questions qui ne sont pas liées au budget ou aux transactions.

3. **Vectorisation sécurisée** : L'index vectoriel est partitionné par utilisateur pour garantir l'isolation des données.

4. **Déploiement local** : L'application est déployée en local, ce qui réduit les risques de fuite de données.

5. **Absence d'authentification complexe** : Dans cette version de développement, l'authentification se fait simplement par UUID. Pour un déploiement en production, il faudrait mettre en place un système d'authentification plus robuste (JWT, OAuth, etc.).

## Optimisations possibles

Pour un déploiement à grande échelle, envisagez les optimisations suivantes :

1. **Base de données** : Remplacer le stockage CSV par une base de données (PostgreSQL, MongoDB).

2. **Cache** : Mettre en place un système de cache pour les requêtes fréquentes.

3. **Vectorisation par lots** : Vectoriser les transactions par lots pour gérer de grands volumes de données.

4. **Déploiement conteneurisé** : Utiliser Docker et Kubernetes pour faciliter le déploiement et la scalabilité.

5. **Authentification robuste** : Implémenter un système d'authentification complet avec JWT et OAuth.

6. **HTTPS** : Mettre en place HTTPS pour sécuriser les communications.

7. **Tests** : Ajouter des tests unitaires et d'intégration pour garantir la stabilité du système.

