// server.js
const express = require('express');
const cors = require('cors');
const axios = require('axios');
const dotenv = require('dotenv');
const path = require('path');

// Charger les variables d'environnement
dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;

// URL de l'API Python RAG
const RAG_API_URL = process.env.RAG_API_URL || 'http://localhost:8000';

// Middleware
app.use(express.json());
app.use(cors({
  origin: 'http://localhost:3000',
  credentials: true
}));

// Servir les fichiers statiques de React en production
if (process.env.NODE_ENV === 'production') {
  app.use(express.static(path.join(__dirname, '../frontend/build')));
}

// Routes API
app.post('/api/check-user', async (req, res) => {
  try {
    const { user_uuid } = req.body;
    
    if (!user_uuid) {
      return res.status(400).json({ error: 'UUID utilisateur requis' });
    }
    
    // Vérifier l'existence de l'utilisateur via l'API Python
    const response = await axios.post(`${RAG_API_URL}/api/check-user`, { user_uuid });
    
    return res.json(response.data);
  } catch (error) {
    console.error('Erreur lors de la vérification de l\'utilisateur:', error.message);
    
    // Gérer spécifiquement les erreurs de l'API RAG
    if (error.response) {
      return res.status(error.response.status).json({ 
        error: error.response.data.detail || 'Erreur du service RAG' 
      });
    }
    
    return res.status(500).json({ error: 'Erreur serveur' });
  }
});

app.post('/api/ask', async (req, res) => {
  try {
    const { user_uuid, question } = req.body;
    
    if (!user_uuid || !question) {
      return res.status(400).json({ error: 'UUID utilisateur et question requis' });
    }
    
    // Poser la question via l'API Python
    const response = await axios.post(`${RAG_API_URL}/api/ask`, { 
      user_uuid, 
      question 
    });
    
    return res.json(response.data);
  } catch (error) {
    console.error('Erreur lors de la requête au système RAG:', error.message);
    
    // Gérer spécifiquement les erreurs de l'API RAG
    if (error.response) {
      return res.status(error.response.status).json({ 
        error: error.response.data.detail || 'Erreur du service RAG' 
      });
    }
    
    return res.status(500).json({ error: 'Erreur serveur' });
  }
});

// Route pour obtenir la liste des utilisateurs disponibles (pour le développement)
app.get('/api/users', async (req, res) => {
  try {
    // Récupérer la liste des utilisateurs via l'API Python
    const response = await axios.get(`${RAG_API_URL}/api/users`);
    
    return res.json(response.data);
  } catch (error) {
    console.error('Erreur lors de la récupération des utilisateurs:', error.message);
    return res.status(500).json({ error: 'Erreur serveur' });
  }
});

// Pour toutes les autres routes en production, servir l'application React
if (process.env.NODE_ENV === 'production') {
  app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, '../frontend/build/index.html'));
  });
}

// Démarrer le serveur
app.listen(PORT, () => {
  console.log(`Serveur démarré sur le port ${PORT}`);
  console.log(`API RAG connectée à ${RAG_API_URL}`);
});