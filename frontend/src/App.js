// App.js
import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import axios from 'axios';

// URL du backend
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:3000';

function App() {
  const [userUuid, setUserUuid] = useState('');
  const [question, setQuestion] = useState('');
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [error, setError] = useState('');
  const [availableUsers, setAvailableUsers] = useState([]);
  const [isLoadingUsers, setIsLoadingUsers] = useState(false);

  const messagesEndRef = useRef(null);

  // Effet pour charger la liste des utilisateurs disponibles (pour le développement)
  useEffect(() => {
    const fetchUsers = async () => {
      try {
        setIsLoadingUsers(true);
        const response = await axios.get(`${API_URL}/api/users`);
        setAvailableUsers(response.data);
      } catch (error) {
        console.error("Erreur lors du chargement des utilisateurs:", error);
      } finally {
        setIsLoadingUsers(false);
      }
    };

    fetchUsers();
  }, []);

  // Effet pour faire défiler vers le bas lors de nouveaux messages
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleUserLogin = async (e) => {
    e.preventDefault();
    
    if (!userUuid.trim()) {
      setError("Veuillez saisir un UUID utilisateur");
      return;
    }

    try {
      setIsLoading(true);
      setError('');
      
      // Vérifier si l'utilisateur existe
      const response = await axios.post(`${API_URL}/api/check-user`, { user_uuid: userUuid });
      
      if (response.data.exists) {
        setIsAuthenticated(true);
        // Ajouter un message de bienvenue
        setMessages([{
          type: 'system',
          content: `Bienvenue! Je suis votre assistant financier personnel. Comment puis-je vous aider avec vos finances aujourd'hui?`
        }]);
      } else {
        setError("Utilisateur non trouvé dans la base de données");
      }
    } catch (error) {
      console.error("Erreur d'authentification:", error);
      setError(error.response?.data?.error || "Erreur lors de la vérification de l'utilisateur");
    } finally {
      setIsLoading(false);
    }
  };

  const handleSendQuestion = async (e) => {
    e.preventDefault();
    
    if (!question.trim()) return;
    
    // Ajouter la question de l'utilisateur aux messages
    const newQuestion = { type: 'user', content: question };
    setMessages(prevMessages => [...prevMessages, newQuestion]);
    
    // Réinitialiser le champ de question
    setQuestion('');
    
    try {
      setIsLoading(true);
      
      // Envoyer la question au backend
      const response = await axios.post(`${API_URL}/api/ask`, {
        user_uuid: userUuid,
        question: newQuestion.content
      });
      
      // Ajouter la réponse aux messages
      setMessages(prevMessages => [
        ...prevMessages, 
        { 
          type: 'assistant', 
          content: response.data.answer,
          sources: response.data.sources 
        }
      ]);
    } catch (error) {
      console.error("Erreur lors de l'envoi de la question:", error);
      setMessages(prevMessages => [
        ...prevMessages, 
        { 
          type: 'error', 
          content: error.response?.data?.error || "Erreur lors de la réception de la réponse" 
        }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleLogout = () => {
    setIsAuthenticated(false);
    setUserUuid('');
    setMessages([]);
  };

  const selectUser = (uuid) => {
    setUserUuid(uuid);
  };

  // Rendu du composant de login
  const renderLogin = () => (
    <div className="login-container">
      <h2>Budget Assistant - Login</h2>
      <form onSubmit={handleUserLogin}>
        <div className="form-group">
          <label htmlFor="userUuid">Identifiant Utilisateur (UUID):</label>
          <input
            type="text"
            id="userUuid"
            value={userUuid}
            onChange={(e) => setUserUuid(e.target.value)}
            placeholder="Entrez votre UUID utilisateur"
            disabled={isLoading}
          />
        </div>
        {error && <div className="error-message">{error}</div>}
        <button type="submit" disabled={isLoading}>
          {isLoading ? 'Vérification...' : 'Se connecter'}
        </button>
      </form>

      {availableUsers.length > 0 && (
        <div className="available-users">
          <h3>Utilisateurs disponibles (pour le développement)</h3>
          {isLoadingUsers ? (
            <p>Chargement des utilisateurs...</p>
          ) : (
            <ul>
              {availableUsers.slice(0, 5).map((uuid, index) => (
                <li key={index}>
                  <button onClick={() => selectUser(uuid)}>
                    {uuid.substring(0, 8)}...
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );

  // Rendu du composant de chat
  const renderChat = () => (
    <div className="chat-container">
      <div className="chat-header">
        <h2>Assistant Budget Personnel</h2>
        <span>Utilisateur: {userUuid.substring(0, 8)}...</span>
        <button onClick={handleLogout} className="logout-btn">
          Déconnexion
        </button>
      </div>
      
      <div className="messages-container">
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.type}`}>
            <div className="message-content">{msg.content}</div>
            {msg.sources && msg.sources.length > 0 && (
              <div className="message-sources">
                <details>
                  <summary>Sources ({msg.sources.length})</summary>
                  <ul>
                    {msg.sources.map((source, idx) => (
                      <li key={idx}>
                        {source.content}
                        {source.metadata && (
                          <span className="metadata">
                            {source.metadata.date && `Date: ${source.metadata.date}`}
                            {source.metadata.category && `, Catégorie: ${source.metadata.category}`}
                          </span>
                        )}
                      </li>
                    ))}
                  </ul>
                </details>
              </div>
            )}
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      
      <form onSubmit={handleSendQuestion} className="question-form">
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Posez une question sur vos finances..."
          disabled={isLoading}
        />
        <button type="submit" disabled={isLoading}>
          {isLoading ? 'Envoi...' : 'Envoyer'}
        </button>
      </form>
    </div>
  );

  return (
    <div className="app">
      {isAuthenticated ? renderChat() : renderLogin()}
    </div>
  );
}

export default App;