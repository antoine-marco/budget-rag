# verify_rag.py
import requests
import json

def test_rag_api():
    """Vérifie que l'API RAG fonctionne et reconnaît l'UUID"""
    
    print("Test de l'API RAG...")
    uuid = "feae4141-715e-42e6-8955-32522e10d326"  # UUID confirmé comme existant
    
    # Test de l'endpoint check-user
    try:
        print(f"\n1. Vérification de l'utilisateur avec UUID: {uuid}")
        response = requests.post(
            "http://localhost:8000/api/check-user", 
            json={"user_uuid": uuid},
            headers={"Content-Type": "application/json"}
        )
        
        print(f"   Statut: {response.status_code}")
        print(f"   Réponse: {response.text}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get("exists"):
                print("   ✅ L'API confirme que l'utilisateur existe!")
            else:
                print("   ❌ L'API indique que l'utilisateur n'existe PAS!")
                print("   Le système RAG ne charge probablement pas correctement le CSV.")
        else:
            print("   ❌ Erreur lors de l'appel à l'API!")
            
    except Exception as e:
        print(f"   ❌ Erreur de connexion: {str(e)}")
        print("   Assurez-vous que l'API Python est en cours d'exécution sur http://localhost:8000")
        return False
    
    # Test de l'endpoint Node.js
    try:
        print("\n2. Vérification du backend Node.js")
        response = requests.post(
            "http://localhost:5001/api/check-user", 
            json={"user_uuid": uuid},
            headers={"Content-Type": "application/json"}
        )
        
        print(f"   Statut: {response.status_code}")
        print(f"   Réponse: {response.text}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get("exists"):
                print("   ✅ Le backend Node.js confirme que l'utilisateur existe!")
            else:
                print("   ❌ Le backend Node.js indique que l'utilisateur n'existe PAS!")
        else:
            print("   ❌ Erreur lors de l'appel au backend Node.js!")
            
    except Exception as e:
        print(f"   ❌ Erreur de connexion au backend Node.js: {str(e)}")
        print("   Assurez-vous que le serveur Node.js est en cours d'exécution sur http://localhost:5001")
        return False
    
    return True

if __name__ == "__main__":
    test_rag_api()