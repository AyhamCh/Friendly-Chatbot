# FriendlyChatbot

Un chatbot conversationnel intelligent et poli, développé avec PyTorch et Transformers. Ce projet offre trois modèles différents pour générer des réponses naturelles et courtoisies.

## Caractéristiques

- Trois modèles au choix : BlenderBot, DialoGPT Amélioré, et GPT-2 Medium
- Support GPU avec CUDA pour accélérer l'inférence (sinon CPU)
- Filtrage de politesse qui bloque les réponses impoles et inappropriées
- Mode conversation interactive pour des échanges continus
- Génération intelligente avec beam search, temperature control, et autres techniques de décoding

## Installation

### Prérequis
- Python 3.7+
- CUDA 11+ (optionnel, pour GPU)

### Installation des dépendances

```bash
pip install transformers torch sentencepiece accelerate
```

## Utilisation

### 1. Charger le chatbot

```python
from FriendlyChatbot import BlenderBotPoli

# Créer une instance du chatbot
bot = BlenderBotPoli()
```

### 2. Générer une réponse

```python
reponse = bot.generer_reponse("Bonjour, comment allez-vous?")
print(reponse)
```

### 3. Mode conversation interactive

```python
conversation_interactive()
```

Puis tapez vos messages :
- `quit` ou `exit` pour quitter
- `reset` pour réinitialiser la conversation

## Modèles disponibles

### BlenderBot (Recommandé)
- Modèle : `facebook/blenderbot-400M-distill`
- Avantages : Très poli, empathique, engageant
- Temps de réponse : 0.4-2.7 secondes
- Meilleur pour : Conversations naturelles et chaleureuses

### DialoGPT Amélioré
- Modèle : `microsoft/DialoGPT-medium`
- Avantages : Filtre les réponses impoles, maintient l'historique
- Temps de réponse : Variable selon les tentatives
- Meilleur pour : Contrôle strict de la politesse

### GPT-2 Medium
- Modèle : `gpt2-medium`
- Avantages : Intelligent, rapide
- Temps de réponse : 0.4 secondes
- Meilleur pour : Réponses précises avec prompts

## Configuration

### Paramètres de génération

Tous les modèles utilisent des paramètres optimisés :

```python
max_length=150          # Longueur maximale de la réponse
temperature=0.7        # Contrôle de la créativité (0-1)
top_k=50              # Top-k sampling
top_p=0.9             # Nucleus sampling
num_beams=8           # Beam search (pour BlenderBot)
repetition_penalty=1.2 # Éviter les répétitions
no_repeat_ngram_size=3 # Pas de répétition de 3-grammes
```

## Structure du code

### Classe BlenderBotPoli

```python
bot = BlenderBotPoli(model_name="facebook/blenderbot-400M-distill")
reponse = bot.generer_reponse("Votre message")
```

### Classe DialoGPTAmeliore

```python
bot = DialoGPTAmeliore(model_name="microsoft/DialoGPT-medium")
reponse = bot.generer_reponse("Votre message")
bot.reinitialiser()  # Réinitialiser l'historique
```

### Classe ChatbotAvance

```python
bot = ChatbotAvance(model_name="gpt2-medium")
reponse = bot.generer_reponse("Votre message")
```

## Exemples de résultats

```
Utilisateur: Hello! How are you today?
Bot: I'm doing well, thank you. How about yourself? Do you have any plans for the weekend?

Utilisateur: What's your favorite color?
Bot: My favorite color is blue. What is yours? Do you like the color blue?

Utilisateur: Can you help me with something?
Bot: Sure, what do you need help with? I'm always willing to lend a hand.
```

## Architecture

- **Tokenizer** : Convertit le texte en tokens pour le modèle
- **Modèle** : Génère les tokens de réponse basés sur l'entrée
- **Décodeur** : Reconvertit les tokens en texte naturel
- **Filtre** : Bloque les réponses inappropriées (pour DialoGPT)

## Performance

Avec une Tesla T4 GPU :
- Mémoire disponible : 15.83 GB
- Temps de chargement du modèle : 13-20 secondes
- Temps de génération par réponse : 0.37-2.71 secondes

## Limitations et améliorations futures

- Le chatbot peut générer des réponses génériques
- Pas de mémoire persistante entre les sessions (sauf DialoGPT)
- Les réponses peuvent parfois être hors de propos
- Amélioration possible avec fine-tuning sur des données spécifiques

## Dépendances principales

- transformers >= 4.0
- torch >= 1.9
- sentencepiece
- accelerate


## Auteur

Créé avec PyTorch et la bibliothèque Transformers de Hugging Face.
