Projet en cours de développement

# RAG PDF + Figures (FR)

Ce projet met en place un petit pipeline RAG (Retrieval-Augmented Generation) sur des PDF :

- extraction du texte d'un PDF (PyMuPDF) et découpage en chunks (LangChain),
- extraction et analyse des figures/pages images via Gemini Vision,
- conversion des analyses de figures en `Document` LangChain et indexation dans FAISS,
- interrogation via un LLM (Gemini).

## Fichiers importants
- `RAG/main.py` — orchestrateur (indexation ou interrogation).
- `RAG/utils.py` — fonctions utilitaires : chargement PDF, découpage, embeddings, création/chargement FAISS, pipelines.
- `RAG/figures.py` — extraction des pages-figures et appel à Gemini Vision.
- `requirements.txt` — dépendances Python.

## Sécurité du cache FAISS
FAISS sérialise des données via pickle. Charger un index local nécessite `allow_dangerous_deserialization=True` (autorisé dans le code) — ne le faites que si vous faites confiance au fichier (index créé localement par vous). Sinon, supprimez/regenérez l'index en réindexant.

## Modèle Gemini
Par défaut les scripts utilisent `gemini-2.5-flash-lite`. Si ton SDK ne supporte pas ce modèle, mets à jour `google-generativeai` ou modifie le nom du modèle dans `RAG/utils.py` et `RAG/figures.py`.


