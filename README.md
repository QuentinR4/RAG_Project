# RAG PDF + Figures + Abréviations (FR)

Ce projet implémente un pipeline RAG (Retrieval-Augmented Generation) pour interroger des rapports PDF en français en combinant texte, figures et abréviations:

- Extraction du texte (PyMuPDF) et découpage en chunks (LangChain).
- Détection des pages contenant des figures puis analyse automatique des figures avec Gemini Vision (retour JSON structuré, rate limit ≈ 15 req/min), conversion en `Document` et indexation.
- Enrichissement des chunks avec les définitions d’abréviations détectées dans le PDF (ex: « gaz à effet de serre (GES) »). Les abréviations sont extraites puis leurs définitions sont injectées dans le texte pour améliorer la compréhension et la recherche.
- Indexation de l’ensemble (texte + figures + abréviations enrichies) dans FAISS puis interrogation via un LLM (Gemini) avec contexte récupéré.
- CLI simple dans `RAG/main.py` pour indexer un ou plusieurs PDF, poser une ou plusieurs questions, ou faire les deux.
- Tests Pytest avec un CSV d’évaluation et génération d’un fichier de résultats horodaté à chaque exécution.

## Fichiers importants
- `RAG/main.py` — orchestrateur (indexation ou interrogation).
- `RAG/utils.py` — fonctions utilitaires : chargement PDF, découpage, embeddings, création/chargement FAISS, pipelines.
- `RAG/figures.py` — extraction des pages-figures et appel à Gemini Vision.
- `RAG/abbreviation.py` - extraction des acronymes, création d'un dictionnaire avec leur signification, pour l'ajouter dans les chunks.
- `requirements.txt` — dépendances Python.

## Sécurité du cache FAISS
FAISS sérialise des données via pickle. Charger un index local nécessite `allow_dangerous_deserialization=True` (autorisé dans le code) — ne le faites que si vous faites confiance au fichier (index créé localement par vous). Sinon, supprimez/regenérez l'index en réindexant.

## Modèle Gemini
Par défaut les scripts utilisent `gemini-2.5-flash-lite`. Si ton SDK ne supporte pas ce modèle, mets à jour `google-generativeai` ou modifie le nom du modèle dans `RAG/utils.py` et `RAG/figures.py`.

## Données générées et cache
- L’index FAISS est sauvegardé dans `RAG/cache/faiss_index`.
- Les images des pages de figures et le résumé JSON sont produits dans `RAG/Dataset/rag_figures/` (ignoré par Git, non versionné).

## Utilisation (CLI)

Le script `RAG/main.py` propose une interface en ligne de commande pour indexer des PDF et poser des questions.

Pré-requis:
- Clé API Gemini dans la variable d'environnement `GEMINI_API_KEY` pour la partie questions.
- Avoir installé les dépendances de `requirements.txt`.

Exemples (PowerShell, depuis la racine du projet):

1) Indexer un PDF:
```powershell
python RAG\main.py --doc ".\RAG\Dataset\HCC_RA_2025-18.07_web.pdf"
```

2) Indexer plusieurs PDF:
```powershell
python RAG\main.py -d ".\RAG\Dataset\HCC_RA_2025-18.07_web.pdf" -d "C:\\mon\\autre.pdf"
```

3) Poser une question (après indexation):
```powershell
$env:GEMINI_API_KEY = "<ta_clé>"
python RAG\main.py --question "De combien sont les émissions de GES du Royaume-Uni en 2024 ?"
```

4) Poser plusieurs questions et ajuster k (nombre de documents récupérés):
```powershell
python RAG\main.py -q "Question 1 ?" -q "Question 2 ?" -k 20
```

5) Indexer puis poser une question dans le même appel:
```powershell
python RAG\main.py -d ".\RAG\Dataset\HCC_RA_2025-18.07_web.pdf" -q "Explique la figure clé sur la page 101"
```

6) Forcer la recréation complète de l'index FAISS (écrase l'existant):
```powershell
python RAG\main.py -d ".\RAG\Dataset\HCC_RA_2025-18.07_web.pdf" --force-reindex
```
## Tests (pytest)

Des tests basiques existent dans `RAG/Test/test_rag_pipeline.py`.

- Pré-requis: activer l'environnement virtuel et installer les deps
	- PowerShell (depuis la racine du repo):
		```powershell
		. .venv\Scripts\Activate.ps1
		pip install -r requirements.txt
		```

- Lancer tous les tests:
	```powershell
	python -m pytest -q
	```

- Lancer uniquement le test RAG:
	```powershell
	python -m pytest RAG\Test\test_rag_pipeline.py -q
	```

- Voir les sorties (prints) et détails:
	```powershell
	python -m pytest RAG\Test\test_rag_pipeline.py -s -vv
	```

Notes:
- Le test lit un CSV d'évaluation `RAG/Test/test_rag.csv` (encodage latin-1, séparateur `;`).
- À chaque exécution, un fichier de résultats horodaté est généré dans `RAG/Test/` sous la forme `test_rag_results_YYYYMMDD_HHMMSS.csv`.
- Assure-toi d'avoir un index FAISS ou d'ajouter un document avant de poser des questions (sinon erreur de cache manquant).





