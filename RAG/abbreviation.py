import re
import time
import json
import PyPDF2
import google.generativeai as genai
import os
from tqdm import tqdm

# ----------- 1. Extraction depuis le PDF -----------

def extraire_premiere_phrase_abreviations(pdf_path):
    texte = ""
    with open(pdf_path, "rb") as f:
        lecteur = PyPDF2.PdfReader(f)
        for page in lecteur.pages:
            texte += page.extract_text() + " "

    texte = re.sub(r"\s+", " ", texte)
    phrases = re.split(r'(?<=[.!?])\s+', texte)

    pattern = r"([A-ZÉÈÊÀÂÎÔÛa-zéèêàâîôûç'’\-]{2,}(?:\s+[A-ZÉÈÊÀÂÎÔÛa-zéèêàâîôûç'’\-]{2,}){0,9})\s*\(([A-Z][A-Z0-9\.]{1,10})\)"

    premieres_occurrences = {}

    for phrase in phrases:
        correspondances = re.findall(pattern, phrase)
        for _, abbr in correspondances:
            abbr = abbr.strip()
            if abbr.isdigit() or len(abbr) < 2 or len(abbr) > 10:
                continue
            if abbr not in premieres_occurrences:
                premieres_occurrences[abbr] = phrase.strip()

    return [(abbr, phrase) for abbr, phrase in premieres_occurrences.items()]


# ----------- 2. Appel groupé à Gemini -----------

def demander_definitions_groupe(abrev_phrases, model):
    """
    Envoie un lot de 10 abréviations + phrases à Gemini et récupère les définitions en JSON.
    """
    prompt = (
        "Voici une liste d'abréviations et leurs phrases. "
        "Pour chacune, retourne un objet JSON au format : "
        '{"abréviation": "...", "définition": "..."} ou {"abréviation": "...", "définition": null} '
        "si la définition n’est pas identifiable dans la phrase.\n\n"
        "Liste :\n"
    )

    for i, (abbr, phrase) in enumerate(abrev_phrases, 1):
        prompt += f"{i}. Abréviation : {abbr}\n   Phrase : {phrase}\n"

    prompt += "\nRéponds uniquement avec un JSON contenant une liste, par exemple :\n" \
              '[{"abréviation": "FMI", "définition": "Fonds monétaire international"}, ...]'

    try:
        response = model.generate_content(prompt)
        texte = response.text.strip()

        # Nettoyer les caractères parasites autour du JSON
        json_str = re.search(r'\[.*\]', texte, re.S)
        if json_str:
            return json.loads(json_str.group(0))
        else:
            # Si Gemini n’a pas bien formaté le JSON, on retourne nulls
            return [{"abréviation": abbr, "définition": None} for abbr, _ in abrev_phrases]

    except Exception as e:
        print(f"⚠️ Erreur Gemini : {e}")
        return [{"abréviation": abbr, "définition": None} for abbr, _ in abrev_phrases]


# ----------- 3. Traitement en lots et sauvegarde JSON -----------

def traiter_en_lots_json(abrev_phrases, model, taille_lot=10, delai=4, sortie_json="log/definitions.json"):
    os.makedirs(os.path.dirname(sortie_json), exist_ok=True)

    resultat_dict = {}

    # Nombre total de lots
    total_lots = (len(abrev_phrases) + taille_lot - 1) // taille_lot
    
    # Barre de progression pour les lots
    with tqdm(total=total_lots, desc="Traitement des lots", unit="lot") as pbar:
        for i in range(0, len(abrev_phrases), taille_lot):
            lot = abrev_phrases[i:i+taille_lot]
            lot_index = i // taille_lot + 1

            # Appel du modèle pour ce lot
            reponses = demander_definitions_groupe(lot, model)

            for j, (abbr, phrase) in enumerate(tqdm(lot, desc=f"Lot {lot_index}", leave=False)):
                definition = None
                if j < len(reponses):
                    definition = reponses[j].get("définition", None)
                resultat_dict[abbr] = definition

            # pause entre lots pour respecter le rate limit
            time.sleep(delai)
            pbar.update(1)
    # Sauvegarder le JSON
    with open(sortie_json, "w", encoding="utf-8") as f:
        json.dump(resultat_dict, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Résultats enregistrés dans {sortie_json}")
    return resultat_dict


# ----------- 4. Pipeline abréviations -----------

def pipeline_abreviations(pdf_path):    
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-2.5-flash-lite")

    chemin_pdf = pdf_path
    abrev_phrases = extraire_premiere_phrase_abreviations(chemin_pdf)
    print("-------Gestion des abréviations-------")
    print(f"Nombre total d’abréviations trouvées : {len(abrev_phrases)}")

    resultat_dict=traiter_en_lots_json(abrev_phrases, model, taille_lot=10, delai=4, sortie_json="./RAG/log/definitions.json")

    return resultat_dict
