import fitz  # PyMuPDF
import os
import json
from pathlib import Path
from typing import List, Dict, Optional
import re
import google.generativeai as genai
from langchain_core.documents import Document
import time
import logging

# -------------------------------
# Configuration
# -------------------------------
PDF_PATH = "./RAG/Dataset/HCC_RA_2025-18.07_web.pdf"
OUTPUT_DIR = "./RAG/Dataset/rag_figures/images" # Dossier pour sauvegarder les images de pages
MIN_DRAWING_ELEMENTS = 15
ZOOM_FACTOR = 3 # Haute résolution

# -------------------------------
# Fonctions utilitaires
# -------------------------------
def ensure_dir(path: str):
    """Crée un dossier s'il n'existe pas."""
    os.makedirs(path, exist_ok=True)

def _extract_page_num_from_filename(stem: str) -> Optional[int]:
    """Extrait le numéro de page depuis un nom de fichier style 'page_123.png' ou 'page_123_...'."""
    m = re.search(r"page_(\d+)", stem)
    return int(m.group(1)) if m else None

# -------------------------------
# Fonction principale
# -------------------------------
def save_identified_pages(pdf_path: str, out_dir: str, min_elements: int):
    """
    Parcourt un PDF, identifie les pages avec des figures potentielles,
    et sauvegarde chaque page identifiée comme une image PNG.
    """
    print(f"--- Lancement de la sauvegarde des pages pour : {os.path.basename(pdf_path)} ---")
    ensure_dir(out_dir)
    
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Erreur : Impossible d'ouvrir le fichier PDF '{pdf_path}'. Détails : {e}")
        return

    print(f"Le document contient {len(doc)} pages.")
    
    saved_pages_count = 0

    # Parcourir chaque page du document
    for i, page in enumerate(doc):
        page_num = i + 1
        
        # 1. Filtre par complexité
        drawings = page.get_drawings()
        if len(drawings) < min_elements:
            continue

        # 2. Filtre par mot-clé
        page_text = page.get_text("text", sort=True).lower()
        if "figure" not in page_text:
            continue

        print(f"✅ Page {page_num}: Figure potentielle détectée. Sauvegarde de la page entière...")

        # 3. Extraire la page entière en tant qu'image et la sauvegarder
        try:
            mat = fitz.Matrix(ZOOM_FACTOR, ZOOM_FACTOR)
            pix = page.get_pixmap(matrix=mat) # Pas de 'clip' pour avoir la page entière
            
            output_filename = f"page_{page_num}.png"
            output_path = os.path.join(out_dir, output_filename)
            
            pix.save(output_path)
            print(f"  -> Page sauvegardée : {output_path}")
            saved_pages_count += 1

        except Exception as e:
            print(f"  -> Erreur lors de la sauvegarde de la page {page_num}: {e}")

    doc.close()

    print("-" * 20)
    print("\nRésumé de la sauvegarde :")
    print(f"{saved_pages_count} pages ont été sauvegardées dans le dossier '{out_dir}'.")
    print("--- Fin de la sauvegarde ---")


def analyze_saved_pages_with_gemini(
    images_dir: str = OUTPUT_DIR,
    model_name: str = "gemini-2.5-flash-lite",
    save_summary_path: Optional[str] = "./RAG/Dataset/rag_figures/_summary.json",
) -> List[Dict]:
    """
    Parcourt toutes les images PNG présentes dans `images_dir`, envoie chaque image au modèle
    Gemini 2.0 Flash et gère le cas où le modèle renvoie plusieurs figures pour une même image.

    Normalisation: on transforme la sortie en une LISTE d'entrées (une par figure). Chaque entrée:
      {
        "source_page": int | None,
        "image_path": str,
        "figure_index_in_image": int,
        "analysis": {  # JSON de la figure
          "titre": str | null,
          "type_graphique": str | null,
          "axes": { "x": {"label": str|null, "unite": str|null}, "y": {"label": str|null, "unite": str|null} },
          "series": [ {"label": str|null, "tendance": str|null} ],
          "valeurs_cles": [str],
          "resume": str
        }
      }
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY manquant dans les variables d'environnement")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    images_path = Path(images_dir)
    if not images_path.exists():
        raise FileNotFoundError(f"Dossier d'images introuvable: {images_dir}")

    image_files = sorted(images_path.glob("*.png"))
    if not image_files:
        print(f"Aucune image PNG trouvée dans {images_dir}")
        return []

    print(f"Analyse Gemini de {len(image_files)} image(s) depuis {images_dir} ...")

    results: List[Dict] = []
    # Rate limiting: maximum 15 requests per minute -> minimum interval between calls
    min_interval = 60.0 / 15.0  # 4.0 seconds
    last_api_call = 0.0
    for img in image_files:
        try:
            page_num = _extract_page_num_from_filename(img.stem)

            # Imposer un retour sous forme de tableau JSON de figures
            prompt = (
                "Tu es un analyste de données. Analyse en français la/les figure(s) présente(s) dans l'image.\n"
                "RETOURNE STRICTEMENT un TABLEAU JSON de figures (même s'il n'y en a qu'une).\n"
                "Chaque élément du tableau doit avoir exactement la structure suivante:\n"
                "{\n"
                "  \"titre\": string | null,\n"
                "  \"type_graphique\": string | null,\n"
                "  \"axes\": {\n"
                "    \"x\": { \"label\": string | null, \"unite\": string | null },\n"
                "    \"y\": { \"label\": string | null, \"unite\": string | null }\n"
                "  },\n"
                "  \"series\": [ { \"label\": string | null, \"tendance\": string | null } ],\n"
                "  \"valeurs_cles\": [string],\n"
                "  \"resume\": string\n"
                "}\n"
                "Si une information est absente, mets null. NE RENVOIE QUE LE TABLEAU JSON, sans texte additionnel."
            )

            with open(img, "rb") as f:
                image_bytes = f.read()

            # Enforce rate limit
            now = time.monotonic()
            elapsed = now - last_api_call
            if elapsed < min_interval:
                wait_for = min_interval - elapsed
                logging.info(f"Rate limit: waiting {wait_for:.2f}s before calling Gemini for {img.name}")
                time.sleep(wait_for)

            response = model.generate_content([
                prompt,
                {"mime_type": "image/png", "data": image_bytes},
            ])
            last_api_call = time.monotonic()

            raw = (response.text or "").strip()
            clean = raw.replace("```json", "").replace("```", "").strip()
            print(clean)
            parsed = json.loads(clean)
            # Normalisation des différentes formes possibles
            if isinstance(parsed, dict) and "figures" in parsed and isinstance(parsed["figures"], list):
                figures = parsed["figures"]
            elif isinstance(parsed, dict):
                figures = [parsed]
            elif isinstance(parsed, list):
                figures = parsed
            else:
                raise ValueError("Réponse JSON inattendue: attendue liste ou objet")

            for idx, fig in enumerate(figures):
                results.append({
                    "source_page": page_num,
                    "image_path": str(img),
                    "figure_index_in_image": idx,
                    "analysis": fig,
                })
            print(f"  -> OK: {img.name} ({len(figures)} figure(s))")
        except Exception as e:
            print(f"  -> Échec: {img.name}: {e}")
            continue

    if save_summary_path:
        try:
            out = Path(save_summary_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Résumé sauvegardé dans {out}")
        except Exception as e:
            print(f"Impossible d'écrire le résumé: {e}")

    return results

def load_figure_analyses(summary_path: str) -> List[Document]:
    """
    Lit le fichier _summary.json et convertit chaque entrée en un Document LangChain.
    """
    p = Path(summary_path)
    if not p.exists():
        return []

    with open(p, "r", encoding="utf-8") as f:
        arr = json.load(f)

    docs: List[Document] = []
    for item in arr:
        analysis = item.get("analysis", {}) if isinstance(item, dict) else {}
        title = analysis.get("titre") if isinstance(analysis, dict) else None
        gtype = analysis.get("type_graphique") if isinstance(analysis, dict) else None
        axes = analysis.get("axes", {}) if isinstance(analysis, dict) else {}
        x = axes.get("x", {}) if isinstance(axes, dict) else {}
        y = axes.get("y", {}) if isinstance(axes, dict) else {}
        series = analysis.get("series", []) if isinstance(analysis, dict) else []
        valeurs = analysis.get("valeurs_cles", []) if isinstance(analysis, dict) else []
        resume = analysis.get("resume", "") if isinstance(analysis, dict) else ""

        parts = []
        if title:
            parts.append(f"Titre: {title}")
        if gtype:
            parts.append(f"Type: {gtype}")
        if x:
            parts.append(f"Axe X: {x.get('label')} ({x.get('unite')})")
        if y:
            parts.append(f"Axe Y: {y.get('label')} ({y.get('unite')})")
        if series:
            ser_texts = [s.get("label") for s in series if isinstance(s, dict) and s.get("label")]
            if ser_texts:
                parts.append(f"Series: {', '.join(ser_texts)}")
        if valeurs:
            parts.append(f"Valeurs clés: {', '.join(valeurs)}")
        if resume:
            parts.append(f"Résumé: {resume}")

        content = "\n".join(parts) or "(figure)"
        metadata = {
            "source": "figure_analysis",
            "source_page": item.get("source_page"),
            "image_path": item.get("image_path"),
            "figure_index_in_image": item.get("figure_index_in_image"),
        }

        docs.append(Document(page_content=content, metadata=metadata))

    return docs
# -------------------------------
# Exécution
# -------------------------------
if __name__ == "__main__":
    save_identified_pages(PDF_PATH, OUTPUT_DIR, MIN_DRAWING_ELEMENTS)
    analyze_saved_pages_with_gemini(OUTPUT_DIR)