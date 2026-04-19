import google.generativeai as genai
import os
import time
import json
from pathlib import Path

# --- KONFIGURATION ---
api_key_file = Path("gemini_api_key.txt")
if not api_key_file.exists():
    raise RuntimeError("gemini_api_key.txt nicht gefunden.")

with open(api_key_file, 'r', encoding='utf-8') as f:
    api_key = f.read().strip()

genai.configure(api_key=api_key)

# --- DEBUG: Verfügbare Modelle prüfen ---
# Dies hilft gegen den 404 Fehler, falls die ID leicht anders heißt
print("--- Verfügbare Modelle in deinem Account ---")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            # Filtert nur neuere Modelle, um die Liste kurz zu halten
            if "gemini" in m.name: 
                print(f"Modell gefunden: {m.name}")
except Exception as e:
    print(f"Warnung: Konnte Modelle nicht listen: {e}")
print("------------------------------------------\n")