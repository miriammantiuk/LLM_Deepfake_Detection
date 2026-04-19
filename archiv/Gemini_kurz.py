# import google.generativeai as genai
from google import genai
import os
import time
import json
from pathlib import Path

# --- KONFIGURATION ---
# Lade API-Key aus lokaler Datei
api_key_file = Path("gemini_api_key.txt")  # Datei im selben Ordner wie das Skript
if not api_key_file.exists():
    raise RuntimeError("gemini_api_key.txt nicht gefunden. Erstelle die Datei mit deinem API-Key.")

with open(api_key_file, 'r', encoding='utf-8') as f:
    api_key = f.read().strip()

if not api_key:
    raise RuntimeError("API-Key in gemini_api_key.txt ist leer.")

client = genai.Client(api_key=api_key)
print("Konfiguration erfolgreich.")

# 1. Definiere den Ordner und die Ausgabedatei
video_ordner = Path(r"C:\Users\miria\Deepfake_Detection\data\processed\1")

# 2. Wähle das Modell (ändere hier den Modellnamen)
model = "gemini-2.5-flash"
# model = "gemini-2.5-pro"

# Dynamischer Output-Name basierend auf dem Modellnamen
output_datei = Path(rf"C:\Users\miria\Deepfake_Detection\{model}_kurz.json")
alle_ergebnisse = []
vorhandene_video_ids = set()
if output_datei.exists():
    try:
        with open(output_datei, 'r', encoding='utf-8') as f:
            alle_ergebnisse = json.load(f)
            vorhandene_video_ids = {entry.get("video_id") for entry in alle_ergebnisse if "video_id" in entry}
        print(f"Geladene bestehende Ergebnisse: {len(alle_ergebnisse)} Videos")
    except json.JSONDecodeError:
        print("Fehler beim Laden der bestehenden JSON-Datei. Starte mit leerer Liste.")

# 3. Schleife durch alle mp4-Dateien
for video_pfad in video_ordner.glob("*.mp4"):
    video_id = video_pfad.stem
    if video_id in vorhandene_video_ids:
        print(f"--- Überspringe {video_pfad.name} (bereits analysiert) ---")
        continue
    
    print(f"\n--- Beginne Analyse für: {video_pfad.name} ---")
    
    try:
        # 4. Lade Video hoch
        print("Lade Video hoch...")
        video_file = client.files.upload(file=video_pfad)
        
        # 5. Warte auf Verarbeitung (optional, da API automatisch wartet)
        print("Video wird verarbeitet...")
        while video_file.state.name == "PROCESSING":
            print("Warte auf Verarbeitung...")
            time.sleep(10)
            video_file = client.files.get(name=video_file.name)
        
        if video_file.state.name == "FAILED":
            print(f"Verarbeitung für {video_pfad.name} fehlgeschlagen.")
            continue
            
        print("Video ist bereit.")
        
        # 6. Analysiere das Video
        prompt = """<role>
                    You are an expert in deepfake detection with specialization in multimodal forensics.
                    </role>

                    <task>
                    Analyze the provided video and evaluate whether it is real or fake.
                    Use the video_id provided in the prompt and copy it exactly into the output.
                    </task>

                    <output_format>
                    Return the result as a valid JSON object with exactly the following fields:
                    {
                    "video_id": "<use the video_id given in the prompt>",
                    "assessment": "<Real or Fake>",
                    "probability_fake": <integer 0-100>,
                    "justification": "<max. 5 bullet points>"
                    }
                    </output_format>

                    <constraints>
                    - Do NOT output explanations outside the JSON.
                    - Keep the justification concise, factual, and forensic.
                    </constraints>
                    """
        
        response = client.models.generate_content(
            model=model,
            contents=[video_file, prompt]
        )
        
        print(f"Antwort für {video_pfad.name}:\n{response.text}\n")
        
        # 7. Parse JSON und speichere es
        try:
            response_text = response.text.strip()
            
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()
            
            result = json.loads(response_text)
            result["video_id"] = video_pfad.stem
            alle_ergebnisse.append(result)
            print(f"✓ JSON erfolgreich geparst für {video_pfad.name}")
            
        except json.JSONDecodeError as e:
            print(f"✗ JSON Parse-Fehler für {video_pfad.name}: {e}")
            print(f"  Raw Response: {response.text[:200]}")
        
        # 9. Aufräumen
        #client.files.delete(file=video_pfad)
        #print("Datei vom Server gelöscht.")

    except Exception as e:
        print(f"Ein unerwarteter Fehler bei {video_pfad.name}: {e}")
        continue

# 10. Speichere alle Ergebnisse in eine JSON-Datei
with open(output_datei, 'w', encoding='utf-8') as f:
    json.dump(alle_ergebnisse, f, indent=2, ensure_ascii=False)

print(f"\n--- Alle Videos analysiert ---")
print(f"Ergebnisse gespeichert in: {output_datei}")

