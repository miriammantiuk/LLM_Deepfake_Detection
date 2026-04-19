# import google.generativeai as genai
from google import genai
import os
import time
import json
from pathlib import Path
from google.genai import types

# --- KONFIGURATION ---
# Pfad zur JSON-Datei mit API-Keys
api_keys_file = Path("api_keys.json")

# Überprüfe, ob die Datei existiert
if not api_keys_file.exists():
    raise RuntimeError(f"API-Keys-Datei '{api_keys_file}' nicht gefunden. Erstelle eine JSON-Datei mit dem Format: {{'keys': ['key1', 'key2', ...]}}")

# Lade die API-Keys aus der JSON-Datei
try:
    with open(api_keys_file, 'r', encoding='utf-8') as f:
        keys_data = json.load(f)
    valid_keys = keys_data.get("keys", [])
    if not valid_keys or not isinstance(valid_keys, list):
        raise RuntimeError("Ungültiges Format in api_keys.json. Erwarte: {'keys': ['key1', 'key2', ...]}")
except json.JSONDecodeError as e:
    raise RuntimeError(f"Fehler beim Laden der API-Keys-Datei: {e}")

# Lade den ersten API-Key
current_key_index = 0
api_key = valid_keys[current_key_index].strip()

if not api_key:
    raise RuntimeError(f"API-Key {current_key_index + 1} ist leer.")

client = genai.Client(api_key=api_key)
print(f"Konfiguration erfolgreich mit Key {current_key_index + 1}.")

# Funktion zum Wechseln des API-Keys
def switch_api_key():
    global current_key_index, api_key, client
    current_key_index = (current_key_index + 1) % len(valid_keys)
    api_key = valid_keys[current_key_index].strip()
    client = genai.Client(api_key=api_key)
    print(f"API-Key gewechselt zu Key {current_key_index + 1}.")

# Zähler für Videos pro Key
videos_processed = 0
videos_per_key = 20

# 1. Definiere den Ordner und die Ausgabedatei
video_ordner = Path(r"C:\Users\miria\Deepfake_Detection\data\processed")

# 2. Wähle das Modell (ändere hier den Modellnamen)
model = "gemini-3-flash-preview"
# model = "gemini-2.5-flash"
# model = "gemini-2.5-pro"

# Dynamischer Output-Name basierend auf dem Modellnamen
output_datei = Path(rf"C:\Users\miria\Deepfake_Detection\{model}_indicators.json")
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

# 3. Schleife durch alle mp4-Dateien (rekursiv in Unterordnern)
videos_list = list(video_ordner.glob("**/*.mp4"))
for video_pfad in videos_list:
    video_id = video_pfad.stem
    if video_id in vorhandene_video_ids:
        print(f"--- Überspringe {video_pfad.name} (bereits analysiert) ---")
        continue
    
    # Zähler für Videos erhöhen (einmal pro Video)
    videos_processed += 1
    print(f"Videos verarbeitet: {videos_processed}")
    
    # API-Key-Rotation nach 20 Videos
    if videos_processed % videos_per_key == 0:
        switch_api_key()
        print(f"API-Key gewechselt nach {videos_processed} Videos.")
    
    processed = False
    retry_count = 0
    while not processed:
        try:
            print(f"\n--- Beginne Analyse für: {video_pfad.name} ---")
            
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
                        Your analysis must be strictly based on the indicators listed below.
                        Use the video_id provided in the prompt and copy it exactly into the output.
                        </task>

                        <indicators>
                        1. Unnatural body movements, stiff or robotic gestures, or inconsistent/missing facial expressions.
                        2. Facial irregularities: uneven skin tones, poor eye rendering, asymmetry, unnatural textures.
                        3. Inconsistencies in lighting, shadows, colors, reflections.
                        4. Signs of face swapping: unnatural lip-sync, distorted mouth shapes, abnormal eye movements or blinking.
                        5. Visual artifacts: blurring, pixel noise, edge glitches, compression issues, resolution inconsistencies, generator- or frequency-pattern anomalies.
                        6. Temporal instabilities: warping, jitter, frame jumps, changes in details across frames.
                        7. Logical coherence and contextual consistency of the visual content.
                        </indicators>

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
                contents=[video_file, prompt],
                config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0) # kein reasoning
                # thinking_config=types.ThinkingConfig(thinking_budget=-1) # dynamic reasoning
                # thinking_config=types.ThinkingConfig(thinking_level="high")
                )
            )
            
            print(f"Antwort für {video_pfad.name}:\n{response.text}\n")
            
            # 7. Parse JSON und speichere es
            try:
                response_text = response.text.strip()
                
                # Entferne alles vor dem ersten '{'
                start_idx = response_text.find('{')
                if start_idx != -1:
                    response_text = response_text[start_idx:]
                
                # Entferne alles nach dem letzten '}'
                end_idx = response_text.rfind('}')
                if end_idx != -1:
                    response_text = response_text[:end_idx + 1]
                
                if response_text.startswith("```"):
                    response_text = response_text.split("```")[1]
                    if response_text.startswith("json"):
                        response_text = response_text[4:]
                    response_text = response_text.strip()
                
                result = json.loads(response_text)
                result["video_id"] = video_pfad.stem
                alle_ergebnisse.append(result)
                print(f"✓ JSON erfolgreich geparst für {video_pfad.name}")
                
                # Zwischenspeichern nach jedem erfolgreichen Video
                with open(output_datei, 'w', encoding='utf-8') as f:
                    json.dump(alle_ergebnisse, f, indent=2, ensure_ascii=False)
                print(f"Ergebnisse zwischengespeichert ({len(alle_ergebnisse)} Videos).")
                
                processed = True
                
            except json.JSONDecodeError as e:
                print(f"✗ JSON Parse-Fehler für {video_pfad.name}: {e}")
                print(f"  Raw Response: {response.text[:200]}")
            
                # 9. Aufräumen
                #client.files.delete(file=video_pfad)
                #print("Datei vom Server gelöscht.")
                
                processed = True
        
        except Exception as e:
            error_msg = str(e).lower()
            if ("quota" in error_msg or "limit" in error_msg or "exhausted" in error_msg or "rate" in error_msg) and retry_count < len(valid_keys):
                print(f"API-Fehler erkannt bei {video_pfad.name}: {e}")
                switch_api_key()
                retry_count += 1
                print(f"Retry {retry_count} für {video_pfad.name} mit neuem Key (Key {current_key_index + 1}).")
            else:
                print(f"Fehler bei {video_pfad.name}: {e}")
                processed = True

# 10. Speichere alle Ergebnisse in eine JSON-Datei
with open(output_datei, 'w', encoding='utf-8') as f:
    json.dump(alle_ergebnisse, f, indent=2, ensure_ascii=False)

print(f"\n--- Alle Videos analysiert ---")
print(f"Ergebnisse gespeichert in: {output_datei}")

