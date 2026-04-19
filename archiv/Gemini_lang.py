import google.generativeai as genai
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

genai.configure(api_key=api_key)
print("Konfiguration erfolgreich.")

# 1. Definiere den Ordner und die Ausgabedatei
video_ordner = Path(r"C:\Users\miria\Deepfake_Detection\data\processed\2")
# output_datei = Path(r"C:\Users\miria\Deepfake_Detection\gemini-2.5-flash_indicators.json")
output_datei = Path(r"C:\Users\miria\Deepfake_Detection\gemini_3_flash_indicators.json")

# 2. Lade bestehende Ergebnisse, falls vorhanden
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

# 3. Wähle das Modell
# model = genai.GenerativeModel("gemini-2.5-flash")
model = genai.GenerativeModel("models/gemini-3-flash-preview")


# 4. Schleife durch alle mp4-Dateien
for video_pfad in video_ordner.glob("*.mp4"):
    video_id = video_pfad.stem
    if video_id in vorhandene_video_ids:
        print(f"--- Überspringe {video_pfad.name} (bereits analysiert) ---")
        continue
    
    print(f"\n--- Beginne Analyse für: {video_pfad.name} ---")
    
    try:
        # 5. Lade Video hoch
        print("Lade Video hoch...")
        video_file = genai.upload_file(path=video_pfad)
        
        # 6. Warte auf Verarbeitung
        print("Video wird verarbeitet...")
        while video_file.state.name == "PROCESSING":
            time.sleep(10)
            video_file = genai.get_file(video_file.name)
        
        if video_file.state.name == "FAILED":
            print(f"Verarbeitung für {video_pfad.name} fehlgeschlagen.")
            continue
            
        print("Video ist bereit.")
        
        # 7. Analysiere das Video
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
        
        response = model.generate_content([prompt, video_file])
        
        print(f"Antwort für {video_pfad.name}:\n{response.text}\n")
        
        # 8. Parse JSON und speichere es
        try:
            # Versuche JSON zu extrahieren, falls es in Text eingebettet ist
            response_text = response.text.strip()
            
            # Falls die Antwort mit ``` umgeben ist, entferne diese
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()
            
            result = json.loads(response_text)
            result["video_id"] = video_pfad.stem  # Füge Video-Namen hinzu falls fehlt
            alle_ergebnisse.append(result)
            print(f"✓ JSON erfolgreich geparst für {video_pfad.name}")
            
        except json.JSONDecodeError as e:
            print(f"✗ JSON Parse-Fehler für {video_pfad.name}: {e}")
            print(f"  Raw Response: {response.text[:200]}")
        
        # 9. Aufräumen
        genai.delete_file(video_file.name)
        print("Datei vom Server gelöscht.")

    except Exception as e:
        print(f"Ein unerwarteter Fehler bei {video_pfad.name}: {e}")
        continue

# 10. Speichere alle Ergebnisse in eine JSON-Datei
with open(output_datei, 'w', encoding='utf-8') as f:
    json.dump(alle_ergebnisse, f, indent=2, ensure_ascii=False)

print(f"\n--- Alle Videos analysiert ---")
print(f"Ergebnisse gespeichert in: {output_datei}")

