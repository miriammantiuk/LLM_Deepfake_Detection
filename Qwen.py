import dashscope
import os
import json
from pathlib import Path
import cv2

# API-Setup
dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'
api_key = os.getenv("QWEN_API_KEY")
if not api_key:
    raise RuntimeError("QWEN_API_KEY Umgebungsvariable nicht gesetzt.")

# Wähle das Modell
# model='qwen3-vl-32b-instruct',
# model='qwen3-vl-235b-a22b-instruct',
# model='qwen3-vl-235b-a22b-thinking'
model='qwen3.5-397b-a17b'

# Thinking-Konfiguration
enable_thinking = True  # True reasoning, False für kein reasoning

# Indicators-Konfiguration
indicators_type = "disabled"  # "enabled" für indicators, "disabled" für baseline

suffix_parts = []
if enable_thinking:
    suffix_parts.append("thinking")
if indicators_type == "enabled":
    suffix_parts.append("indicators")
suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""

# Indicators
add_indicators = indicators_type == "enabled"

prompt_baseline = """<role>
                        You are an expert in deepfake detection with specialization in multimodal forensics.
                        </role>

                        <task>
                        Analyze the provided video and evaluate whether it is real or fake.
                        </task>

                        <output_format>
                        Return the result as a valid JSON object with exactly the following fields:
                        {{
                        "video_id": "{video_id}",
                        "assessment": "<Real or Fake>",
                        "probability_fake": <integer 0-100>,
                        "justification": "<max. 5 bullet points>"
                        }}
                        </output_format>

                        <constraints>
                        - Do NOT output explanations outside the JSON.
                        - Keep the justification concise, factual, and forensic.
                        </constraints>        
                        """

prompt_indicators = """<role>
                        You are an expert in deepfake detection with specialization in multimodal forensics.
                        </role>

                        <task>
                        Analyze the provided video and evaluate whether it is real or fake.
                        Your analysis must be strictly based on the indicators listed below.
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
                        {{
                        "video_id": "{video_id}",
                        "assessment": "<Real or Fake>",
                        "probability_fake": <integer 0-100>,
                        "justification": "<max. 5 bullet points>"
                        }}
                        </output_format>

                        <constraints>
                        - Do NOT output explanations outside the JSON.
                        - Keep the justification concise, factual, and forensic.
                        </constraints>"""

# Pfad zu den Videos
video_ordner = Path(r"C:\Users\miria\Deepfake_Detection\data\processed")
output_datei = Path(rf"C:\Users\miria\Deepfake_Detection\{model}{suffix}_3.json")

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

# Analysiere alle MP4-Dateien im Ordner (rekursiv in Unterordnern)
for video_pfad in video_ordner.glob("**/*.mp4"):
    video_id = video_pfad.stem
    if video_id in vorhandene_video_ids:
        print(f"--- Überspringe {video_pfad.name} (bereits analysiert) ---")
        continue
    
    print(f"\n--- Analysiere: {video_pfad.name} ---")
    # Video-Überprüfung
    cap = cv2.VideoCapture(str(video_pfad))
    if not cap.isOpened():
        print("Fehler: Video konnte nicht geladen werden.")
        cap.release()
        continue
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video geladen: {frames} Frames bei {fps} FPS")
    cap.release()
    
    video_id = video_pfad.stem  # Extrahiere die Video-ID hier

    prompt_text = (prompt_indicators if add_indicators else prompt_baseline).format(video_id=video_id)

    try:

        messages = [
            {
                "role": "user",
                "content": [
                    {"video": str(video_pfad), "fps": 2}, #Standardmäßig 2 fps um API-Token zu sparen. Kann je nach Video und Anforderung angepasst werden.
                    {"text": prompt_text}
                ]
            }
        ]
        
        # API-Aufruf
        response = dashscope.MultiModalConversation.call(
            api_key=api_key,
            model=model,
            messages=messages,
            enable_thinking=enable_thinking
        )
        # Prüfe, ob die Antwort gültig ist
        if not response or not hasattr(response, 'output') or response.output is None:
            print(f"API-Fehler: Ungültige Antwort für {video_pfad.name}")
            continue
        
        response_text = response.output.choices[0].message.content[0]["text"]
        print(f"API-Antwort: {response_text}")
        
        # JSON parsen (Markdown-Codeblock bereinigen)
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1].strip()
            if response_text.startswith("json"):
                response_text = response_text[4:].strip()
        
        result = json.loads(response_text)
        alle_ergebnisse.append(result)
        print("✓ Erfolgreich geparst")
        
        # Zwischenspeichern nach jedem erfolgreichen Video
        with open(output_datei, 'w', encoding='utf-8') as f:
            json.dump(alle_ergebnisse, f, indent=2, ensure_ascii=False)
        print(f"Ergebnisse zwischengespeichert ({len(alle_ergebnisse)} Videos).")
        
    except Exception as e:
        print(f"Fehler bei {video_pfad.name}: {e}")
        continue

# Ergebnisse speichern
with open(output_datei, 'w', encoding='utf-8') as f:
    json.dump(alle_ergebnisse, f, indent=2, ensure_ascii=False)

print(f"\nErgebnisse gespeichert in: {output_datei}")