import os
import json
from pathlib import Path

import cv2
from openai import OpenAI

# =========================
# API-Setup
# =========================
api_key = os.getenv("MOONSHOT_API_KEY")
if not api_key:
    raise RuntimeError("MOONSHOT_API_KEY Umgebungsvariable nicht gesetzt.")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.moonshot.ai/v1",
)

# Kimi 2.5 Modell
model = "kimi-k2.5"

# Thinking-Konfiguration
thinking_type = "disabled"  # "enabled" reasoning, "disabled" für kein reasoning

# Indicators-Konfiguration
indicators_type = "disabled"  # "enabled" für indicators, "disabled" für baseline

suffix_parts = []
if thinking_type == "enabled":
    suffix_parts.append("thinking")
if indicators_type == "enabled":
    suffix_parts.append("indicators")
suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""

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

# Indicators
add_indicators = indicators_type == "enabled"
prompt_text = prompt_indicators if add_indicators else prompt_baseline


# =========================
# Pfade
# =========================
video_ordner = Path(r"C:\Users\miria\Deepfake_Detection\data\processed")
output_datei = Path(rf"C:\Users\miria\Deepfake_Detection\{model}{suffix}_3.json")

# =========================
# Hilfsfunktionen
# =========================
def clean_json_text(text: str) -> str:
    """Bereinigt ggf. Markdown-Codeblöcke um JSON-Text."""
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1].strip()
        if text.startswith("json"):
            text = text[4:].strip()
    return text

def save_results(path: Path, results: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

# =========================
# Bereits vorhandene Ergebnisse laden
# =========================
alle_ergebnisse = []
vorhandene_video_ids = set()

if output_datei.exists():
    try:
        with open(output_datei, "r", encoding="utf-8") as f:
            alle_ergebnisse = json.load(f)
            vorhandene_video_ids = {
                entry.get("video_id")
                for entry in alle_ergebnisse
                if isinstance(entry, dict) and "video_id" in entry
            }
        print(f"Geladene bestehende Ergebnisse: {len(alle_ergebnisse)} Videos")
    except json.JSONDecodeError:
        print("Fehler beim Laden der bestehenden JSON-Datei. Starte mit leerer Liste.")

# =========================
# Analysiere alle MP4-Dateien rekursiv
# =========================
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

    cap.release()

    prompt_text = (prompt_indicators if add_indicators else prompt_baseline).format(video_id=video_id).strip()

    try:
        # 1) Video zu Moonshot hochladen
        print("Lade Video zu Moonshot hoch ...")
        uploaded_file = client.files.create(
            file=video_pfad,
            purpose="video",
        )

        video_url = f"ms://{uploaded_file.id}"

        # 2) Anfrage an Kimi 2.5
        completion = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": video_url
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt_text
                        }
                    ],
                }
            ],
            extra_body={
                    "thinking": {"type": thinking_type}
            }, # Kimi 2.5 nutzt standardmäßig "reasoning"
        )

        response_text = completion.choices[0].message.content
        if not response_text:
            print(f"API-Fehler: Leere Antwort für {video_pfad.name}")
            continue

        response_text = clean_json_text(response_text)
        print(f"API-Antwort: {response_text}")

        result = json.loads(response_text)

        # Sicherheitsnetz: video_id notfalls setzen/korrigieren
        result["video_id"] = video_id

        alle_ergebnisse.append(result)
        vorhandene_video_ids.add(video_id)

        # Nach jedem erfolgreichen Video speichern
        save_results(output_datei, alle_ergebnisse)
        print("✓ Erfolgreich geparst und gespeichert")

    except Exception as e:
        print(f"Fehler bei {video_pfad.name}: {e}")
        continue

print(f"\nErgebnisse gespeichert in: {output_datei}")