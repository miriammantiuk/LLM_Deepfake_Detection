import os
import json
import base64
from pathlib import Path
import cv2
from openai import OpenAI

# =========================
# API setup (OpenRouter)
# =========================
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set.")

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
)

model = "bytedance-seed/seed-2.0-lite"

# Thinking configuration
thinking_type = False  # True = reasoning enabled, False = disabled

# Indicators configuration
indicators_type = "disabled"  # "enabled" for indicators prompt, "disabled" for baseline

suffix_parts = []
if thinking_type == True:
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

# =========================
# Paths
# =========================
video_ordner = Path(r"C:\Users\miria\Deepfake_Detection\data\processed")
output_datei = Path(rf"C:\Users\miria\Deepfake_Detection\{model.split('/')[-1]}{suffix}_3.json")

# =========================
# Helper functions
# =========================
def encode_video_to_base64(video_path: Path) -> str:
    """Reads the video file and returns it as a Base64-encoded string."""
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode("utf-8")

def clean_json_text(text: str) -> str:
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
# Load existing results
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
        print(f"Loaded existing results: {len(alle_ergebnisse)} videos")
    except json.JSONDecodeError:
        print("Error loading existing JSON file. Starting fresh.")

# =========================
# Process all MP4 files (recursive)
# =========================
for video_pfad in video_ordner.glob("**/*.mp4"):
    video_id = video_pfad.stem

    if video_id in vorhandene_video_ids:
        print(f"--- Skipping {video_pfad.name} (already processed) ---")
        continue

    print(f"\n--- Processing with Seed-2.0-Lite: {video_pfad.name} ---")

    # Validate video
    cap = cv2.VideoCapture(str(video_pfad))
    if not cap.isOpened():
        print("Error: could not open video.")
        cap.release()
        continue
    cap.release()

    prompt_text = (prompt_indicators if add_indicators else prompt_baseline).format(video_id=video_id).strip()

    try:
        # Encode video as Base64 for API transfer
        print("Preparing video data...")
        video_base64 = encode_video_to_base64(video_pfad)

        # Request to OpenRouter (Seed-2.0-lite)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_text
                        },
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": f"data:video/mp4;base64,{video_base64}"
                            }
                        }
                    ],
                }
            ],
            extra_body={
                "reasoning": {"enabled": thinking_type}
            }
        )

        response_text = completion.choices[0].message.content
        if not response_text:
            print(f"Error: empty response from OpenRouter for {video_pfad.name}")
            continue

        response_text = clean_json_text(response_text)
        print(f"API response: {response_text}")

        result = json.loads(response_text)
        result["video_id"] = video_id

        alle_ergebnisse.append(result)
        vorhandene_video_ids.add(video_id)

        save_results(output_datei, alle_ergebnisse)
        print(f"✓ {video_id} successfully processed.")

    except Exception as e:
        print(f"Critical error processing {video_pfad.name}: {e}")
        continue

print(f"\nDone! Results saved to: {output_datei}")
