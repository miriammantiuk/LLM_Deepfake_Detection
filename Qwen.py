import dashscope
import os
import json
from pathlib import Path
import cv2

# API setup
dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'
api_key = os.getenv("QWEN_API_KEY")
if not api_key:
    raise RuntimeError("QWEN_API_KEY environment variable not set.")

# Select model
# model='qwen3-vl-32b-instruct',
# model='qwen3-vl-235b-a22b-instruct',
# model='qwen3-vl-235b-a22b-thinking'
model='qwen3.5-397b-a17b'

# Thinking configuration
enable_thinking = True  # True = reasoning enabled, False = disabled

# Indicators configuration
indicators_type = "disabled"  # "enabled" for indicators prompt, "disabled" for baseline

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

# Path to videos
video_ordner = Path(r"C:\Users\miria\Deepfake_Detection\data\processed")
output_datei = Path(rf"C:\Users\miria\Deepfake_Detection\{model}{suffix}_3.json")

alle_ergebnisse = []
vorhandene_video_ids = set()
if output_datei.exists():
    try:
        with open(output_datei, 'r', encoding='utf-8') as f:
            alle_ergebnisse = json.load(f)
            vorhandene_video_ids = {entry.get("video_id") for entry in alle_ergebnisse if "video_id" in entry}
        print(f"Loaded existing results: {len(alle_ergebnisse)} videos")
    except json.JSONDecodeError:
        print("Error loading existing JSON file. Starting with empty list.")

# Process all MP4 files in the folder (recursive)
for video_pfad in video_ordner.glob("**/*.mp4"):
    video_id = video_pfad.stem
    if video_id in vorhandene_video_ids:
        print(f"--- Skipping {video_pfad.name} (already processed) ---")
        continue

    print(f"\n--- Processing: {video_pfad.name} ---")
    # Validate video
    cap = cv2.VideoCapture(str(video_pfad))
    if not cap.isOpened():
        print("Error: could not open video.")
        cap.release()
        continue
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video loaded: {frames} frames at {fps} FPS")
    cap.release()

    video_id = video_pfad.stem

    prompt_text = (prompt_indicators if add_indicators else prompt_baseline).format(video_id=video_id)

    try:

        messages = [
            {
                "role": "user",
                "content": [
                    {"video": str(video_pfad), "fps": 2},  # 2 fps to reduce API token usage
                    {"text": prompt_text}
                ]
            }
        ]

        # API call
        response = dashscope.MultiModalConversation.call(
            api_key=api_key,
            model=model,
            messages=messages,
            enable_thinking=enable_thinking
        )
        # Validate response
        if not response or not hasattr(response, 'output') or response.output is None:
            print(f"API error: invalid response for {video_pfad.name}")
            continue

        response_text = response.output.choices[0].message.content[0]["text"]
        print(f"API response: {response_text}")

        # Parse JSON (strip markdown code block if present)
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1].strip()
            if response_text.startswith("json"):
                response_text = response_text[4:].strip()

        result = json.loads(response_text)
        alle_ergebnisse.append(result)
        print("✓ Successfully parsed")

        # Save incrementally after each successful video
        with open(output_datei, 'w', encoding='utf-8') as f:
            json.dump(alle_ergebnisse, f, indent=2, ensure_ascii=False)
        print(f"Results saved incrementally ({len(alle_ergebnisse)} videos).")

    except Exception as e:
        print(f"Error processing {video_pfad.name}: {e}")
        continue

# Final save
with open(output_datei, 'w', encoding='utf-8') as f:
    json.dump(alle_ergebnisse, f, indent=2, ensure_ascii=False)

print(f"\nResults saved to: {output_datei}")
