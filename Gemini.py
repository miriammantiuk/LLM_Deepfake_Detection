# import google.generativeai as genai
from google import genai
import os
import time
import json
from pathlib import Path
from google.genai import types

# --- CONFIGURATION ---
# Path to the JSON file containing API keys
api_keys_file = Path("api_keys.json")

# Check if the file exists
if not api_keys_file.exists():
    raise RuntimeError(f"API keys file '{api_keys_file}' not found. Create a JSON file with the format: {{'keys': ['key1', 'key2', ...]}}")

# Load API keys from the JSON file
try:
    with open(api_keys_file, 'r', encoding='utf-8') as f:
        keys_data = json.load(f)
    valid_keys = keys_data.get("keys", [])
    if not valid_keys or not isinstance(valid_keys, list):
        raise RuntimeError("Invalid format in api_keys.json. Expected: {'keys': ['key1', 'key2', ...]}")
except json.JSONDecodeError as e:
    raise RuntimeError(f"Error loading API keys file: {e}")

# Load the first API key
current_key_index = 0
api_key = valid_keys[current_key_index].strip()

if not api_key:
    raise RuntimeError(f"API key {current_key_index + 1} is empty.")

client = genai.Client(api_key=api_key)
print(f"Configuration successful with key {current_key_index + 1}.")

# Switch to the next API key (used on quota errors)
def switch_api_key():
    global current_key_index, api_key, client
    current_key_index = (current_key_index + 1) % len(valid_keys)
    api_key = valid_keys[current_key_index].strip()
    client = genai.Client(api_key=api_key)
    print(f"Switched to API key {current_key_index + 1}.")

# Counter for videos processed per key
videos_processed = 0
videos_per_key = 20

# 1. Define input folder and output file
video_ordner = Path(r"C:\Users\miria\Deepfake_Detection\data\processed")

# 2. Select model
model = "gemini-3.1-flash-lite-preview"
# model = "gemini-3-flash-preview"
# model = "gemini-2.5-flash"
# model = "gemini-2.5-pro"

# Thinking configuration
thinking_budget = 0  # -1 = dynamic reasoning, 0 = disabled

# Indicators configuration
indicators_type = "enabled"  # "enabled" for indicators prompt, "disabled" for baseline

suffix_parts = []
if thinking_budget == -1:
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

prompt_indicators = """<role>
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
                        </constraints>"""

# Dynamic output filename based on model name and thinking budget
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

# 3. Iterate over all MP4 files (recursive)
videos_list = list(video_ordner.glob("**/*.mp4"))
for video_pfad in videos_list:
    video_id = video_pfad.stem
    if video_id in vorhandene_video_ids:
        print(f"--- Skipping {video_pfad.name} (already processed) ---")
        continue

    # API key rotation after every N videos
    # if videos_processed % videos_per_key == 0:
        # switch_api_key()
        # print(f"API key rotated after {videos_processed} videos.")

    processed = False
    retry_count = 0
    while not processed:
        try:
            print(f"\n--- Starting analysis for: {video_pfad.name} ---")

            # 4. Upload video
            print("Uploading video...")
            video_file = client.files.upload(file=video_pfad)

            # 5. Wait for processing
            print("Processing video...")
            while video_file.state.name == "PROCESSING":
                print("Waiting for processing...")
                time.sleep(10)
                video_file = client.files.get(name=video_file.name)

            if video_file.state.name == "FAILED":
                print(f"Processing failed for {video_pfad.name}.")
                continue

            print("Video is ready.")

            # 6. Analyse video
            prompt = prompt_indicators if add_indicators else prompt_baseline

            response = client.models.generate_content(
                model=model,
                contents=[video_file, prompt],
                config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget) #, temperature=0
                )
            )

            print(f"Response for {video_pfad.name}:\n{response.text}\n")

            # 7. Parse JSON and save
            try:
                response_text = response.text.strip()

                # Strip everything before the first '{'
                start_idx = response_text.find('{')
                if start_idx != -1:
                    response_text = response_text[start_idx:]

                # Strip everything after the last '}'
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
                print(f"✓ JSON successfully parsed for {video_pfad.name}")

                # Save incrementally after each successful video
                with open(output_datei, 'w', encoding='utf-8') as f:
                    json.dump(alle_ergebnisse, f, indent=2, ensure_ascii=False)
                print(f"Results saved incrementally ({len(alle_ergebnisse)} videos).")

                videos_processed += 1
                print(f"Videos processed: {videos_processed}")
            except json.JSONDecodeError as e:
                print(f"✗ JSON parse error for {video_pfad.name}: {e}")
                print(f"  Raw response: {response.text[:200]}")

                # 9. Cleanup (optional)
                #client.files.delete(file=video_pfad)
                #print("File deleted from server.")

            processed = True

        except Exception as e:
            error_msg = str(e).lower()
            if ("quota" in error_msg or "limit" in error_msg or "exhausted" in error_msg or "rate" in error_msg) and retry_count < len(valid_keys):
                print(f"API error for {video_pfad.name}: {e}")
                switch_api_key()
                retry_count += 1
                print(f"Retry {retry_count} for {video_pfad.name} with new key (key {current_key_index + 1}).")
            else:
                print(f"Error processing {video_pfad.name}: {e}")
                processed = True

# 10. Final save
with open(output_datei, 'w', encoding='utf-8') as f:
    json.dump(alle_ergebnisse, f, indent=2, ensure_ascii=False)

print(f"\n--- All videos processed ---")
print(f"Results saved to: {output_datei}")
