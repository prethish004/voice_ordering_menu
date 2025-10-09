from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import numpy as np
import soundfile as sf
import noisereduce as nr
import openai
from dotenv import load_dotenv
import os
import pandas as pd
import re
import subprocess
from collections import defaultdict
from io import StringIO
from pathlib import Path
import string
# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load default menu
menu_df = pd.DataFrame()
default_menu_path = "menu.csv"
if os.path.exists(default_menu_path):
    menu_df = pd.read_csv(default_menu_path)
    print(f"✅ Loaded default menu: {default_menu_path}")
else:
    print("⚠️ Default menu.csv not found. Please upload one via /upload-csv/")

@app.post("/upload-csv/")
async def upload_csv(csvFile: UploadFile = File(...)):
    try:
        contents = await csvFile.read()
        decoded_csv = contents.decode("utf-8")

        # Save to disk
        with open("menu 4.csv", "w", encoding="utf-8") as f:
            f.write(decoded_csv)

        # Load to memory
        global menu_df
        menu_df = pd.read_csv(StringIO(decoded_csv))

        return {
            "status": "success",
            "message": f"CSV file '{csvFile.filename}' uploaded successfully.",
            "rows": len(menu_df),
            "columns": list(menu_df.columns)
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

SUPPORTED_AUDIO_TYPES = {
    "audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3", "audio/webm",
    "audio/ogg", "audio/x-m4a", "audio/mp4", "audio/flac", "audio/aac",
    "audio/x-aiff"
}

SUPPORTED_VIDEO_TYPES = {
    "video/mp4", "video/webm", "video/x-m4v", "video/x-msvideo"
}

def denoise_audio(data, sr=16000):
    try:
        noise_sample = data[:sr] if len(data) >= sr else data
        if np.all(noise_sample == 0):
            return data
        y_nr = nr.reduce_noise(y=data, sr=sr, y_noise=noise_sample)
        return y_nr
    except Exception:
        return data

def convert_audio_to_wav(input_path, output_path):
    cmd = ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", "-f", "wav", output_path]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

def extract_audio_from_video(input_path, output_path):
    cmd = ["ffmpeg", "-y", "-i", input_path, "-vn", "-acodec", "mp3", "-ar", "16000", "-ac", "1", output_path]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

# def transcribe_audio(path):
#     try:
#         with open(path, "rb") as audio_file:
#             response = openai.Audio.transcribe(
#                 model="gpt-4o-mini-transcribe",
#                 file=audio_file
#             )
#         return response["text"]
#     except Exception as e:
#         print(f"❌ Transcription failed: {e}")
#         return ""
def transcribe_audio(path):
    try:
        with open(path, "rb") as audio_file:
            response = openai.Audio.transcribe(
                model="gpt-4o-transcribe",
                file=audio_file
            )
        return response["text"]
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        return ""


# def translate_text(transcribed_text):
#     """Translate text into English if it's Tamil/Hindi/other supported language."""
#     prompt = f"""
# You are a translation assistant. Detect the input language (English, Tamil, or Hindi).
# Translate the following customer transcript into **English (US)** without changing meaning:

# Transcript:
# \"\"\"{transcribed_text}\"\"\"

# Output ONLY the translated English text, nothing else.
# """
#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-4o",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0
#         )
#         return response['choices'][0]['message']['content'].strip()
#     except Exception as e:
#         print(f"❌ Translation failed: {e}")
#         return transcribed_text   # fallback: return original if translation fails
    
def translate_text(transcribed_text):
    """Translate and normalize transcript into clean English menu-style items."""
    prompt = f"""
You are a professional FOOD ORDER translation assistant.
Your job is to convert spoken or written orders in Tamil, Hindi, Telugu, or English (including mixed forms) into a **clean, structured English food menu list**.

Follow these rules carefully:

1. **Language & Translation**
   - Detect and translate Tamil, Hindi, Telugu, Tanglish, or mixed-language input into **English (US)**.
   - Keep dish names in their **original form** (e.g., “Pongal”, “Sambar Rice”, “Rasam”).

2. **Number Normalization**
   - Convert all numeric words (in Tamil, Hindi, Telugu, or English — even if misspelled) into digits.
   - Use approximate spelling recognition for common phonetic or speech-to-text variations.
     - Example: "monu" → "moonu" → 3  
       "randu" → "rendu" → 2  
       "airu" → "aaru" → 6  
       "paththu" → "pathu" → 10  
       "sevan" → "seven" → 7
   - Supported mappings:
     - **Tamil:** ஒன்று=1, இரண்டு=2, மூன்று=3, நான்கு=4, ஐந்து=5, ஆறு=6, ஏழு=7, எட்டு=8, ஒன்பது=9, பத்து=10  
     - **Tanglish:** onnu=1, rendu=2, moonu=3, naalu=4, anju=5, aaru=6, elu=7, ettu=8, onbathu=9, pathu=10  
     - **Hindi:** ek=1, do=2, teen=3, char=4, paanch=5, chhah=6, saat=7, aath=8, nau=9, das=10  
     - **Telugu:** okati=1, rendu=2, moodu=3, naalugu=4, aidu=5, aaru=6, edu=7, enimidhi=8, tommidi=9, padi=10  
     - **English:** one=1, two=2, three=3, four=4, five=5, six=6, seven=7, eight=8, nine=9, ten=10

3. **Item Preservation**
   - Preserve every food item mentioned — do not skip, infer, or summarize.

4. **Quantity Defaults**
   - If quantity is missing, assume it is 1.

5. **Output Format**
   - Use format: `<number> <item name>`, separated by commas.
   - Example: `4 Raita, 5 S.I. Appalam, 3 White Rice, 5 Pongal, 4 Sambar Rice`

6. **Clean Output**
   - Output only the final clean list — no explanations or extra text.

Transcript:
\"\"\"{transcribed_text}\"\"\"

Output:
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"❌ Translation failed: {e}")
        return transcribed_text   # fallback

correction_map = {
    "teo": "teh o",
    "teho": "teh o",
    "tehc": "teh c",
    "teh-c": "teh c",
    "tehc hot": "teh c hot",
    "teh-c hot": "teh c hot",
    "teho ice": "teh o ice",
    "teo ice": "teh o ice",
    "kopio": "kopi o",
    "kopio ice": "kopi o ice",
}
def apply_corrections(text, correction_map):
    for wrong, right in correction_map.items():
        text = text.replace(wrong, right)
    return text


# def gpt_menu_match(transcribed_text, menu_df):
def gpt_menu_match(transcribed_text, menu_df, live_transcript=""):
    # Translate first
    transcribed_text = apply_corrections(transcribed_text.lower(), correction_map)
    translated_text = translate_text(transcribed_text)
    # menu_str = ", ".join(menu_df["Item_Name"].astype(str).tolist())
    # menu_itemid = ", ".join(menu_df["Item_id"].astype(str).tolist())
    menu_str = ", ".join(f"{row['Item_id']} | {row['Item_Name']}" for _, row in menu_df.iterrows())
    prompt = f"""
You are an AI voice assistant for a restaurant in Singapore. Your job is to extract food items and their quantities from a spoken order transcript.
 
Here are two versions of the order:
1. Raw Transcription (may contain Tamil/English mix): 
\"\"\"{transcribed_text}\"\"\"
2. Machine Translated Transcript (may contain errors like 'rice lemonade' instead of 'Nasi Lemak'): 
\"\"\"{translated_text}\"\"\"
3. Live Transcript (browser SpeechRecognition, may contain noise): 
\"\"\"{live_transcript}\"\"\"

Menu items:
{menu_str}
Important: In this menu, "Mee" and "Mee Hoon" are **not the same**. Always treat "Mee Hoon" dishes as separate. Do not substitute "Mee Hoon" for "Mee" or vice versa.

Instructions:
1. Always use the raw transcription for identifying food names. Use the live transcript as supporting evidence (for numbers or corrections). Use the translated transcript only to understand filler words, numbers, or context. If there is a conflict, prioritize the raw transcription for dish names and then match with the menu list.
2. If a generic or vague term is spoken (e.g., “vada”), select the closest match from the menu **only if it clearly maps to a known menu item**. If there is ambiguity or no close match, treat it as **Unavailable**.
3. If dish names are incomplete or mispronounced, match them to the most likely correct dish using contextual understanding **only if the dish exists in the menu** and the match is strong. Do **not** guess or stretch uncertain terms into menu items.
   3A. If a dish name contains a prefix like "Mee Hoon" or "Mee", prioritize exact full-name matching. Do not collapse "Mee Hoon Goreng Mutton" into "Mee Goreng Mutton". These are distinct items and should not be confused.
   3B. Be careful with menu items that differ only by short suffixes or prefixes (e.g., “Mee Hoon” vs “Mee”, “Teh O” vs “Teh C”, “Maggi” vs “Mee Goreng”). These are distinct dishes. Only match when the full phrase is clearly present. Do not collapse one into another based on partial match.
   3C. Be careful with "Koli Soru / Koali Soaru / Kolisoaru" and similar pronunciations.Always match these ONLY to "Koali Soaru (Chicken Broth Rice)" from the menu, never to "Chicken Biryani" or other chicken rice dishes.
   3D. Dishes like “Mee Goreng”, “Maggi Goreng”, “Kway Teow Goreng”, and “Mee Hoon Goreng” are different types of fried noodles. Do not substitute one for another even if the ingredients (like anchovies) are similar. Use exact phrase match when possible.
   3E. Be careful with common mishearing issues:
       - "nokkari", "nokari", "no kari", "no curry", "Nogari","no-curry", "no curry please" (and similar tokens) all mean "NO CURRY" (customer wants the item without curry/sauce).
       - Business rule (apply always): If the spoken segment contains an item followed by any of the NO-CURRY tokens (examples above), interpret that as the customer requesting the item **without curry**, NOT with curry.
       - Example: "Aloo Paratha nokkari", "Aloo Parotta no curry", "aloo paratha no kari" → Interpret as "Aloo Parotta (No Curry)".
       - Matching rule: Prefer a verbatim menu item named "<Item> No Curry" (case-insensitive) if it exists in the provided menu. If the menu does NOT contain a "<Item> No Curry" entry, match to the base menu item "<Item>" and mark the match with an attribute "(no_curry)" in the output or put it under Matched Items with a note in the unavailable column if your schema requires.
       - Do NOT treat occurrences of the plain word 'curry' alone as automatically adding curry — only treat the tokens listed above or when context indicates "no curry" request. When the transcript has the item followed by the word "curry" but the raw transcription actually used a variant of 'no curry' (e.g., nokkari), interpret as NO CURRY.
       - "two tea", "to tea", "2t" ,"t" → Always match to "Tea" with correct quantity.
       - "paruppu adai" → Do NOT map to "Sambar Vada". Only match to "Paruppu Adai" if it exists in the menu. If it does not exist, classify under **Unavailable Items**.
       - Always interpret numeric words ("one", "two", "three", etc.) as digits. 
       - Prevent collapsing similar sounding dishes unless explicitly spoken (e.g., "adai" ≠ "vadai").
    3F. When two menu items have overlapping names (e.g., "Curd" and "Curd Rice", "Tea" and "Teh C Ice"), always treat them as distinct items. 
        - If both are mentioned in the order (e.g., "curd and curd rice"), count both separately. 
        - Never merge or overwrite one because of partial name overlap. 
        - In such cases, "Curd" refers to the standalone dish (yogurt), while "Curd Rice" refers to the rice-based dish.
        - Always detect both if they are clearly mentioned.

4. Do not hallucinate or invent item names. If the item clearly refers to a valid food or drink but is not in the menu list, classify it under **Unavailable Items**.
5. Quantities can appear before or after the dish name, or immediately after it without a space (e.g., “2 dosa”, “dosa 2”, or “chicken 65 5”). Always use numeric digits (e.g., “2”).
6. If quantity is not specified, assume it is 1.
7. If the same dish is mentioned multiple times, sum the total quantity across the order, even if some mentions have no explicit number (assume quantity 1 if missing). Example: "medhu vada 1, medhu vada, medhu vada" → Quantity = 3
8. Ignore filler or polite words like “bhaiya”, “anna”, “boss”, “please”, etc.
9. Match spoken items to the closest valid menu item using contextual understanding, only if it is safe and unambiguous to do so.
Examples:
- Spoken Order: give me 2 nan → Matched Menu Item: Butter Naan | Quantity: 2
- Spoken Order: garlic non → Matched Menu Item: Garlic Naan set | Quantity: 1
- Spoken Order: mee goreng  → Matched Menu Item: Mee Goreng 
- Spoken Order: dry ginger coffee / ginger coffee  → Matched Menu Item: ginger tea
- Spoken Order: patham keel / badam keer / badam kheer / Badam Kheer  → Matched Menu Item: Badam gheer
- Spoken Order: kodu / card / curd → Matched Menu Item: curd
- Spoken Order: curd and curd rice → Matched Menu Items:
      Curd | Quantity: 1
      Curd Rice | Quantity: 1
10. If the customer repeats or corrects an item mid-sentence, use only the last and most complete version.
11. Always prioritize exact item name matches from the menu; only fall back to relevant or similar matches if no exact match exists.
12. Output must be in English only. Return item names **exactly as they appear in the menu list**.
13. Do not return dish names in Hindi, Tamil, or any other script — only use the English names from the menu.
14. Do not generate or fabricate item_ids. Only use item_ids from the provided menu list.
15. Do not output the full menu under any condition — especially if the audio is blank, noisy, or from a movie.
16. If you are unsure whether an item matches anything in the menu, err on the side of listing it under **Unavailable Items**. Do not force a match.
17. If no food items are identified, return empty lists for both Matched Items and Unavailable Items.
18. Be especially careful with short or common food words like “naan” which may be misheard as “non”, “none”, “nan”,"naan", etc. If such words appear in a food context, interpret them as “Naan” and match to the correct menu item.
19. Do NOT invent, assume, or generalize chicken dishes (like curry, fry, kebab, etc.) unless it’s clearly and fully spoken. If customer says "chicken 65", only match it to "Chicken 65", not similar items like "chicken curry" or "seekh kebab".
✅ Output Format:
Translated Transcript: {translated_text}
Matched Items: item_id | item_name | quantity, ...
Unavailable Items: item_name | quantity, ...
 
 STRICT VALIDATION RULES
-Only output menu items exactly as they appear in the provided menu list (menu_str).
-Before including any item under “Matched Items”, mentally check: “Does this name exist verbatim in the menu list?”
-If yes, include it under Matched Items with its item_id.
-If no, even if it sounds similar or is contextually related, place it under Unavailable Items.
-Never invent or infer new dishes (e.g., don’t make up “Chicken Fried Rice Special” if only “Chicken Fried Rice” exists).
-Do not output any menu items that are not in the provided menu_str.
-The final output must contain only items present in the menu list (case-insensitive match allowed, but text must match exactly once normalized).
-If in doubt or the name is incomplete, classify as Unavailable instead of guessing.

✅ Example:
Translated Transcript: I would like one Veg Zinger Burger and two Sprite
Matched Items: 101 | Veg Zinger Burger | 1
Unavailable Items: Sprite | 2
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        output = response['choices'][0]['message']['content'].strip()

        # Parse the translated text and matched items from the response
        translated_text = ""
        matched_items = ""
        unavailable_items = ""
        for line in output.splitlines():
            if line.lower().startswith("translated transcript:"):
                translated_text = line.split(":", 1)[1].strip()
            elif line.lower().startswith("matched items:"):
                matched_items = line.split(":", 1)[1].strip()
            elif line.lower().startswith("unavailable items:"):
                unavailable_items = line.split(":", 1)[1].strip()

        return translated_text, matched_items, unavailable_items

    except Exception as e:
        return "", "", ""

def parse_order(text, menu_df,live_transcript=""):
    translated_text, response, unavailable_items = gpt_menu_match(text, menu_df,live_transcript)
    print("GPT RESPONSE:", response)
    if not response:
        return [], "", [], 0.0, translated_text, unavailable_items

    df = menu_df.copy()
    df.columns = [col.lower().strip() for col in df.columns]

    id_col = next((col for col in df.columns if col in ["id", "item id", "item_id"]), None)
    name_col = next((col for col in df.columns if "item" in col and "name" in col), None)
    price_col = next((col for col in df.columns if "price" in col), None)

    if not (id_col and name_col and price_col):
        raise ValueError("Required columns (Id, Item_Name, Price) not found in the menu.")

    # Build a lookup using clean item names
    menu_lookup = {}
    for _, row in df.iterrows():
        full_name = row[name_col].strip()
        lower_name = full_name.lower()
        parts = re.split(r"/|\(|\)", lower_name)
        variants = [lower_name] + [p.strip() for p in parts if p.strip()]
        for variant in variants:
            cleaned_key = variant.translate(str.maketrans('', '', string.punctuation)).strip()
            if cleaned_key not in menu_lookup:
                menu_lookup[cleaned_key] = {
                    "Id": int(row[id_col]),
                    "Rate": int(row[price_col]),
                    "Name": full_name
                }

    order = defaultdict(lambda: {"Id": "", "Quantity": 0, "Rate": 0, "Total": 0})
    unmatched_items = []
    entries = [e.strip() for e in response.split(",") if e.strip()]
    total_transcribed = len(entries)
    total_matched = 0

    for entry in entries:
        parts = [p.strip() for p in entry.split("|")]
        if len(parts) == 3:
            try:
                item_id = int(parts[0])
                item_name = parts[1]
                quantity = int(parts[2])
            except ValueError:
                print(f"⚠️ Skipping entry due to value error: {entry}")
                continue

            # Clean item name for lookup
            lookup_key = item_name.lower().translate(str.maketrans('', '', string.punctuation)).strip()
            menu_item = None

            if lookup_key in menu_lookup:
                menu_item = menu_lookup[lookup_key]
            else:
                try:
                    match_row = df[df[id_col] == item_id]
                    if not match_row.empty:
                        row = match_row.iloc[0]
                        menu_item = {
                            "Id": int(row[id_col]),
                            "Name": row[name_col]
                        }
                        print(f"✅ Fallback matched using ID: {item_id} → {row[name_col]}")
                    else:
                        print(f"❌ No row found for ID fallback: {item_id}")
                except Exception as e:
                    print(f"❌ ID fallback error: {e}")

            if menu_item:
                order[menu_item["Name"]]["Id"] = menu_item["Id"]
                order[menu_item["Name"]]["Quantity"] += quantity
                total_matched += 1
            else:
                unmatched_items.append(item_name)
                print(f"❌ No match for: {item_name}")
        else:
            unmatched_items.append(entry.strip())
            print(f"❌ Bad format or missing '|': {entry.strip()}")

    parsed = [
        {
            "Item": k,
            "Id": int(v["Id"]),
            "Quantity": int(v["Quantity"]),
        }
        for k, v in order.items()
    ]

    match_accuracy = round((total_matched / total_transcribed) * 100, 2) if total_transcribed > 0 else 0.0
    print(f"🔎 Match accuracy: {match_accuracy}% ({total_matched}/{total_transcribed})")
    print("Parsed order:", parsed)

    return parsed, response, unmatched_items, match_accuracy, translated_text, unavailable_items

from fastapi import UploadFile, File, Form, Request
from pathlib import Path
import os, tempfile, shutil
import json
from datetime import datetime
from uuid import uuid4
import soundfile as sf
SESSION_STATE = {}

@app.post("/process/")
async def process_audio_file(file: UploadFile = File(...), session_id: str = Form(...),live_transcript: str = Form(default="")):
    print("File received:", file.filename, "Session ID:", session_id)
    print("Live transcript received:", live_transcript)

    try:
        ext = Path(file.filename).suffix.lower()
        AUDIO_EXTENSIONS = {".wav", ".mp3", ".webm", ".ogg", ".m4a", ".mp4", ".flac", ".aac", ".aiff"}
        VIDEO_EXTENSIONS = {".mp4", ".webm", ".m4v", ".avi"}

        session_dir = os.path.join("audio", session_id)
        os.makedirs(session_dir, exist_ok=True)

        # Save original upload
        base_name = uuid4().hex
        raw_temp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        raw_temp.write(await file.read())
        raw_temp.close()

        if ext in VIDEO_EXTENSIONS:
            audio_path = raw_temp.name + "_audio.mp3"
            extract_audio_from_video(raw_temp.name, audio_path)
        elif ext in AUDIO_EXTENSIONS:
            audio_path = raw_temp.name
        else:
            os.unlink(raw_temp.name)
            return {"error": f"Unsupported file extension: {ext}"}

        # Convert to WAV
        converted_path = audio_path + "_converted.wav"
        convert_audio_to_wav(audio_path, converted_path)

        # final_original_path = os.path.join(session_dir, f"{base_name}_original.wav")
        # shutil.copyfile(converted_path, final_original_path)
# Save the original uploaded audio (converted to WAV) to session folder
        existing_files = [f for f in os.listdir(session_dir) if f.endswith("_original.wav")]
        next_index = len(existing_files) + 1
        final_original_path = os.path.join(session_dir, f"{session_id}_{next_index}_original.wav")
        shutil.copyfile(converted_path, final_original_path)

        # Denoise
        data, samplerate = sf.read(converted_path)
        if data.ndim > 1:
            data = data[:, 0]
        denoised = denoise_audio(data, samplerate)

        final_denoised_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        sf.write(final_denoised_path, denoised, samplerate)

        for path in [raw_temp.name, audio_path, converted_path]:
            if os.path.exists(path):
                os.remove(path)

        # Transcribe
        transcription = transcribe_audio(final_denoised_path)
        if menu_df.empty:
            return {"error": "No menu available."}

        if session_id not in SESSION_STATE:
            SESSION_STATE[session_id] = {
                "responses": [],
                "transcriptions": [],
                "translations": [],
                "parsed_orderes": [],
                "unavailable_items": []  # ✅ NEW
            }

        parsed_order, matched_items, unmatched_items, match_accuracy, translated_text, unavailable_items = parse_order(transcription, menu_df,live_transcript=live_transcript)

        SESSION_STATE[session_id]["transcriptions"].append(transcription)
        SESSION_STATE[session_id]["responses"].append(matched_items)
        SESSION_STATE[session_id]["translations"].append(translated_text)
        SESSION_STATE[session_id]["parsed_orderes"].append(parsed_order)
        SESSION_STATE[session_id]["unavailable_items"].append(unavailable_items)  # ✅ NEW

        result = {
            "session_id": session_id,
            "transcription": transcription,
            "translated_transcript": translated_text,
            "gpt_response": matched_items,
            "order": parsed_order,
            "unavailable": unavailable_items,  # ✅ NEW
            "accuracy": match_accuracy
        }

        print("🔍 Response returned:", json.dumps(result, indent=2, ensure_ascii=False))
        return result

    except Exception as e:
        return {"error": f"Unexpected error: {e}"}
    
@app.post("/feedback/")
async def receive_feedback(request: Request):
    data = await request.json()
    session_id = data.get("session_id")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    audio_files = []
    if session_id:
        session_dir = os.path.join("audio", session_id)
        if os.path.exists(session_dir):
            audio_files = os.listdir(session_dir)

    feedback_log = {
        "timestamp": timestamp,
        "session_id": session_id,
        "feedback": data.get("feedback", ""),
        "issues": data.get("issues", []),
        "transcriptions": SESSION_STATE.get(session_id, {}).get("transcriptions", []),
        "translated_transcripts": SESSION_STATE.get(session_id, {}).get("translations", []),
        "gpt_response": SESSION_STATE.get(session_id, {}).get("responses", []),
        "final_order": SESSION_STATE.get(session_id, {}).get("parsed_orderes", []),
        "unavailable_items": SESSION_STATE.get(session_id, {}).get("unavailable_items", []),  # ✅ NEW
        "audio_files": audio_files
    }

    print("\nReceived Feedback:")
    print(json.dumps(feedback_log, indent=4))

    with open("session_feedback.log", "a") as f:
        f.write(json.dumps(feedback_log) + "\n")

    return {"message": "Feedback received"}

# Instructions:
# - If the input is in a non-English language (e.g., Hindi, Tamil, Telugu, Urdu, malay), translate it to English before processing.
# - Customers may speak in different languages or accents — including English, Hindi, Tamil, and Malay. Understand their pronunciation and return the   best-matching menu item.
# - Quantities can appear before or after the dish name (e.g., "2 Pav Bhaji", "Pav Bhaji 2") — both are valid.
# - Ignore filler or address words like "bhaiya", "anna", "dada", "boss", or similar – they are not part of the order.
# - Identify food items and map them to the closest valid menu item, even if the names are partially spoken or mispronounced.
#   Examples: 
#   "Spoken Order: hot crispy chicken" → Original Menu item: "Hot & Crispy Chicken Bucket (6 pcs)", 
#   "Spoken Order: paneer popper" → "Original Menu item: Paneer Poppers", 
#   "Spoken Order: ginger burger" → "Original Menu item: Veg Zinger Burger", 
#   "Spoken Order: grilled zinger" → "Original Menu item: Tandoori Zinger Burger"
# - Use numeric digits for quantities (e.g., "two" → 2). If quantity is not stated, assume 1.
# - If a quantity is mentioned **after** a dish name (e.g., "butter chicken 2"), associate it correctly.
# - If no quantity is mentioned, assume it is 1.
# - Do not return "NaN" or leave blanks — always return numeric values.
# - If the same item appears multiple times, combine them and total the quantity.
# - If the user stutters or changes their mind mid-sentence (e.g., says "aaannnwwww one dosa... annnn one onion dosa"), discard incomplete or earlier mentions and keep only the final, most specific version ("1 Onion Dosa"). Ignore false starts, repetitions, or abandoned phrases.
# - Only include items that exist in the menu. Do not hallucinate and invent or guess items not found.
# - Output should be strictly in english with a single clean, comma-separated list (no bullet points or extra text), e.g.:
#   1 Veg Zinger Burger, 1 Hot & Crispy Chicken Bucket (6 pcs), 1 Pepsi 500 ml.
# - Format: "1 Item A, 2 Item B"






