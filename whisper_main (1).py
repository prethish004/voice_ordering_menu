# from fastapi import FastAPI, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# import tempfile
# import numpy as np
# import soundfile as sf
# import noisereduce as nr
# import openai
# from dotenv import load_dotenv
# import os
# import pandas as pd
# import re
# import torch
# import whisper
# import subprocess
# from collections import defaultdict
# from io import StringIO
# from pathlib import Path
# import string
# # Load environment variables
# load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")

# app = FastAPI()

# # CORS Middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load default menu
# menu_df = pd.DataFrame()
# default_menu_path = "menu.csv"
# if os.path.exists(default_menu_path):
#     menu_df = pd.read_csv(default_menu_path)
#     print(f"✅ Loaded default menu: {default_menu_path}")
# else:
#     print("⚠️ Default menu.csv not found. Please upload one via /upload-csv/")

# @app.post("/upload-csv/")
# async def upload_csv(csvFile: UploadFile = File(...)):
#     try:
#         contents = await csvFile.read()
#         decoded_csv = contents.decode("utf-8")

#         # Save to disk
#         with open("menu.csv", "w", encoding="utf-8") as f:
#             f.write(decoded_csv)

#         # Load to memory
#         global menu_df
#         menu_df = pd.read_csv(StringIO(decoded_csv))

#         return {
#             "status": "success",
#             "message": f"CSV file '{csvFile.filename}' uploaded successfully.",
#             "rows": len(menu_df),
#             "columns": list(menu_df.columns)
#         }

#     except Exception as e:
#         return {"status": "error", "message": str(e)}

# SUPPORTED_AUDIO_TYPES = {
#     "audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3", "audio/webm",
#     "audio/ogg", "audio/x-m4a", "audio/mp4", "audio/flac", "audio/aac",
#     "audio/x-aiff"
# }

# SUPPORTED_VIDEO_TYPES = {
#     "video/mp4", "video/webm", "video/x-m4v", "video/x-msvideo"
# }

# def denoise_audio(data, sr=16000):
#     try:
#         noise_sample = data[:sr] if len(data) >= sr else data
#         if np.all(noise_sample == 0):
#             return data
#         y_nr = nr.reduce_noise(y=data, sr=sr, y_noise=noise_sample)
#         return y_nr
#     except Exception:
#         return data

# def convert_audio_to_wav(input_path, output_path):
#     cmd = ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", "-f", "wav", output_path]
#     subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

# def extract_audio_from_video(input_path, output_path):
#     cmd = ["ffmpeg", "-y", "-i", input_path, "-vn", "-acodec", "mp3", "-ar", "16000", "-ac", "1", output_path]
#     subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = whisper.load_model("large-v3-turbo").to(device)

# def transcribe_audio(path):
#     """Transcribe audio using Whisper Large v3."""
#     try:
#         result = model.transcribe(path)
#         return result["text"]
#     except Exception as e:
#         print(f"❌ Transcription failed: {e}")
#         return ""

# # def translate_text(transcribed_text):
# #     """Translate text into English if it's Tamil/Hindi/other supported language."""
# #     prompt = f"""
# # You are a translation assistant. Detect the input language (English, Tamil, or Hindi).
# # Translate the following customer transcript into **English (US)** without changing meaning:

# # Transcript:
# # \"\"\"{transcribed_text}\"\"\"

# # Output ONLY the translated English text, nothing else.
# # """
# #     try:
# #         response = openai.ChatCompletion.create(
# #             model="gpt-4o",
# #             messages=[{"role": "user", "content": prompt}],
# #             temperature=0
# #         )
# #         return response['choices'][0]['message']['content'].strip()
# #     except Exception as e:
# #         print(f"❌ Translation failed: {e}")
# #         return transcribed_text   # fallback: return original if translation fails
# def translate_text(transcribed_text):
#     """Translate and normalize transcript into clean English menu-style items."""
#     prompt = f"""
# You are a multilingual food menu translation assistant. Read the transcript and return a single, exact output: a comma-separated list of menu items in clean English with numeric quantities as digits. NOTHING else.

# CRITICAL RULES (follow exactly):
# 1. Detect and translate mixed Tamil/Hindi/English/romanized text to clean English menu items.
# 2. Preserve standard dish names (dosa, pongal, sambar rice, curd rice, idli, biryani/briyani -> briyani, rava, naan, uthappam, parotta, chapati/chapatti -> chapati, noodles, etc.). Correct obvious common English spellings but do NOT invent new names.

# Transcript:
# \"\"\"{transcribed_text}\"\"\"

# Output ONLY the cleaned menu list in English, with quantities normalized, nothing else.
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
#         return transcribed_text   # fallback
   

# correction_map = {
#     "teo": "teh o",
#     "teho": "teh o",
#     "tehc": "teh c",
#     "teh-c": "teh c",
#     "tehc hot": "teh c hot",
#     "teh-c hot": "teh c hot",
#     "teho ice": "teh o ice",
#     "teo ice": "teh o ice",
#     "kopio": "kopi o",
#     "kopio ice": "kopi o ice",
# }
# def apply_corrections(text, correction_map):
#     for wrong, right in correction_map.items():
#         text = text.replace(wrong, right)
#     return text


# # def gpt_menu_match(transcribed_text, menu_df):
# def gpt_menu_match(transcribed_text, menu_df, live_transcript=""):
#     # Translate first
#     transcribed_text = apply_corrections(transcribed_text.lower(), correction_map)
#     translated_text = translate_text(transcribed_text)
#     # menu_str = ", ".join(menu_df["Item_Name"].astype(str).tolist())
#     # menu_itemid = ", ".join(menu_df["Item_id"].astype(str).tolist())
#     menu_str = ", ".join(f"{row['Item_id']} | {row['Item_Name']}" for _, row in menu_df.iterrows())
#     prompt = f"""
# You are an AI voice assistant for a restaurant in Singapore. Your job is to extract food items and their quantities from a spoken order transcript.
 
# Here are two versions of the order:
# 1. Raw Transcription (may contain Tamil/English mix): 
# \"\"\"{transcribed_text}\"\"\"
# 2. Machine Translated Transcript (may contain errors like 'rice lemonade' instead of 'Nasi Lemak'): 
# # \"\"\"{translated_text}\"\"\"
# 3. Live Transcript (browser SpeechRecognition, may contain noise): 
# \"\"\"{live_transcript}\"\"\"

# Menu items:
# {menu_str}
# Important: In this menu, "Mee" and "Mee Hoon" are **not the same**. Always treat "Mee Hoon" dishes as separate. Do not substitute "Mee Hoon" for "Mee" or vice versa.
# STRICT VALIDATION RULES
# -Only output menu items exactly as they appear in the provided menu list (menu_str).
# -Before including any item under “Matched Items”, mentally check: “Does this name exist verbatim in the menu list?”
# -If yes, include it under Matched Items with its item_id.
# -If no, even if it sounds similar or is contextually related, place it under Unavailable Items.
# -Never invent or infer new dishes (e.g., don’t make up “Chicken Fried Rice Special” if only “Chicken Fried Rice” exists).
# -Do not output any menu items that are not in the provided menu_str.
# -The final output must contain only items present in the menu list (case-insensitive match allowed, but text must match exactly once normalized).
# -If in doubt or the name is incomplete, classify as Unavailable instead of guessing.
 
# Instructions:
# 1. Always use the raw transcription for identifying food names. Use the live transcript as supporting evidence (for numbers or corrections). Use the translated transcript only to understand filler words, numbers, or context. If there is a conflict, prioritize the raw transcription for dish names and then match with the menu list.
# 2. If a generic or vague term is spoken (e.g., “vada”), select the closest match from the menu **only if it clearly maps to a known menu item**. If there is ambiguity or no close match, treat it as **Unavailable**.
# 3. If dish names are incomplete or mispronounced, match them to the most likely correct dish using contextual understanding **only if the dish exists in the menu** and the match is strong. Do **not** guess or stretch uncertain terms into menu items.
#    3A. If a dish name contains a prefix like "Mee Hoon" or "Mee", prioritize exact full-name matching. Do not collapse "Mee Hoon Goreng Mutton" into "Mee Goreng Mutton". These are distinct items and should not be confused.
#    3B. Be careful with menu items that differ only by short suffixes or prefixes (e.g., “Mee Hoon” vs “Mee”, “Teh O” vs “Teh C”, “Maggi” vs “Mee Goreng”). These are distinct dishes. Only match when the full phrase is clearly present. Do not collapse one into another based on partial match.
#    3C. Be careful with "Koli Soru / Koali Soaru / Kolisoaru" and similar pronunciations.Always match these ONLY to "Koali Soaru (Chicken Broth Rice)" from the menu, never to "Chicken Biryani" or other chicken rice dishes.
#    3D. Dishes like “Mee Goreng”, “Maggi Goreng”, “Kway Teow Goreng”, and “Mee Hoon Goreng” are different types of fried noodles. Do not substitute one for another even if the ingredients (like anchovies) are similar. Use exact phrase match when possible.
#    3E. Be careful with common mishearing issues:
#        - "2 boli" should always map to "Boli" with quantity 2 (never "body" or "toli").
#        - "two tea", "to tea", "2t" ,"t" → Always match to "Tea" with correct quantity.
#        - "paruppu adai" → Do NOT map to "Sambar Vada". Only match to "Paruppu Adai" if it exists in the menu. If it does not exist, classify under **Unavailable Items**.
#        - Always interpret numeric words ("one", "two", "three", etc.) as digits. 
#        - Prevent collapsing similar sounding dishes unless explicitly spoken (e.g., "adai" ≠ "vadai").
#    3F. When two menu items have overlapping names (e.g., "Curd" and "Curd Rice", "Tea" and "Teh C Ice"), always treat them as distinct items. 
#         - If both are mentioned in the order (e.g., "curd and curd rice"), count both separately. 
#         - Never merge or overwrite one because of partial name overlap. 
#         - In such cases, "Curd" refers to the standalone dish (yogurt), while "Curd Rice" refers to the rice-based dish.
#         - Always detect both if they are clearly mentioned.
# 4. Do not hallucinate or invent item names. If the item clearly refers to a valid food or drink but is not in the menu list, classify it under **Unavailable Items**.
# 5. Quantities can appear before or after the dish name, or immediately after it without a space (e.g., “2 dosa”, “dosa 2”, or “chicken 65 5”). Always use numeric digits (e.g., “2”).
# 6. If quantity is not specified, assume it is 1.
# 7. If the same dish is mentioned multiple times, sum the total quantity across the order, even if some mentions have no explicit number (assume quantity 1 if missing). Example: "medhu vada 1, medhu vada, medhu vada" → Quantity = 3
# 8. Ignore filler or polite words like “bhaiya”, “anna”, “boss”, “please”, etc.
# 9. Match spoken items to the closest valid menu item using contextual understanding, only if it is safe and unambiguous to do so.

# Examples:
# - Spoken Order: goreng rice → Matched Menu Item: Nasi Goreng
# - Spoken Order: egg prata → Matched Menu Item: Egg Paratha / Muttai Parotta
# - Spoken Order: sambhar wada → Matched Menu Item: Sambar Vada
# - Spoken Order: chicken 65 5 → Matched Menu Item: Chicken 65 | Quantity: 5
# - Spoken Order: give me 2 nan → Matched Menu Item: Butter Naan | Quantity: 2
# - Spoken Order: garlic non → Matched Menu Item: Garlic Naan | Quantity: 1
# - Spoken Order: chicken murtabak / chicken muttak / chick mutabak / chicken mutabak / chicken muttapak → Matched Menu Item: Chicken Murtabak
# - Spoken Order: koli soru / koali soaru / kolisoaru / chicken soaru / broth rice → Matched Menu Item: Koali Soaru (Chicken Broth Rice)
# - Spoken Order: mutton murtabak / muttabaka / mutton muttak / mutabak / mutton muttapak→ Matched Menu Item: Mutton Murtabak
# - Spoken Order: mee hoon goreng mutton → Matched Menu Item: Mee Hoon Goreng Mutton
# - Spoken Order: mee goreng mutton → Matched Menu Item: Mee Goreng Mutton
# - Spoken Order: teh o → Matched Menu Item: Teh O
# - Spoken Order: teh c → Matched Menu Item: Teh C
# - Spoken Order: tea halwa → Matched Menu Item: Teh Halia
# - Spoken Order: Desi hot → Matched Menu Item: Teh C hot
# - Spoken Order: brew → Matched Menu Item: Bru / Nescafe Coffee
# - Spoken Order: kopi o halia → Matched Menu Item: Kopi O Halia
# - Spoken Order: mee goreng with ikan bilis → Matched Menu Item: Mee Goreng Ikan Bilis
# - Spoken Order: race / rise / plain rice / rice  → Matched Menu Item: Rice(plain)
# - Spoken Order: kamatorise / Tomato / toma rice  → Matched Menu Item: Tomato Rice
# - Spoken Order: copee o / coffee o / copy o /copy aur/ Gobi o / copi o  → Matched Menu Item: Kopi O
# - Spoken Order: 7up / fanta / pokka  → Matched Menu Item: Pokka 500 ml
# - Spoken Order: bangali drink / bangala / paan drinks / pongal drink / Bottle drinks  → Matched Menu Item: Bangala Drink
# - Spoken Order: dry ginger coffee / ginger coffee  → Matched Menu Item: Sukku Coffee
# - Spoken Order: thoy / dhoy / doy  → Matched Menu Item: Doi
# - Spoken Order: SI அப்பலம் / applam / அப்பலம் / SI  → Matched Menu Item: S.I appalam

# 10. If the customer repeats or corrects an item mid-sentence, use only the last and most complete version.
# 11. Output must be in English only. Return item names **exactly as they appear in the menu list**.
# 12. Do not return dish names in Hindi, Tamil, or any other script — only use the English names from the menu.
# 13. Do not generate or fabricate item_ids. Only use item_ids from the provided menu list.
# 14. Do not output the full menu under any condition — especially if the audio is blank, noisy, or from a movie.
# 15. If you are unsure whether an item matches anything in the menu, err on the side of listing it under **Unavailable Items**. Do not force a match.
# 16. If no food items are identified, return empty lists for both Matched Items and Unavailable Items.
# 17. Be especially careful with short or common food words like “naan” which may be misheard as “non”, “none”, “nan”,"naan", etc. If such words appear in a food context, interpret them as “Naan” and match to the correct menu item.
# 18. Do NOT invent, assume, or generalize chicken dishes (like curry, fry, kebab, etc.) unless it’s clearly and fully spoken. If customer says "chicken 65", only match it to "Chicken 65", not similar items like "chicken curry" or "seekh kebab".
# ✅ Output Format:
# Translated Transcript: {translated_text}
# Matched Items: item_id | item_name | quantity, ...
# Unavailable Items: item_name | quantity, ...
 
# ✅ Example:
# Translated Transcript: I would like one Veg Zinger Burger and two Sprite
# Matched Items: 101 | Veg Zinger Burger | 1
# Unavailable Items: Sprite | 2
# """

#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-4o-mini",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0
#         )
#         output = response['choices'][0]['message']['content'].strip()

#         # Parse the translated text and matched items from the response
#         translated_text = ""
#         matched_items = ""
#         unavailable_items = ""
#         for line in output.splitlines():
#             if line.lower().startswith("translated transcript:"):
#                 translated_text = line.split(":", 1)[1].strip()
#             elif line.lower().startswith("matched items:"):
#                 matched_items = line.split(":", 1)[1].strip()
#             elif line.lower().startswith("unavailable items:"):
#                 unavailable_items = line.split(":", 1)[1].strip()

#         return translated_text, matched_items, unavailable_items

#     except Exception as e:
#         return "", "", ""

# def parse_order(text, menu_df,live_transcript=""):
#     translated_text, response, unavailable_items = gpt_menu_match(text, menu_df,live_transcript)
#     print("GPT RESPONSE:", response)
#     if not response:
#         return [], "", [], 0.0, translated_text, unavailable_items

#     df = menu_df.copy()
#     df.columns = [col.lower().strip() for col in df.columns]

#     id_col = next((col for col in df.columns if col in ["id", "item id", "item_id"]), None)
#     name_col = next((col for col in df.columns if "item" in col and "name" in col), None)
#     price_col = next((col for col in df.columns if "price" in col), None)

#     if not (id_col and name_col and price_col):
#         raise ValueError("Required columns (Id, Item_Name, Price) not found in the menu.")

#     # Build a lookup using clean item names
#     menu_lookup = {}
#     for _, row in df.iterrows():
#         full_name = row[name_col].strip()
#         lower_name = full_name.lower()
#         parts = re.split(r"/|\(|\)", lower_name)
#         variants = [lower_name] + [p.strip() for p in parts if p.strip()]
#         for variant in variants:
#             cleaned_key = variant.translate(str.maketrans('', '', string.punctuation)).strip()
#             if cleaned_key not in menu_lookup:
#                 menu_lookup[cleaned_key] = {
#                     "Id": int(row[id_col]),
#                     "Rate": int(row[price_col]),
#                     "Name": full_name
#                 }

#     order = defaultdict(lambda: {"Id": "", "Quantity": 0, "Rate": 0, "Total": 0})
#     unmatched_items = []
#     entries = [e.strip() for e in response.split(",") if e.strip()]
#     total_transcribed = len(entries)
#     total_matched = 0

#     for entry in entries:
#         parts = [p.strip() for p in entry.split("|")]
#         if len(parts) == 3:
#             try:
#                 item_id = int(parts[0])
#                 item_name = parts[1]
#                 quantity = int(parts[2])
#             except ValueError:
#                 print(f"⚠️ Skipping entry due to value error: {entry}")
#                 continue

#             # Clean item name for lookup
#             lookup_key = item_name.lower().translate(str.maketrans('', '', string.punctuation)).strip()
#             menu_item = None

#             if lookup_key in menu_lookup:
#                 menu_item = menu_lookup[lookup_key]
#             else:
#                 try:
#                     match_row = df[df[id_col] == item_id]
#                     if not match_row.empty:
#                         row = match_row.iloc[0]
#                         menu_item = {
#                             "Id": int(row[id_col]),
#                             "Name": row[name_col]
#                         }
#                         print(f"✅ Fallback matched using ID: {item_id} → {row[name_col]}")
#                     else:
#                         print(f"❌ No row found for ID fallback: {item_id}")
#                 except Exception as e:
#                     print(f"❌ ID fallback error: {e}")

#             if menu_item:
#                 order[menu_item["Name"]]["Id"] = menu_item["Id"]
#                 order[menu_item["Name"]]["Quantity"] += quantity
#                 total_matched += 1
#             else:
#                 unmatched_items.append(item_name)
#                 print(f"❌ No match for: {item_name}")
#         else:
#             unmatched_items.append(entry.strip())
#             print(f"❌ Bad format or missing '|': {entry.strip()}")

#     parsed = [
#         {
#             "Item": k,
#             "Id": int(v["Id"]),
#             "Quantity": int(v["Quantity"]),
#         }
#         for k, v in order.items()
#     ]

#     match_accuracy = round((total_matched / total_transcribed) * 100, 2) if total_transcribed > 0 else 0.0
#     print(f"🔎 Match accuracy: {match_accuracy}% ({total_matched}/{total_transcribed})")
#     print("Parsed order:", parsed)

#     return parsed, response, unmatched_items, match_accuracy, translated_text, unavailable_items

# from fastapi import UploadFile, File, Form, Request
# from pathlib import Path
# import os, tempfile, shutil
# import json
# from datetime import datetime
# from uuid import uuid4
# import soundfile as sf
# SESSION_STATE = {}

# @app.post("/process/")
# async def process_audio_file(file: UploadFile = File(...), session_id: str = Form(...),live_transcript: str = Form(default="")):
#     print("File received:", file.filename, "Session ID:", session_id)
#     print("Live transcript received:", live_transcript)

#     try:
#         ext = Path(file.filename).suffix.lower()
#         AUDIO_EXTENSIONS = {".wav", ".mp3", ".webm", ".ogg", ".m4a", ".mp4", ".flac", ".aac", ".aiff"}
#         VIDEO_EXTENSIONS = {".mp4", ".webm", ".m4v", ".avi"}

#         session_dir = os.path.join("audio", session_id)
#         os.makedirs(session_dir, exist_ok=True)

#         # Save original upload
#         base_name = uuid4().hex
#         raw_temp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
#         raw_temp.write(await file.read())
#         raw_temp.close()

#         if ext in VIDEO_EXTENSIONS:
#             audio_path = raw_temp.name + "_audio.mp3"
#             extract_audio_from_video(raw_temp.name, audio_path)
#         elif ext in AUDIO_EXTENSIONS:
#             audio_path = raw_temp.name
#         else:
#             os.unlink(raw_temp.name)
#             return {"error": f"Unsupported file extension: {ext}"}

#         # Convert to WAV
#         converted_path = audio_path + "_converted.wav"
#         convert_audio_to_wav(audio_path, converted_path)

#         # final_original_path = os.path.join(session_dir, f"{base_name}_original.wav")
#         # shutil.copyfile(converted_path, final_original_path)
# # Save the original uploaded audio (converted to WAV) to session folder
#         existing_files = [f for f in os.listdir(session_dir) if f.endswith("_original.wav")]
#         next_index = len(existing_files) + 1
#         final_original_path = os.path.join(session_dir, f"{session_id}_{next_index}_original.wav")
#         shutil.copyfile(converted_path, final_original_path)

#         # Denoise
#         data, samplerate = sf.read(converted_path)
#         if data.ndim > 1:
#             data = data[:, 0]
#         denoised = denoise_audio(data, samplerate)

#         final_denoised_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
#         sf.write(final_denoised_path, denoised, samplerate)

#         for path in [raw_temp.name, audio_path, converted_path]:
#             if os.path.exists(path):
#                 os.remove(path)

#         # Transcribe
#         transcription = transcribe_audio(final_denoised_path)
#         if menu_df.empty:
#             return {"error": "No menu available."}

#         if session_id not in SESSION_STATE:
#             SESSION_STATE[session_id] = {
#                 "responses": [],
#                 "transcriptions": [],
#                 "translations": [],
#                 "parsed_orderes": [],
#                 "unavailable_items": []  # ✅ NEW
#             }

#         parsed_order, matched_items, unmatched_items, match_accuracy, translated_text, unavailable_items = parse_order(transcription, menu_df,live_transcript=live_transcript)

#         SESSION_STATE[session_id]["transcriptions"].append(transcription)
#         SESSION_STATE[session_id]["responses"].append(matched_items)
#         SESSION_STATE[session_id]["translations"].append(translated_text)
#         SESSION_STATE[session_id]["parsed_orderes"].append(parsed_order)
#         SESSION_STATE[session_id]["unavailable_items"].append(unavailable_items)  # ✅ NEW

#         result = {
#             "session_id": session_id,
#             "transcription": transcription,
#             "translated_transcript": translated_text,
#             "gpt_response": matched_items,
#             "order": parsed_order,
#             "unavailable": unavailable_items,  # ✅ NEW
#             "accuracy": match_accuracy
#         }

#         print("🔍 Response returned:", json.dumps(result, indent=2, ensure_ascii=False))
#         return result

#     except Exception as e:
#         return {"error": f"Unexpected error: {e}"}
    
# @app.post("/feedback/")
# async def receive_feedback(request: Request):
#     data = await request.json()
#     session_id = data.get("session_id")
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#     audio_files = []
#     if session_id:
#         session_dir = os.path.join("audio", session_id)
#         if os.path.exists(session_dir):
#             audio_files = os.listdir(session_dir)

#     feedback_log = {
#         "timestamp": timestamp,
#         "session_id": session_id,
#         "feedback": data.get("feedback", ""),
#         "issues": data.get("issues", []),
#         "transcriptions": SESSION_STATE.get(session_id, {}).get("transcriptions", []),
#         "translated_transcripts": SESSION_STATE.get(session_id, {}).get("translations", []),
#         "gpt_response": SESSION_STATE.get(session_id, {}).get("responses", []),
#         "final_order": SESSION_STATE.get(session_id, {}).get("parsed_orderes", []),
#         "unavailable_items": SESSION_STATE.get(session_id, {}).get("unavailable_items", []),  # ✅ NEW
#         "audio_files": audio_files
#     }

#     print("\nReceived Feedback:")
#     print(json.dumps(feedback_log, indent=4))

#     with open("session_feedback.log", "a") as f:
#         f.write(json.dumps(feedback_log) + "\n")

#     return {"message": "Feedback received"}
# # Instructions:
# # - If the input is in a non-English language (e.g., Hindi, Tamil, Telugu, Urdu, malay), translate it to English before processing.
# # - Customers may speak in different languages or accents — including English, Hindi, Tamil, and Malay. Understand their pronunciation and return the   best-matching menu item.
# # - Quantities can appear before or after the dish name (e.g., "2 Pav Bhaji", "Pav Bhaji 2") — both are valid.
# # - Ignore filler or address words like "bhaiya", "anna", "dada", "boss", or similar – they are not part of the order.
# # - Identify food items and map them to the closest valid menu item, even if the names are partially spoken or mispronounced.
# #   Examples: 
# #   "Spoken Order: hot crispy chicken" → Original Menu item: "Hot & Crispy Chicken Bucket (6 pcs)", 
# #   "Spoken Order: paneer popper" → "Original Menu item: Paneer Poppers", 
# #   "Spoken Order: ginger burger" → "Original Menu item: Veg Zinger Burger", 
# #   "Spoken Order: grilled zinger" → "Original Menu item: Tandoori Zinger Burger"
# # - Use numeric digits for quantities (e.g., "two" → 2). If quantity is not stated, assume 1.
# # - If a quantity is mentioned **after** a dish name (e.g., "butter chicken 2"), associate it correctly.
# # - If no quantity is mentioned, assume it is 1.
# # - Do not return "NaN" or leave blanks — always return numeric values.
# # - If the same item appears multiple times, combine them and total the quantity.
# # - If the user stutters or changes their mind mid-sentence (e.g., says "aaannnwwww one dosa... annnn one onion dosa"), discard incomplete or earlier mentions and keep only the final, most specific version ("1 Onion Dosa"). Ignore false starts, repetitions, or abandoned phrases.
# # - Only include items that exist in the menu. Do not hallucinate and invent or guess items not found.
# # - Output should be strictly in english with a single clean, comma-separated list (no bullet points or extra text), e.g.:
# #   1 Veg Zinger Burger, 1 Hot & Crispy Chicken Bucket (6 pcs), 1 Pepsi 500 ml.
# # - Format: "1 Item A, 2 Item B"

import time
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
import torch
import whisper
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
        with open("menu.csv", "w", encoding="utf-8") as f:
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

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device selected for Whisper: {device}")
# Load Whisper model on the selected device. Keep as large-v3-turbo if you need that model;
# here we keep the original model name but move it to device and will enable fp16 at transcribe call.
model_load_start = time.time()
try:
    model = whisper.load_model("large-v3-turbo", device=device)
    model_load_time = time.time() - model_load_start
    print(f"Model loaded in {model_load_time:.3f}s (model: large-v3-turbo, device={device})")
except Exception as e:
    model = None
    print(f"❌ Failed to load Whisper model: {e}")

def transcribe_audio(path):
    """Transcribe audio using Whisper Large v3 with fp16 on GPU when available.
    Prints the time taken by Whisper for each transcription as: Whisper time: X.XXXs
    """
    if model is None:
        print("❌ Whisper model not loaded. Skipping transcription.")
        return ""

    t0 = time.time()
    try:
        # use fp16 if running on cuda
        use_fp16 = True if device == "cuda" else False
        # model.transcribe supports fp16 flag
        result = model.transcribe(path, fp16=use_fp16)
        whisper_time = time.time() - t0
        print(f"Whisper time: {whisper_time:.3f}s (fp16={use_fp16})")
        return result.get("text", "")
    except Exception as e:
        whisper_time = time.time() - t0
        print(f"❌ Transcription failed after {whisper_time:.3f}s: {e}")
        return ""


def translate_text(transcribed_text):
    """Translate and normalize a mixed-language food order into a clean English menu-style list."""
    prompt = f"""
You are a multilingual FOOD ORDER TRANSLATOR and normalizer.
Your task is to read a spoken or written food order in Tamil, Hindi, or English,
and output a **clean, normalized English menu list** suitable for digital ordering.

STRICT INSTRUCTIONS:
1. Translate the entire text into English (US), keeping dish names as they are if they are already known Indian dishes.
   - Tamil: "சாம்பார் சாதம்" → "Sambar Rice"
   - Hindi: "aloo paratha" → "Aloo Paratha"
   - Keep known dish words like dosa, idli, biryani, chutney, etc. unchanged.

2. Convert ALL numbers or quantity words to digits.
   - Tamil: "ஒன்று"=1, "இரண்டு"=2, "மூன்று"=3, "நான்கு"=4, "ஐந்து"=5, "ஆறு"=6, "ஏழு"=7, "எட்டு"=8, "ஒன்பது"=9, "பத்து"=10
   - Hindi: "ek"=1, "do"=2, "teen"=3, "char"=4, "paanch"=5, "chhe"=6, "saat"=7, "aath"=8, "nau"=9, "das"=10
   - English words: "one"=1, "two"=2, etc.

3. Handle **both number-before-item and item-before-number cases**.
   - Example: "2 dosa", "dosa 2", "two dosa", "dosa two" → all become "2 Dosa"
   - Example: "idli three", "three idli" → "3 Idli"

4. Preserve important modifiers such as:
   - "set", "combo", "with", "plate", "parcel", "meals"
   - Example: "aloo parotta set 2" → "2 Aloo Parotta Set"
   - Example: "idiyappam with curry 3" → "3 Idiyappam with Curry"

5. Separate items using commas, in the order they were mentioned.

6. Output format:
   👉 `<quantity> <item name>`, comma-separated.
   Example: `2 Dosa, 1 Sambar Rice, 3 Idli, 2 Chutney, 1 Curd Rice`

7. Never skip, merge, or reorder items.
8. Do NOT include explanations, notes, or any extra text — only return the final, clean translated line.

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
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"❌ Translation failed: {e}")
        return transcribed_text  # fallback

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
# \"\"\"{translated_text}\"\"\"
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
       - "nokkari", "nokari", "no kari", "no curry", "no-curry", "no curry please" (and similar tokens) all mean "NO CURRY" (customer wants the item without curry/sauce).
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
- Spoken Order: soup / of day / soup of the day  → Matched Menu Item: soup of the day
- Spoken Order: Chilli Bhutta / chilli parata / chili parota → Matched Menu Item: chilli prata	
- Spoken Order: curd and curd rice → Matched Menu Items:
      Curd | Quantity: 1
      Curd Rice | Quantity: 1
- Spoken Order: SI அப்பலம் / applam / அப்பலம் / SI  → Matched Menu Item: S.I appalam
- Spoken Order: Aloo Paratha nokkari / Aloo Paratha with Curry  → Matched Menu Item: aloo prata - no curry
- Spoken Order: Idiyappam → Matched Menu Item: Idiyappam set

10. If the customer repeats or corrects an item mid-sentence, use only the last and most complete version.
11. Output must be in English only. Return item names **exactly as they appear in the menu list**.
12. Do not return dish names in Hindi, Tamil, or any other script — only use the English names from the menu.
13. Do not generate or fabricate item_ids. Only use item_ids from the provided menu list.
14. Do not output the full menu under any condition — especially if the audio is blank, noisy, or from a movie.
15. If you are unsure whether an item matches anything in the menu, err on the side of listing it under **Unavailable Items**. Do not force a match.
16. If no food items are identified, return empty lists for both Matched Items and Unavailable Items.
17. Be especially careful with short or common food words like “naan” which may be misheard as “non”, “none”, “nan”,"naan", etc. If such words appear in a food context, interpret them as “Naan” and match to the correct menu item.
18. Do NOT invent, assume, or generalize chicken dishes (like curry, fry, kebab, etc.) unless it’s clearly and fully spoken. If customer says "chicken 65", only match it to "Chicken 65", not similar items like "chicken curry" or "seekh kebab".
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
        t1 = time.time()
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        llm_time = time.time() - t1
        print(f"LLM time (gpt_menu_match): {llm_time:.3f}s")
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

