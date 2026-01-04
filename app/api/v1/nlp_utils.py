import numpy as np
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# Columns
# -----------------------------
static_cols = ["species","breed","sex","neutered"]

time_cols = [
    "age_years","weight_kg",
    "vomiting","diarrhea","coughing","sneezing","itching","hair_loss","red_skin",
    "ear_scratching","ear_discharge","fleas_seen","ticks_seen","worms_seen",
    "lethargy","loss_appetite","gagging_hairballs","constipation","bloody_stool",
    "wound_present","hot_spot","diet_change_recent","new_food","outdoor_activity_high",
    "drank_unclean_water","grooming_recent","contact_with_other_animals",
    "last_vet_visit_days_ago"
]

season_cols = ["season_rainy", "season_summer", "season_winter"]

# -----------------------------
# Symptom keywords
# -----------------------------
SYMPTOM_KEYWORDS = {
    "vomiting": ["vomit", "vomiting", "threw up"],
    "diarrhea": ["diarrhea", "loose stool", "watery stool"],
    "coughing": ["cough"],
    "sneezing": ["sneeze"],
    "itching": ["itch", "scratching"],
    "hair_loss": ["hair loss", "losing hair"],
    "red_skin": ["red skin", "rash", "inflamed skin"],
    "ear_scratching": ["scratching ears"],
    "ear_discharge": ["ear discharge"],
    "fleas_seen": ["fleas"],
    "ticks_seen": ["ticks"],
    "worms_seen": ["worms"],
    "lethargy": ["lethargic", "tired", "weak"],
    "loss_appetite": ["not eating", "loss of appetite"],
    "gagging_hairballs": ["gagging", "hairball"],
    "constipation": ["constipation"],
    "bloody_stool": ["blood in stool", "bloody stool"],
    "wound_present": ["wound", "injury"],
    "hot_spot": ["hot spot"],
    "diet_change_recent": ["diet change"],
    "new_food": ["new food"],
    "outdoor_activity_high": ["outdoor", "outside a lot"],
    "drank_unclean_water": ["dirty water", "unclean water"],
    "grooming_recent": ["groomed", "grooming"],
    "contact_with_other_animals": ["other animals", "stray animals"]
}

# -----------------------------
# Extraction helpers
# -----------------------------
def extract_last_vet_days(text):
    match = re.search(r'(\d+)\s*(day|days)', text)
    if match:
        return int(match.group(1))
    if "month" in text:
        return 30
    return 0

def extract_time_features_from_text(text):
    text = text.lower()
    features = {}
    for col in time_cols:
        features[col] = 0
    for symptom, keywords in SYMPTOM_KEYWORDS.items():
        features[symptom] = int(any(k in text for k in keywords))
    features["last_vet_visit_days_ago"] = extract_last_vet_days(text)
    return features

# -----------------------------
# Build model input
# -----------------------------
def build_model_input_from_text(user_text, pet_static_info, season, seq_len):
    static_vals = [pet_static_info[c] for c in static_cols]
    time_feats = extract_time_features_from_text(user_text)
    time_feats["age_years"] = pet_static_info["age_years"]
    time_feats["weight_kg"] = pet_static_info["weight_kg"]
    time_vals = [time_feats[c] for c in time_cols]
    season_vals = [
        int(season == "rainy"),
        int(season == "summer"),
        int(season == "winter")
    ]
    visit_vector = static_vals + time_vals + season_vals
    X = np.array([visit_vector], dtype="float32")
    X_padded = pad_sequences([X], maxlen=seq_len, padding="post", dtype="float32", value=0.0)
    return X_padded
