def confidence_label(score: float) -> str:
    if score >= 0.75:
        return "high"
    elif score >= 0.5:
        return "moderate"
    else:
        return "low"


DIAG_EXPLANATIONS = {
    "FleasTicks": {
        "title": "Fleas or Tick Infestation",
        "description": "External parasites can cause itching, skin irritation, and discomfort."
    },
    "Worms": {
        "title": "Intestinal Worms",
        "description": "Worm infections may lead to weight loss, diarrhea, and reduced appetite."
    },
    "MildRespiratory": {
        "title": "Mild Respiratory Infection",
        "description": "Respiratory issues can cause coughing, sneezing, or nasal discharge."
    },
    "AllergicDermatitis": {
        "title": "Allergic Skin Condition",
        "description": "Allergies may cause itching, redness, hair loss, or skin inflammation."
    },
    "Gastroenteritis": {
        "title": "Gastrointestinal Upset (Gastroenteritis)",
        "description": "This condition commonly causes vomiting, diarrhea, lethargy, and appetite loss."
    }
}


def format_chat_response(predicted_class: str, confidence_map: dict) -> str:
    confidence = confidence_map[predicted_class]
    label = confidence_label(confidence)

    diag = DIAG_EXPLANATIONS[predicted_class]

    message = (
        f"Based on the symptoms you described, your pet may be experiencing "
        f"**{diag['title']}**.\n\n"
        f"{diag['description']}\n\n"
        f"This assessment has a **{label} level of confidence ({confidence:.0%})**."
    )

    if confidence >= 0.75:
        message += (
            "\n\nIf symptoms persist or worsen, it’s recommended to consult a veterinarian "
            "for a proper examination and treatment."
        )
    else:
        message += (
            "\n\nThis is an initial assessment only. Monitoring your pet and consulting "
            "a veterinarian is advised if you have concerns."
        )

    return message