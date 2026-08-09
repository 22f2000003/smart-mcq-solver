
import os
import re
import html
import tempfile

import numpy as np
import streamlit as st
import torch

from transformers import AutoTokenizer, AutoModelForMultipleChoice
from huggingface_hub import hf_hub_download


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

MODEL_REPO = "zarrinnehal/smart-mcq-solver-models"

DEBERTA_BASE = "microsoft/deberta-v3-small"
ROBERTA_BASE = "roberta-base"

N_FOLDS = 5
MAX_LEN = 256

OPTIONS = ["A", "B", "C", "D", "E"]


# --------------------------------------------------
# TEXT CLEANING
# Same cleaning used in the Kaggle notebook
# --------------------------------------------------

def clean_text(text):

    if not isinstance(text, str):
        return ""

    text = html.unescape(text)
    text = text.replace('""', '"')
    text = re.sub(r"\s+", " ", text).strip()

    return text


# --------------------------------------------------
# LOAD TOKENIZERS
# --------------------------------------------------

@st.cache_resource
def load_tokenizers():

    deberta_tokenizer = AutoTokenizer.from_pretrained(
        DEBERTA_BASE
    )

    roberta_tokenizer = AutoTokenizer.from_pretrained(
        ROBERTA_BASE
    )

    return deberta_tokenizer, roberta_tokenizer

deberta_tokenizer, roberta_tokenizer = load_tokenizers()


# --------------------------------------------------
# TOKENIZE ONE MCQ
# Same structure as Kaggle notebook
# --------------------------------------------------

def tokenize_question(prompt, options, tokenizer):

    first_sentences = [prompt] * 5
    second_sentences = [options[opt] for opt in OPTIONS]

    encoded = tokenizer(
        first_sentences,
        second_sentences,
        padding="max_length",
        truncation="only_first",
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    # Add batch dimension
    inputs = {
        "input_ids": encoded["input_ids"].unsqueeze(0),
        "attention_mask": encoded["attention_mask"].unsqueeze(0)
    }

    if "token_type_ids" in encoded:
        inputs["token_type_ids"] = (
            encoded["token_type_ids"].unsqueeze(0)
        )

    return inputs


# --------------------------------------------------
# LOAD ONE MODEL FOLD
# --------------------------------------------------

def load_model_fold(model_type, fold, token):

    folder = f"{model_type}_fold_{fold}"

    # Download the model files from the private HF repository
    config_file = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=f"{folder}/config.json",
        token=token
    )

    model_file = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=f"{folder}/model.safetensors",
        token=token
    )

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()

    # Copy downloaded files into the directory expected by Transformers
    import shutil

    shutil.copy(config_file, os.path.join(temp_dir, "config.json"))
    shutil.copy(model_file, os.path.join(temp_dir, "model.safetensors"))

    model = AutoModelForMultipleChoice.from_pretrained(
        temp_dir
    )

    return model


# --------------------------------------------------
# GET MODEL PREDICTIONS
# --------------------------------------------------

def get_model_probability(
    model_type,
    base_model,
    prompt,
    options,
    token
):

    tokenizer = (
        deberta_tokenizer
        if model_type == "deberta"
        else roberta_tokenizer
    )

    fold_probabilities = []

    # Run one fold at a time
    for fold in range(N_FOLDS):

        st.write(
            f"Running {model_type.upper()} fold {fold + 1}/5..."
        )

        model = load_model_fold(
            model_type,
            fold,
            token
        )

        model.eval()

        inputs = tokenize_question(
            prompt,
            options,
            tokenizer
        )

        with torch.no_grad():

            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                token_type_ids=inputs.get(
                    "token_type_ids",
                    None
                )
            )

        logits = outputs.logits

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        fold_probabilities.append(
            probabilities.cpu().numpy()[0]
        )

        # Free memory before loading next fold
        del model
        del inputs
        del outputs

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Average the five fold probabilities
    return np.mean(
        fold_probabilities,
        axis=0
    )


# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------

st.set_page_config(
    page_title="Smart MCQ Solver",
    page_icon="🧠"
)

st.title("🧠 Smart MCQ Solver")

st.write(
    "Answer multiple-choice questions using "
    "a DeBERTa + RoBERTa ensemble."
)

prompt = st.text_area(
    "Enter your question:"
)

st.subheader("Answer Options")

option_a = st.text_input("A")
option_b = st.text_input("B")
option_c = st.text_input("C")
option_d = st.text_input("D")
option_e = st.text_input("E")


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button("Predict"):

    if not prompt.strip():

        st.warning("Please enter a question.")

        st.stop()

    options = {
        "A": clean_text(option_a),
        "B": clean_text(option_b),
        "C": clean_text(option_c),
        "D": clean_text(option_d),
        "E": clean_text(option_e)
    }

    if any(not value for value in options.values()):

        st.warning(
            "Please enter all five answer options."
        )

        st.stop()

    # Clean question
    prompt = clean_text(prompt)

    # Get Hugging Face token
    hf_token = st.secrets["HF_TOKEN"]

    st.info("Running the ensemble. This may take some time...")

    # --------------------------------------------------
    # DeBERTa
    # --------------------------------------------------

    deberta_probs = get_model_probability(
        "deberta",
        DEBERTA_BASE,
        prompt,
        options,
        hf_token
    )

    # --------------------------------------------------
    # RoBERTa
    # --------------------------------------------------

    roberta_probs = get_model_probability(
        "roberta",
        ROBERTA_BASE,
        prompt,
        options,
        hf_token
    )

    # --------------------------------------------------
    # 70/30 PROBABILITY ENSEMBLE
    # Same as final Kaggle notebook
    # --------------------------------------------------

    final_probs = (
        0.70 * deberta_probs
        + 0.30 * roberta_probs
    )

    # Rank answers
    ranked_indices = np.argsort(
        final_probs
    )[::-1]

    top_three = [
        OPTIONS[i]
        for i in ranked_indices[:3]
    ]

    st.success(
        "Top 3 predicted answers: "
        + " → ".join(top_three)
    )

    # Show probabilities
    st.subheader("Prediction Probabilities")

    for i in ranked_indices:

        st.write(
            f"**{OPTIONS[i]}:** "
            f"{final_probs[i]:.4f}"
        )

    st.caption(
        "Ensemble: 70% DeBERTa + 30% RoBERTa"
    )



