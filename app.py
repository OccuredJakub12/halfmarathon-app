import joblib
import openai
import numpy as np
import os
from io import BytesIO

import json
import re
import streamlit as st

from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context


from dotenv import load_dotenv

load_dotenv()
# Inicjalizacja klienta Langfuse

lf_client = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST")
)

# Klucz OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

st.title("Predykcja czasu półmaratonu")


@st.cache_resource
def load_model_from_spaces():
    import boto3
    endpoint_url = os.getenv("DO_ENDPOINT")
    access_key = os.getenv("DO_ACCESS_KEY")
    secret_key = os.getenv("DO_SECRET_KEY")
    bucket = os.getenv("DO_BUCKET")
    model_key = "models/latest_model.pkl"

    session = boto3.session.Session()
    client = session.client(
        "s3",
        region_name="fra1",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    obj = client.get_object(Bucket=bucket, Key=model_key)
    buffer = BytesIO(obj["Body"].read())
    return joblib.load(buffer)

model = load_model_from_spaces()

user_input = st.text_area("Napisz coś o sobie (wiek, płeć, tempo na 5 km)")


@observe()
def extract_features_from_text(text: str) -> dict:
    prompt = f"""
Z tekstu wyciągnij:
- wiek (integer)
- płeć (0 = mężczyzna, 1 = kobieta)
- tempo na 5 km w sekundach (integer)

Tekst: '''{text}'''

Zwróć tylko czysty JSON w formacie:
{{"age": 32, "gender": 1, "pace_5k_seconds": 1680}}
Jeśli czegoś brak, użyj null.
"""
    data = {"age": None, "gender": None, "pace_5k_seconds": None}

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        content = response.choices[0].message.content.strip()
        content_clean = re.sub(r"^```json\s*|\s*```$", "", content, flags=re.IGNORECASE).strip()
        data = json.loads(content_clean)

    except Exception as e:
        st.error(f"Błąd parsowania LLM: {e}")
        raw = locals().get("content", "(brak odpowiedzi)")
        st.write("Odpowiedź LLM (niesparsowana):", raw)

    return data

def check_missing(data):
    miss = []
    if data.get("age") is None:       miss.append("wiek")
    if data.get("gender") is None:    miss.append("płeć")
    if data.get("pace_5k_seconds") is None: miss.append("tempo na 5 km")
    return miss

if st.button("Przewidź czas półmaratonu"):
    if not user_input.strip():
        st.warning("Proszę podać dane o sobie.")
    else:
        with st.spinner("Przetwarzam dane..."):
            features = extract_features_from_text(user_input)
            missing = check_missing(features)
            if missing:
                st.error(f"Brakuje danych: {', '.join(missing)}")
            else:
                X = np.array([[features["age"], features["gender"], features["pace_5k_seconds"]]])
                pred_sec = model.predict(X)[0]
                # format HH:MM:SS
                h, rem = divmod(int(pred_sec), 3600)
                m, s = divmod(rem, 60)
                st.success(f"Przewidywany czas: {h}h {m}m {s}s ({pred_sec} sekund)")


langfuse_context.flush()

