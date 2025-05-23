# %% [markdown]
# # Pipeline trenowania modelu półmaratonu

# %% 
# 📚 Importy i środowisko
import os, re
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
import joblib
import boto3
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, r2_score

# %% 
# 📡 Połączenie z DigitalOcean Spaces
from botocore.client import Config
session = boto3.session.Session()
client = session.client(
    "s3",
    region_name="fra1",
    endpoint_url=os.getenv("DO_ENDPOINT"),
    aws_access_key_id=os.getenv("DO_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("DO_SECRET_KEY"),
    config=Config(s3={"addressing_style":"path"})
)

# %% 
# 📥 Wczytanie CSV (2023 + 2024)
bucket = os.getenv("DO_BUCKET")
files = ['halfmarathon_wroclaw_2023__final.csv', 'halfmarathon_wroclaw_2024__final.csv']
dfs = []
for fname in files:
    obj = client.get_object(Bucket=bucket, Key=fname)
    dfs.append(pd.read_csv(obj["Body"], sep=";"))
df = pd.concat(dfs, ignore_index=True)
df.head()

# %% 
# 🔧 Normalizacja i czyszczenie
df.columns = [re.sub(r"\W+", "", c).lower() for c in df.columns]

# 1) Wiek
df['age'] = 2024 - pd.to_numeric(df['rocznik'], errors='coerce')
df = df[(df['age']>=10)&(df['age']<=100)]

# 2) Płeć
df['gender'] = df['płeć'].str.strip().str.lower().map({"m":0,"k":1})

# 3) 5km tempo → sekundy
df['5kmpace_sec'] = pd.to_numeric(df['5kmtempo'], errors='coerce') * 60

# 4) Parsowanie czasu półmaratonu (bez ryzyka błędu na 'DNS' itp.)
def time_to_seconds(t):
    if not isinstance(t, str) or t.count(':') != 2:
        return np.nan
    h, m, s = t.split(':')
    try:
        return int(h)*3600 + int(m)*60 + int(s)
    except ValueError:
        return np.nan

df['halfmarathontime_sec'] = df['czas'].apply(time_to_seconds)

# 5) Usuwamy wszystkie wiersze, które mają brakujące kluczowe dane
df.dropna(subset=['age','gender','5kmpace_sec','halfmarathontime_sec'], inplace=True)

# Wyświetl rozmiar po czyszczeniu
df.shape


# %% 
# 🔍 Feature selection (opcjonalnie)
X = df[['age','gender','5kmpace_sec']]
y = df['halfmarathontime_sec']
selector = SelectKBest(f_regression, k='all').fit(X, y)
pd.DataFrame({'feature': X.columns, 'score': selector.scores_})

# %% 
# 🏋️‍♂️ Trening i walidacja
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
print("MAE:", mean_absolute_error(y_test, model.predict(X_test)))
print("R2 :", r2_score(y_test, model.predict(X_test)))

# %% 
# 💾 Zapis modelu lokalnie i upload do Spaces
joblib.dump(model, "latest_model.pkl")
with open("latest_model.pkl", "rb") as f:
    client.put_object(Bucket=bucket, Key="models/latest_model.pkl", Body=f)

print("Model zapisany i wrzucony do Spaces.")

# %%
