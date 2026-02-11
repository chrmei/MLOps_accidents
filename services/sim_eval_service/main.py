import json
import requests
import pandas as pd
import os

USERNAME = os.getenv("ADMIN_USERNAME", "admin")
PASSWORD = os.getenv("ADMIN_PASSWORD", "admin")
AUTH_API_URL = "http://auth:8004/api/v1/auth/login"
EVAL_API_URL = "http://predict:8003/api/v1/predict/evaluate"
INTERIM_DATA_PATH = "/app/data/preprocessed/interim_dataset.csv"
REF_DATA_START = 0
REF_DATA_SIZE = 1000
EVAL_DATA_START = 1000 # Start of 1000 has no data drift, Start of -1000 has data drift
EVAL_DATA_SIZE = 1000

def slice_by_start_and_size(df: pd.DataFrame, start: int, size: int) -> pd.DataFrame:
    n = len(df)
    if start < 0:
        start += n
    start = max(0, start)
    end = start + size
    return df.iloc[start:end]

if __name__ == "__main__":
    credentials = {
        "username": USERNAME,
        "password": PASSWORD,
    }
    auth_response = requests.post(AUTH_API_URL, json=credentials)
    auth_response.raise_for_status()
    access_token = auth_response.json().get("access_token")
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    input_data = pd.read_csv(INTERIM_DATA_PATH)
    input_data = input_data.sort_values(by=['an', 'mois', 'jour']).reset_index(drop=True)
    ref_df = slice_by_start_and_size(input_data, REF_DATA_START, REF_DATA_SIZE)
    eval_df = slice_by_start_and_size(input_data, EVAL_DATA_START, EVAL_DATA_SIZE)
    ref_dicts = ref_df.to_dict(orient="records")
    eval_dicts = eval_df.to_dict(orient="records")
    payload = {
        "eval_data": eval_dicts,
        "ref_data": ref_dicts if ref_dicts else None,
    }

    response = requests.post(
        EVAL_API_URL,
        json=payload,
        headers=headers,
        timeout=300
    )

    response.raise_for_status()
    print(f"Evaluation successful. Response code: {response.status_code}")
    print("Evaluation Metrics:")
    print(json.dumps(response.json(), indent=4))
