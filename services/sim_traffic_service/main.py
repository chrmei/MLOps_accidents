from time import sleep
import requests
import pandas as pd
import os

USERNAME = os.getenv("ADMIN_USERNAME", "admin")
PASSWORD = os.getenv("ADMIN_PASSWORD", "admin")
AUTH_API_URL = "http://auth:8004/api/v1/auth/login"
PREDICT_API_URL = "http://predict:8003/api/v1/predict/"
INTERIM_DATA_PATH = "/app/data/preprocessed/interim_dataset.csv"
NUM_REQ_PER_SEC = 20
NUM_SEC = 5
NUM_REQ = NUM_REQ_PER_SEC * NUM_SEC

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
    test_df = input_data.tail(NUM_REQ)
    test_dicts = test_df.to_dict(orient="records")

    for j, record in enumerate(test_dicts, 1):
        payload = {"features": record}
        
        response = requests.post(
            PREDICT_API_URL,
            json=payload,
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        if j % NUM_REQ_PER_SEC == 0:
            print(f"{j} / {NUM_REQ} requests sent successfully, sleeping for 1 second...")
            sleep(1)

    print(f"All {NUM_REQ} prediction requests sent successfully.")