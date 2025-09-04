import requests
model_url="http://0.0.0.0:8009"
response = requests.post(
                f'{model_url}/v1/completions', 
                json={
                    "prompt": "xijinpin is",
                    "model": "ReCall",
                    "temperature": 0.0,
                    "max_tokens": 512
                }
            ).json()
print(response["choices"][0]["text"])