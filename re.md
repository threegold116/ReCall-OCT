curl http://0.0.0.0:8009/generate -H "Content-Type: application/json" -d '{
  "model": "ReCall-Origin",
  "messages": [
    {"role": "user", "content": "Give me a short introduction to large language models."}
  ],
  "temperature": 0.7,
  "top_p": 0.8,
  "top_k": 20,
  "max_tokens": 200,
  "presence_penalty": 1.5,
  "chat_template_kwargs": {"enable_thinking": false}
}'
curl http://0.0.0.0:8009/generate -H "Content-Type: application/json" -d '{
  "model": "ReCall-Origin",
  "text": "Beijin is",
  "sampling_params": {
                        "temperature": 0.0,
                        "max_new_tokens": 512
                    }
}'

curl http://localhost:8009/v1/models