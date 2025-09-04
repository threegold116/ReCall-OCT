#### 文件
1. Mirrow-Api model: a-vllm_server.sh
2. Rappid Api: a-api_server.sh(包含a-vllm_server.sh功能)
3. ReCall模型服务: a-sglang_server.sh
4. 测试文件: a-eval.sh


#### 测试服务
1. api服务
python test_server.py
2. ReCall服务

curl http://0.0.0.0:8010/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "ReCall",
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
curl http://0.0.0.0:8010/v1/completions -H "Content-Type: application/json" -d '{
  "model": "ReCall",
  "prompt": "xijinpin is",
  "temperature": 0.7,
  "top_p": 0.8,
  "top_k": 20,
  "max_tokens": 200,
  "presence_penalty": 1.5
}'

curl http://0.0.0.0:8010/generate -H "Content-Type: application/json" -d '{
  "prompt": "xijinpinu ",
  "sampling_params": {
      "temperature": 0.0,
      "max_new_tokens": 512
  }
}'

3. Judge服务

curl http://0.0.0.0:8888/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "Qwen2.5-72B-Instruct-GPTQ-Int4",
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

#### 运行逻辑
1. 顺序
```
bash ./a-api_server.sh #运行api服务
bash ./a-recall_server.sh #运行recall模型
bash ./a-eval_run.sh #运行推理服务
bash ./a-eval_mertic.sh #运行指标计算


```

2. 需要修改
- a-recall_server.sh中的model_path
- a-eval_run.sh中的model_name和mode