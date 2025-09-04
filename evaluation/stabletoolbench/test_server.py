import requests
import json
import os

url = 'http://0.0.0.0:2222/virtual'
data = {
    "category": "Artificial_Intelligence_Machine_Learning",
    "tool_name": "TTSKraken",
    "api_name": "List Speakers",
    "tool_input": '{}',
    "strip": "truncate",
    "toolbench_key": ""
}
{   'category': 'Media', 
    'tool_name': 'vimeo', 
    'api_name': 'getrelatedtags', 
    'tool_input': 
        '{"category":"Music","format":"json"}', 
    'strip': 'truncate', 
    'toolbench_key': ''
}
headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json',
}
key="D3pxHh4hTLhVVH6HnD5Vg50MhL9PYqgv5rQTntGuBh9IQsoZS8"
# Make the POST request
# response = requests.post(url, headers=headers, data=json.dumps(data))
# print(response.text)

url = 'http://8.130.32.149:8080/rapidapi'

url = 'http://0.0.0.0:2222/virtual'
data = {
    "category": "Artificial_Intelligence_Machine_Learning",
    "tool_name": "TTSKraken",
    "api_name": "List Languages",
    "tool_input": '{}',
    "strip": "truncate",
    "toolbench_key": key
}

# url = 'http://8.130.32.149:8080/rapidapi'
# response = requests.post(url, headers=headers, data=json.dumps(data))
# print(response.text)
# data = {
#     "category": "Artificial_Intelligence_Machine_Learning",
#     "tool_name": "TTSKraken",
#     "api_name": "List Languages",
#     "tool_input": '{}',
#     "strip": "truncate",
#     "toolbench_key": key
# }

response = requests.post(url, headers=headers, data=json.dumps(data))
print(response.text)