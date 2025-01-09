import requests

API_URL = "https://skaminski88-flowiseai-sk.hf.space/api/v1/prediction/b08a80c5-ac31-4693-baad-e5712b9cf673"

def query(payload):
    response = requests.post(API_URL, json=payload)
    return response.json()
    
output = query({
    "question": "I have problem with motivation of employees",
})

print(output)