import requests

# Replace these with your Azure Form Recognizer endpoint and key
endpoint = 'https://westeurope.api.cognitive.microsoft.com/'
api_version = '2024-02-29-preview'  # Use the correct API version if required
key = '40482e269e274c149a6e81f2b38a8d97'

# Construct the URL to list all document models
url = f"{endpoint}/formrecognizer/documentModels?api-version={api_version}"

# Set the headers including the API key
headers = {
    "Ocp-Apim-Subscription-Key": key
}

# Send the GET request to list all models
response = requests.get(url, headers=headers)

# Check if the request was successful
if response.status_code == 200:
    models = response.json()
    print("List of Models:")
    for model in models.get('value', []):
        print(f"Model ID: {model['modelId']}, Created on: {model['createdOn']}")
else:
    # If the request failed, print the error message
    print(f"Failed to list models: {response.status_code} - {response.text}")
