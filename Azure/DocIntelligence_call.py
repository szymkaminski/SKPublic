# Install the necessary package
# pip install azure-ai-formrecognizer

from azure.ai.formrecognizer import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

# Replace these with your Azure Form Recognizer endpoint and key
endpoint = ""
key = ""

# Initialize the client
client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key)
                                        , api_version="2024-02-29-preview")

# Retrieve your model by ID
model_id = "IRS_confirmations_rates"

# Get the model details
model = client.get_document_model(model_id=model_id)

# Print out model details
print(f"Model ID: {model.model_id}")
print(f"Description: {model.description}")
print(f"Created on: {model.created_on}")

# List down the document types in the model
for doc_type in model.doc_types:
    print(f"Document type: {doc_type}")