from groq import Client
import pandas as pd
import numpy as np
import json
from datetime import datetime
import openai
# from transformers import AutoTokenizer

def json_to_dataframe(data):
    # Check if data is a string and convert to list of dictionaries if necessary
    if isinstance(data, str):
        import json
        data = json.loads(data)
    
    # # Convert JSON data to DataFrame
    df = pd.DataFrame(data)
    # Convert 'date' column to datetime and 'value' column to float
    # df['date'] = pd.to_datetime(df['date'])
    # df['value'] = df['value'].astype(float)
    
    return df

# llama3-8b-8192
# gpt-4o-mini

selected_model="gpt-4o"

# Load Llama tokenizer
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")  # Replace with your specific model

if selected_model.startswith("gpt-"):
    openai.api_key = ''
else: 
    client = Client(api_key="")
    
system_message=f"""You are a financial assistant, you return financial data in json-like format: date,value for the asset and time period your user asks. 
please provide daily frequency of data. do not paste any text in addition to the financial data, unless the user asks you for something else related to the dataset. 
Wrap the json data in an array, i.e. your response should start with '[' and end with ']'. for currency data, please use 4 digits. Please always sort the data descending
"""
# date should be in date format, the value should be in numeric format

messages_array = [{"role": "system", "content": system_message}]

# daj tutaj jakąś listę tyvh dostępnych modeli
# dodaj też funkcję która wybiera klucz dostosowany do modelu, tak żeby można było z OpenAI API też korzystać, gdyby inne modele nie dawały rady

csvdata=[]
explanation=[]

while True:
    # Get input text
    # user_input = input("Enter the prompt (or type 'quit' to exit): ")
    user_input = "please select one month of EURPLN data with high volatility from last 10 years"
    if user_input.lower() == "quit":
        break
    if len(user_input) == 0:
        print("Please enter a prompt.")
        continue

    print(f"\nSending request for summary to {selected_model} endpoint...\n\n")
    
    # Send request to OpenAI model
    messages_array.append({"role": "user", "content": user_input})

    if selected_model.startswith("gpt-"):
        chat_completion = openai.chat.completions.create(
            messages=messages_array,
            model=selected_model,
            temperature=0.7, 
            max_tokens=1000
        )
    else:
        chat_completion = client.chat.completions.create(
        messages=messages_array,
        model=selected_model,
        temperature=0.7, 
        max_tokens=1000
        )
        
    model_response= chat_completion.choices[0].message.content
    # print(model_response)
    csvdata=model_response
    messages_array.append({"role": "assistant", "content": model_response})
    
    # tokens = tokenizer.encode(csvdata, add_special_tokens=True)
    # token_count = len(tokens)
    # print("Token count:", token_count)
    
    #in a second iteration, the model explains why he selected that period
    additional_question="please provide a written explanation what happened in the economy during this period"
    messages_array.append({"role": "user", "content": additional_question})
    
    if selected_model.startswith("gpt-"):
        chat_completion = openai.chat.completions.create(
            messages=messages_array,
            model=selected_model,
            temperature=0.7, 
            max_tokens=1000
        )
    else:
        chat_completion = client.chat.completions.create(
        messages=messages_array,
        model=selected_model,
        temperature=0.7, 
        max_tokens=100
        )
    
    model_response= chat_completion.choices[0].message.content
    messages_array.append({"role": "assistant", "content": model_response})
    explanation=model_response
    
    break

# print(csvdata)
print(csvdata)
csvdata=json_to_dataframe(csvdata)
financial_info=np.array(csvdata.iloc[:,1])
rets=np.log(financial_info[:-1] / financial_info[1:])

print(csvdata)
print(financial_info)
print(rets)

StartValue=4

# Calculate cumulative product of exponential returns
price_path = np.exp(np.cumsum(rets)) * StartValue
# Insert the starting value as the first price
price_path = np.insert(price_path, 0, StartValue)

print(price_path)
print(explanation)
