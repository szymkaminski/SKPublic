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

import numpy as np
import pandas as pd

def simulate_price_path(process, T=1, n=100, mu=0, sigma=1, theta=0, kappa=0, x0=0):
    """
    Simulates price paths using specified stochastic process.
    
    Parameters:
    - process (str): The process to simulate ('brownian', 'vasicek', or 'ornstein_uhlenbeck').
    - T (float): Total time (e.g., 1 year).
    - n (int): Number of time steps.
    - mu (float): Drift term (Brownian/Vasicek).
    - sigma (float): Volatility term.
    - theta (float): Mean reversion level (Vasicek/O-U).
    - kappa (float): Mean reversion speed (Vasicek/O-U).
    - x0 (float): Initial value of the process.

    Returns:
    - pd.DataFrame: DataFrame containing time steps and simulated values.
    """
    dt = T / n  # Time step
    time = np.linspace(0, T, n + 1)
    path = np.zeros(n + 1)
    path[0] = x0
    
    if process.lower() == 'brownian':
        for i in range(1, n + 1):
            path[i] = path[i-1] + mu * dt + sigma * np.sqrt(dt) * np.random.normal()
    
    elif process.lower() == 'vasicek':
        for i in range(1, n + 1):
            path[i] = (path[i-1] +
                       kappa * (theta - path[i-1]) * dt +
                       sigma * np.sqrt(dt) * np.random.normal())
    
    elif process.lower() == 'ornstein_uhlenbeck':
        for i in range(1, n + 1):
            path[i] = (path[i-1] +
                       kappa * (theta - path[i-1]) * dt +
                       sigma * np.sqrt(dt) * np.random.normal())
    else:
        raise ValueError("Invalid process. Choose 'brownian', 'vasicek', or 'ornstein_uhlenbeck'.")

    return pd.DataFrame({'Time': time, 'Value': path})


selected_model="gpt-4o"

# Load Llama tokenizer
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")  # Replace with your specific model

if selected_model.startswith("gpt-"):
    openai.api_key = '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'
else: 
    client = Client(api_key="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")
    
system_message=f"""You are a financial assistant, you return financial data in json-like format: date,value for the asset and time period your user asks. 
please provide daily frequency of data. do not paste any text in addition to the financial data, unless the user asks you for something else related to the dataset. 
Wrap the json data in an array, i.e. your response should start with '[' and end with ']'. for currency data, please use 4 digits. Please always sort the data descending
"""
# date should be in date format, the value should be in numeric format

messages_array = [{"role": "system", "content": system_message}]

    # Define the function's JSON schema
    function_definition = {
        "name": "simulate_price_path",
        "description": "Simulates price paths using specified stochastic process.",
        "parameters": {
            "type": "object",
            "properties": {
                "process": {"type": "string", "description": "The process to simulate ('brownian', 'vasicek', 'ornstein_uhlenbeck')."},
                "T": {"type": "number", "description": "Total time (e.g., 1 year)."},
                "n": {"type": "integer", "description": "Number of time steps."},
                "mu": {"type": "number", "description": "Drift term (Brownian/Vasicek)."},
                "sigma": {"type": "number", "description": "Volatility term."},
                "theta": {"type": "number", "description": "Mean reversion level (Vasicek/O-U)."},
                "kappa": {"type": "number", "description": "Mean reversion speed (Vasicek/O-U)."},
                "x0": {"type": "number", "description": "Initial value of the process."}
            },
            "required": ["process", "T", "n", "sigma", "x0"]
        }
    }

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
            max_tokens=1000,
            functions=function_definition,
            function_call="auto"
        )
            # Check if OpenAI decided to call the function
        if "function_call" in response["choices"][0]["message"]:
            function_call = response["choices"][0]["message"]["function_call"]
            arguments = function_call["arguments"]

            # Save arguments into dedicated variables
            process = arguments.get("process", process)
            T = float(arguments.get("T", T))
            n = int(arguments.get("n", n))
            mu = float(arguments.get("mu", mu))
            sigma = float(arguments.get("sigma", sigma))
            theta = float(arguments.get("theta", theta))
            kappa = float(arguments.get("kappa", kappa))
            x0 = float(arguments.get("x0", x0))
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
