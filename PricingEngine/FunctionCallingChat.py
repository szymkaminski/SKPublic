# KONIECZNIE SKRÓĆ ZAKRES DANYCH IMPORTOWANYCH BO TO BARDZO DUŻO TOKENÓW ZUŻYWA!!!!
# tu pamiętam że interpolacja chyba nie chciała działać. Przed przerzuceniem do właściwej funkcji proponuję rozwiązać ten problem
# chyba krokiem nr 1 jest podanie tabeli z discount factorami nie w formie wklejenia do czatu ale wczytania z pliku
# chyba, żeby poprosić użytkownika, żeby wkleił do czata link. myslę, że najpierw trzeba ogarnąc problem z funkcją Interp a potem się zobaczy co z tym
# wklejenie do czatu to będzie później to ogarnięcia, chyba że skorzystam z tej funkcji Parse co jest w LoadingData?
# żeby działał interp chyba trzeba go zmienić podobnie jak zrobiłem w pliku LoadingData

# Function call zwraca wyniki w JSonie, więc moim zdaniem 1szą rzecz co trzeba zrobić po function callu
# to przerobienie tego na DataFrame

# I think in order for the code to work I should convert the Json output from OpenAI to some kind of data frame
# to be use'able for the Interp function

import openai
import json
from datetime import datetime
from scipy.interpolate import interp1d
import pandas as pd


# Define your OpenAI API key
openai.api_key = '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'

def read_csv_file(file_name):
    return pd.read_csv(file_name, sep='[;]', engine='python', header=0, decimal=',')

DiscountCurves = read_csv_file(r"C:\Users\skaminski\Documents\03. Projekty\Aramco\Python\Input Pricing Engine\DiscountCurvesFC.csv")

def json_to_dataframe(json_data):
    # Convert JSON to DataFrame
    df = pd.DataFrame(json_data)
    print(df)
    # Convert 'CFDate' column to datetime format
    df['CFDate'] = pd.to_datetime(df['CFDate'], format='%Y-%m-%d')
    
    return df

def AccrualFactor(BeginDate, FinalDate, DayCountConvention):
    # funkcja jeszcze niedopracowana bo obsługuje tylko dwie konwencje - będa dodawane w zależności od potrzeb
    # Act/365 jest fixed, warto byłoby zrobić też Act/365(actual)
        # Convert BeginDate and FinalDate from strings to datetime objects
    BeginDate = datetime.strptime(BeginDate, "%Y-%m-%d")
    FinalDate = datetime.strptime(FinalDate, "%Y-%m-%d")
    DayCountConvention=int(DayCountConvention)
    
    delta = (FinalDate - BeginDate).days
    if DayCountConvention == 1:
        return delta / 365
    elif DayCountConvention == 2:
        return delta / 360
    else:
        return delta / 100

def Interp(DscCurves, InterpDate):
    # Convert InterpDate to datetime if it's a string
    # df=pd.DataFrame(DscCurves)
    DscCurves=json_to_dataframe(DscCurves)
    print(DscCurves)
    
    if isinstance(InterpDate, str):
        InterpDate = datetime.strptime(InterpDate, '%Y-%m-%d')
    # def convert_to_datetime(date_str):
    #     for date_format in ('%Y-%m-%d', '%d.%m.%Y'):
    #         try:
    #             return datetime.strptime(date_str, date_format)
    #         except ValueError:
    #             continue
    #     raise ValueError(f"Date format for '{date_str}' is unsupported.")
    
    # print(DscCurves.iloc[:,'CFDate'])
    # DscCurves['CFDate'] = DscCurves['CFDate'].apply(
    #     lambda x: convert_to_datetime(x) if isinstance(x, str) else x
    # )
    
    # Convert dates to ordinal for interpolation
    # dates_ordinal = DscCurves['CFDate'].map(lambda x: x.toordinal())
    # interp_date_ordinal = InterpDate.toordinal()
    
    dates_ordinal = DscCurves['CFDate'].map(datetime.toordinal)
    interp_date_ordinal = InterpDate.toordinal()
    
    # Create interpolation function
    interp_function = interp1d(dates_ordinal, DscCurves['Discount'], kind='linear', fill_value='extrapolate')
    
    # Interpolate the value
    interpolated_value = interp_function(interp_date_ordinal)
    
    return interpolated_value


# Prepare the system message
system_message = f"""
You are a helpful assistant that specializes in financial mathematics. 
If the user wants to calculate the accrual, use the AccrualFactor function
Use this format for the call: AccrualFactor(BeginDate,FinalDate,DayCountConvention). 
Before calling the function, please convert dates BeginDate and FinalDate to numbers
If the user wants to interpolate a value, use {DiscountCurves} as the source for DSCCurves
and ask the user for interpolation date (InterpDate)
"""
# Use this format for the call: Calculator(operation, num1, num2)
# Function to handle the OpenAI API call
def chat_with_openai(user_input, messages):
    messages.append({"role": "user", "content": user_input})
    
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=messages,
        functions=[
            {
                "name": "AccrualFactor",
                "description": "Calculates the value of the accrual factor for a given set of user inputs",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "DayCountConvention": {
                            "type": "integer",
                            "enum": [1,2],
                            "description": "1 for Actual/365, 2 for Actual/360"
                        },
                        "BeginDate": {
                            "type": "string",
                            "format": "date",
                            "description": "User inputs the Start date in the format YYYY-MM-DD"
                        },
                        "FinalDate": {
                            "type": "string",
                            "format": "date",
                            "description": "End date in the format YYYY-MM-DD"
                        }
                    },
                    "required": ["DayCountConvention", "BeginDate", "FinalDate"],
                },
            },
            {
                "name": "Interp",
                "description": "The function returns the interpolated discount value for a given date provided by the user",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "DscCurves": {
                            "type": "array",
                            "description": "Discount curve data with dates and discount factors",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "CFDate": {
                                        "type": "string",
                                        "format": "date",
                                        "description": "Date of the cash flow in YYYY-MM-DD format"
                                    },
                                    "Discount": {
                                        "type": "number",
                                        "description": "Discount factor associated with the CFDate"
                                    }
                                },
                                "required": ["CFDate", "Discount"]
                            }
                        },
                        "InterpDate": {
                            "type": "string",
                            "format": "date",
                            "description": "Date for which the interpolated discount value is needed, in YYYY-MM-DD format"
                        }
                    },
                    "required": ["DscCurves", "InterpDate"]
                },
            }        
        ],
        function_call="auto",  # This enables the model to decide when to call a function
    )

    # If OpenAI decides to call a function
    if response.choices[0].finish_reason == "function_call":
        function_call = response.choices[0].message.function_call
        # Extract the function arguments
        function_name=function_call.name
        arguments = function_call.arguments
        arguments2=json.loads(arguments)
        
        if function_name=="AccrualFactor":
            operation = arguments2["DayCountConvention"]
            num1 = arguments2["BeginDate"]
            num2 = arguments2["FinalDate"]
            
            result = AccrualFactor(num1, num2, operation)
            assistant_message= f"The accrual factor between {num1} and {num2} for day-count convention {operation} is: {result}"

        elif function_name=="Interp":
            DscCurves=arguments2["DscCurves"]
            InterpDate=arguments2["InterpDate"]
            # print(arguments2)
            print(DscCurves)
            result=Interp(DscCurves,InterpDate)
            assistant_message = f"The result of the interpolation for {InterpDate} is: {result}"
        
        messages.append({"role": "assistant", "content": assistant_message})
        return assistant_message    
    else:
        assistant_message=response.choices[0].message.content
        messages.append({"role": "assistant", "content": assistant_message})
        return assistant_message

messages = [{"role": "system", "content": system_message}]

while True:
    # Get input text
    user_input = input("Enter the prompt (or type 'quit' to exit): ")
    if user_input.lower() == "quit":
        break
    if len(user_input) == 0:
        print("Please enter a prompt.")
        continue

    #get the response from OpenAI and print it
    output = chat_with_openai(user_input, messages)
    print(output)
