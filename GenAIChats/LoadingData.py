# next tasks
# 1. Test the process of correcting column names for the second table RepaymentSchedule -->DONE
# 2. Attempt to perform a whole valuation -->DONE and verified with AsPricingEngine --> DONE
# build a function that identifies a custom separator using OpenAI API
# generuje się zły separator, prawdopodobnie dlatego że źle czyta plik bez modelu do embeddings. Spróbuję z Pinecone
# tą bazą to się trzeba będzie pobawić na razie na jakimś przykładowym dokumencie
# 4. Embed the FunctionCalling feature so that the valuation can be performed using natural language
#       at this stage I think I should remove the possibility of pricing more than one IRS within one single file
#       solve the problem in the FunctionCallingChat with the Interp function first
# 5. Build the second agent who will read transaction confirmations 
# 6. combine the two agents together

# the AI separator is not properly passed as a separator - perhaps a conversion to str is required? to be investigated

import numpy as np
import pandas as pd
import openai
from datetime import datetime
from scipy.interpolate import interp1d

openai.api_key = ''

# Define your standard column names
#FaceValue to be potentially renamed to Notional or something similar
standard_columns = ['AsOfDate', 'Discount','CFDate','StartDate','EndDate','PaymentDate','FaceValue','IRSCode']

def get_input(prompt, input_type, validation_fn=None):
    while True:
        try:
            user_input = input(prompt)
            if input_type == 'string':
                if validation_fn and not validation_fn(user_input):
                    raise ValueError
                return user_input
            elif input_type == 'percentage':
                value = float(user_input.strip('%')) / 100
                return value
            elif input_type == 'date':
                value = datetime.strptime(user_input, '%Y-%m-%d')
                return value
            else:
                raise ValueError("Unsupported input type")
        except ValueError:
            if input_type == 'string':
                print("Invalid input. Please enter a valid string.")
            elif input_type == 'percentage':
                print("Invalid input. Please enter a valid percentage (e.g., 5%).")
            elif input_type == 'date':
                print("Invalid input. Please enter a date in the format YYYY-MM-DD.")
            else:
                print("Unsupported input type.")

def get_column_mapping(table_columns):
    # Use OpenAI API to suggest a mapping for a given column name
    # loop allows to return a value for all columns in a given table
    # cleans the corrected_table_columns in case it had any previous content
    corrected_table_columns=[]
    for column in table_columns:
        system_message=[{"role": "system", "content": f"""Map the column name '{column}' to the most relevant standard column from {standard_columns}. 
                    Suggest the best match based on typical financial data context.
                    return only the column name, without additional text."""}]
        response = openai.chat.completions.create(
            model="gpt-4o-mini",  # Make sure to use a model that supports completions and contextual understanding
            messages=system_message,
            max_tokens=50,
            temperature=0
        )
        # Extract suggested column mapping from the response
        suggestion = response.choices[0].message.content
        corrected_table_columns.append(suggestion)
    return corrected_table_columns


def parse_pasted_data(data_str):
    try:
        # Convert pasted data string to a 2D array, assuming comma-separated
        data = [list(map(float, row.split(','))) for row in data_str.strip().splitlines()]
        return np.array(data)
    except ValueError as e:
        print("Invalid data format:", e)
        return None

def AISeparator(filename):
    system_message = f"""
    You are identifying the separator in file {filename}. You return just the separator, without any additional text or characters
    please look at the header of the file only
    """
    messages_array = [{"role": "system", "content": system_message}]
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages_array,
        temperature=0.05,
        max_tokens=100,
        top_p=0.05,
        frequency_penalty=0,
        presence_penalty=0
    )
    generated_text = response.choices[0].message.content
    # generated_text = str(generated_text).strip()
    return generated_text
    

def load_data_from_file(file_path):
    # I think in the future this function will need to be amended because there can be different separators. 
    # So far no idea how to do it, perhaps I should build a mini-chatbot whose only task will be to define separators?
    try:
        if file_path.endswith('.csv'):
            separator=AISeparator(file_path)
            print(separator)
            data = pd.read_csv(file_path, sep=';', engine='python', header=0, decimal=',', encoding='utf-8')
        elif file_path.endswith('.txt'):
            data = pd.read_csv(file_path, delimiter='\t' if '\t' in open(file_path).read(100) else ',')
        elif file_path.endswith(('.xls', '.xlsx')):
            data = pd.read_excel(file_path)
        else:
            print("Unsupported file format.")
            return None

        # Convert to 2D array if necessary
        if isinstance(data, pd.DataFrame):
            return data
    except Exception as e:
        print("Error loading file:", e)
        return None

def Interp(DscCurves, InterpDate):
    # Convert InterpDate to datetime if it's a string
    if isinstance(InterpDate, str):
        for date_format in ('%Y-%m-%d', '%d.%m.%Y'):
            try:
                InterpDate = datetime.strptime(InterpDate, date_format)
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"Date format for '{InterpDate}' is unsupported.")
    
    # Convert CFDate to datetime objects, handling both 'YYYY-MM-DD' and 'DD.MM.YYYY' formats
    def convert_to_datetime(date_str):
        for date_format in ('%Y-%m-%d', '%d.%m.%Y'):
            try:
                return datetime.strptime(date_str, date_format)
            except ValueError:
                continue
        raise ValueError(f"Date format for '{date_str}' is unsupported.")
    
    DscCurves['CFDate'] = DscCurves['CFDate'].apply(
        lambda x: convert_to_datetime(x) if isinstance(x, str) else x
    )
    
    # Convert dates to ordinal for interpolation
    dates_ordinal = DscCurves['CFDate'].map(lambda x: x.toordinal())
    interp_date_ordinal = InterpDate.toordinal()
    
    # Create interpolation function
    interp_function = interp1d(dates_ordinal, DscCurves['Discount'], kind='linear', fill_value='extrapolate')
    
    # Interpolate the value
    interpolated_value = interp_function(interp_date_ordinal)
    
    return interpolated_value

def AccrualFactor(BeginDate, FinalDate, DayCountConvention):
    # funkcja jeszcze niedopracowana bo obsługuje tylko dwie konwencje - będa dodawane w zależności od potrzeb
    # Act/365 jest fixed, warto byłoby zrobić też Act/365(actual)
    if isinstance(BeginDate, str):
        BeginDate = datetime.strptime(BeginDate, '%d.%m.%Y')  # Adjust format as needed
    if isinstance(FinalDate, str):
        FinalDate = datetime.strptime(FinalDate, '%d.%m.%Y')  # Adjust format as needed
    
    delta = (FinalDate - BeginDate).days
    
    if DayCountConvention == 1:
        return delta / 365
    elif DayCountConvention == 2:
        return delta / 360
    else:
        return delta / 365

def RateFWD(BeginDate, FinalDate, DiscountCurves, DayCountConvention):
    # Dummy forward rate calculation function
    df_begin = Interp(DiscountCurves, BeginDate)
    df_end = Interp(DiscountCurves, FinalDate)
    accrual = AccrualFactor(BeginDate, FinalDate, DayCountConvention)
    rate = (df_begin / df_end - 1) / accrual
    return rate

def Pricer(fix_rates,RepaymentSchedule,DiscountCurves,FirstFlowRate):#CompanyName, ValuationDate as optional arguments so far
    # Prepare ResultsTable with appropriate data types
    ResultsTable = pd.DataFrame(columns=['IRSCode','StartDate', 'EndDate', 'PaymentDate', 'FaceValue', 
                                            'UndscCFFixed','UndscCFFloat','DF', 'FixedRate','FloatRate',
                                            'DscCFFixed','DscCFFloat'])
    
    # Load input data intoResultsTable from source table
    ResultsTable['IRSCode'] = RepaymentSchedule['IRSCode']
    ResultsTable['StartDate'] = RepaymentSchedule['StartDate']
    ResultsTable['EndDate'] = RepaymentSchedule['EndDate']
    ResultsTable['PaymentDate'] = RepaymentSchedule['CFDate']
    ResultsTable['FaceValue'] = RepaymentSchedule['FaceValue']
    
    MatchRates=pd.merge(fix_rates,ResultsTable,on='IRSCode',how='right')

    # MAIN CODE
    
    # Initialize the loop
    for i, (DFs, StartDFs, EndDFs) in enumerate(zip(RepaymentSchedule['CFDate'], RepaymentSchedule['StartDate'], RepaymentSchedule['EndDate'])):
        # Saves discount factors in ResultsTable
        ResultsTable.loc[i, 'DF'] = Interp(DiscountCurves, DFs)
        # Stores FIXED rates
        ResultsTable.loc[i,'FixedRate']=MatchRates.loc[i,'FixedRate_x']
        # Calculates FIXED RATE flows
        ResultsTable.loc[i, 'UndscCFFixed'] = ResultsTable.loc[i, 'FaceValue'] * ResultsTable.loc[i,'FixedRate'] * AccrualFactor(StartDFs, EndDFs, 1)
        # Calculates FLOATING rates
        if i == 0:
            ResultsTable.loc[i, 'FloatRate'] = FirstFlowRate
        else:
            ResultsTable.loc[i, 'FloatRate'] = RateFWD(StartDFs, EndDFs, DiscountCurves, 1)
        # Calculates FLOATING RATE flows
        ResultsTable.loc[i, 'UndscCFFloat'] = ResultsTable.loc[i, 'FaceValue'] * ResultsTable.loc[i, 'FloatRate'] * AccrualFactor(StartDFs, EndDFs, 1)
        # Saves discounted flows for FIXED RATE
        ResultsTable.loc[i, 'DscCFFixed'] = ResultsTable.loc[i, 'UndscCFFixed'] * ResultsTable.loc[i, 'DF']
        # Saves discounted flows for FLOATING RATE
        ResultsTable.loc[i, 'DscCFFloat'] = ResultsTable.loc[i, 'UndscCFFloat'] * ResultsTable.loc[i, 'DF']

            
    print(ResultsTable)
    # the line below should be a loop which prices all IRSes separately and not all at once
    print(f"The fair value of the instrument is equal to {sum(ResultsTable['DscCFFixed'])-sum(ResultsTable['DscCFFloat'])}.")
    print (fix_rates)

def ColumnMapper(TableName):
    TableName=pd.DataFrame(TableName)
    columns=TableName.columns
    newcolumns=get_column_mapping(columns)
    TableName.columns=newcolumns
    return TableName
        
# data = parse_pasted_data(user_pasted_data)  # If user pasted data
# or
# data = load_data_from_file(file_path)  # If user uploaded a file

DiscountCurves = load_data_from_file(r"\DiscountCurves.csv")
RepaymentSchedule = load_data_from_file(r"\RepaymentSchedule.csv")
# tutaj przydałaby się pętla która od razu koryguje nazwy kolumn dla wszystkich tabel które zostały wczytane

# DSCCheck=r"\DiscountCurves.csv"
# separator=AISeparator(DSCCheck)
# print(f"The separator for the file is: '{separator}'")


DiscountCurves=ColumnMapper(DiscountCurves)
RepaymentSchedule=ColumnMapper(RepaymentSchedule)

print(DiscountCurves)
print(RepaymentSchedule)

# InterpDate=get_input("Enter interpolation date: ", 'date')
# print(Interp(DiscountCurves, InterpDate))

# below are all inputs to the Pricer function that the target AI model will have to ask using natural language
# print("Please provide the company name as a string.")
# CompanyName = get_input("Enter Company Name: ", 'string')

unique_irscode = RepaymentSchedule['IRSCode'].unique()

# prepares table that stores fixed rates
fix_rates=pd.DataFrame(columns=['IRSCode','FixedRate'])

x=0
for irscode in unique_irscode:
    print(f"Please provide the fixed rate for IRS '{irscode}' as a percentage (e.g., 5%).")
    fix_rates.loc[x,'IRSCode']=irscode
    fix_rates.loc[x,'FixedRate'] = get_input("Enter Fixed Rate (as percentage, e.g., 5%): ", 'percentage')
    x=x+1
    
# tu trzeba też dodac pętlę po IRSach, podobnie jak dla FixedRAte'u
print("Please provide the floating rate for the first flow as a percentage (e.g., 5%).")
FirstFlowRate = get_input("Enter Fixed Rate (as percentage, e.g., 5%): ", 'percentage')

# print("Please provide the valuation date in the format YYYY-MM-DD.")
# ValuationDate = get_input("Enter Valuation Date (YYYY-MM-DD): ", 'date')

# Pricer(fix_rates,RepaymentSchedule,DiscountCurves,FirstFlowRate)


