import sys

import pandas as pd
from datetime import datetime
import numpy as np
from scipy.interpolate import interp1d
from openpyxl import load_workbook
from openpyxl.utils.dataframe import dataframe_to_rows


# add link to Azure doc intelligence https://documentintelligence.ai.azure.com/studio

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

def read_csv_file(file_name):
    return pd.read_csv(file_name, sep='[;]', engine='python', header=0, decimal=',')

def Interp(DscCurves, InterpDate):
    # Convert InterpDate to datetime if it's a string
    if isinstance(InterpDate, str):
        InterpDate = datetime.strptime(InterpDate, '%Y-%m-%d')
    
    # Convert dates to ordinal for interpolation
    dates_ordinal = DscCurves['CFDate'].map(datetime.toordinal)
    interp_date_ordinal = InterpDate.toordinal()
    
    # Create interpolation function
    interp_function = interp1d(dates_ordinal, DscCurves['Discount'], kind='linear', fill_value='extrapolate')
    
    # Interpolate the value
    interpolated_value = interp_function(interp_date_ordinal)
    
    return interpolated_value

def AccrualFactor(BeginDate, FinalDate, DayCountConvention):
    # funkcja jeszcze niedopracowana bo obsługuje tylko dwie konwencje - będa dodawane w zależności od potrzeb
    # Act/365 jest fixed, warto byłoby zrobić też Act/365(actual)
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

def main():
    # Load CSV files
    DiscountCurves = read_csv_file(r"\DiscountCurvesOLD.csv")
    RepaymentSchedule = read_csv_file(r"\RepaymentScheduleOLD.csv")
    
    # Convert relevant columns to datetime
    if 'CFDate' in DiscountCurves.columns:
        DiscountCurves['CFDate'] = pd.to_datetime(DiscountCurves['CFDate'], format='%d.%m.%Y')
    if 'StartDate' in RepaymentSchedule.columns:
        RepaymentSchedule['StartDate'] = pd.to_datetime(RepaymentSchedule['StartDate'], format='%d.%m.%Y')
    if 'EndDate' in RepaymentSchedule.columns:
        RepaymentSchedule['EndDate'] = pd.to_datetime(RepaymentSchedule['EndDate'], format='%d.%m.%Y')
    if 'PaymentDate' in RepaymentSchedule.columns:
        RepaymentSchedule['PaymentDate'] = pd.to_datetime(RepaymentSchedule['PaymentDate'], format='%d.%m.%Y')
    
    
    # Prepare ResultsTable with appropriate data types
    ResultsTable = pd.DataFrame(columns=['IRSCode','StartDate', 'EndDate', 'PaymentDate', 'FaceValue', 
                                            'UndscCFFixed','UndscCFFloat','DF', 'FixedRate','FloatRate',
                                            'DscCFFixed','DscCFFloat'])
    
    # prepares table that stores fixed rates
    fix_rates=pd.DataFrame(columns=['IRSCode','FixedRate'])
    
    # Load input data intoResultsTable from source table
    ResultsTable['IRSCode'] = RepaymentSchedule['IRSCode']
    ResultsTable['StartDate'] = RepaymentSchedule['StartDate']
    ResultsTable['EndDate'] = RepaymentSchedule['EndDate']
    ResultsTable['PaymentDate'] = RepaymentSchedule['PaymentDate']
    ResultsTable['FaceValue'] = RepaymentSchedule['FaceValue']

    unique_irscode = ResultsTable['IRSCode'].unique()
    
    # Get user inputs
    print("Please provide the company name as a string.")
    CompanyName = get_input("Enter Company Name: ", 'string')
    
    x=0
    for irscode in unique_irscode:
        print(f"Please provide the fixed rate for IRS '{irscode}' as a percentage (e.g., 5%).")
        fix_rates.loc[x,'IRSCode']=irscode
        fix_rates.loc[x,'FixedRate'] = get_input("Enter Fixed Rate (as percentage, e.g., 5%): ", 'percentage')
        x=x+1

    MatchRates=pd.merge(fix_rates,ResultsTable,on='IRSCode',how='right')
    
#    print("Please provide the fixed rate as a percentage (e.g., 5%).")
#    FixedRate = get_input("Enter Fixed Rate (as percentage, e.g., 5%): ", 'percentage')
    
    # tu trzeba też dodac pętlę po IRSach, podobnie jak dla FixedRAte'u
    print("Please provide the floating rate for the first flow as a percentage (e.g., 5%).")
    FirstFlowRate = get_input("Enter Fixed Rate (as percentage, e.g., 5%): ", 'percentage')

    print("Please provide the valuation date in the format YYYY-MM-DD.")
    ValuationDate = get_input("Enter Valuation Date (YYYY-MM-DD): ", 'date')

    # MAIN CODE
    
    # Initialize the loop
    for i, (DFs, StartDFs, EndDFs) in enumerate(zip(RepaymentSchedule['PaymentDate'], RepaymentSchedule['StartDate'], RepaymentSchedule['EndDate'])):
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
    print(f"The fair value of the instrument is equal to {sum(ResultsTable['DscCFFixed'])-sum(ResultsTable['DscCFFloat'])}.")
    print (fix_rates)
    
    # Load the base Excel file
    base_excel_path = r"\BaseFileIRS.xlsx"
    wb = load_workbook(base_excel_path)
    
    # Builds a custom name for the xls file and saves it into a given folder
    Datename=ValuationDate.strftime("%Y-%m-%d")
    outputfilepath=(r"\Output Pricing Engine")
    filename=outputfilepath+'\\'+CompanyName+'_IRS_'+Datename+'.xlsx'
    ResultsTable.to_excel(filename, index=False)
    
if __name__ == "__main__":
    main()
