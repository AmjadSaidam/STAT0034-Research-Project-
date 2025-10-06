import zipfile
import pandas as pd 
from sklearn.model_selection import train_test_split
import re 
from polygon import RESTClient
import requests
import numpy as np
import time
import streamlit as st

# Thesis Data for Reproducable Results 
# =============================================
def extract_ticker(asset_name: str) -> str:
    """
    Extracts the ticker from a string like 'NVDA(some text)'.
    Returns the cleaned ticker in uppercase.
    """
    match = re.match(r"([A-Z]+)\s*\(", asset_name.upper())
    if match:
        return match.group(1)
    raise ValueError(f"Ticker not found in: {asset_name}")

def ticker_standerdise(data):
    # insure only ticker names 
    tickers = [extract_ticker(data.columns[i]) for i in range(len(data.columns))]
    data.columns = tickers

def process_zip_with_csvs(zip_path, column_extract = "Close", get_prices = False):
    # Step 1: Load and extract data from ZIP
    data_frames = []
    
    # read "r" zip file 
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get list of CSV files in the ZIP
        csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
        
        for csv_file in csv_files:
            # Read CSV directly from ZIP
            with zip_ref.open(csv_file) as f:
                df = pd.read_csv(f)
                
                # Extract close price  
                # Adjust column name as needed for your data
                if column_extract in df.columns:
                    # Create a single-column DataFrame with returns
                    temp_df = pd.DataFrame()
                    if get_prices:
                        temp_df['Prices'] = df[column_extract].dropna(how = 'any')
                    else:
                        temp_df['Returns'] = df[column_extract].pct_change().dropna(how = "any")
                    
                    # Use filename (without extension) as column name
                    asset_name = csv_file.split('/')[-1].replace('.csv', '')
                    temp_df.columns = [asset_name]
                    
                    data_frames.append(temp_df)
    
    # Step 2: Combine all DataFrames
    if not data_frames:
        raise ValueError(f"No valid CSV files with '{column_extract}' prices found in the ZIP")
    
    combined_df = pd.concat(data_frames, axis=1)
    
    # Drop any rows with NA values (that might remain after pct_change)
    combined_df = combined_df.dropna()
    ticker_standerdise(combined_df)
    
    # Step 3: Train-test split
    # Convert to numpy array (each column is an asset's return series)
    returns_array = combined_df.values
    
    # Split into training and testing sets (80-20 split)
    asset_returns, asset_returns_unseen = train_test_split(
        returns_array, 
        test_size=0.3, 
        shuffle=False  # Important for time series data
    )
    
    return combined_df, asset_returns, asset_returns_unseen

# FMP API Data
# =============================================
def fmp_data(tickers, start_date, end_date, get_returns = True, api_key = ""): 
    data = pd.DataFrame()
    for ticker in tickers:
        url = f"https://financialmodelingprep.com/stable/historical-price-eod/full?symbol={ticker}&from={start_date}\
            &to={end_date}&apikey={api_key}"
        res = requests.get(url)
        json_data = res.json() # transform to json format 
        data[ticker] = [json_data[i]['close'] for i in range(len(json_data))] # for each dict get required key
    if get_returns:
        data = data.pct_change().dropna(how = 'any')

    # split data 
    data_in_sample, data_out_sample = train_test_split(np.array(data), test_size = 0.3, shuffle  = False)
    
    return data, data_in_sample, data_out_sample

# Polygon.io API Data 
# =============================================
def get_poly(client, **inputs): 
    """Get Single Ticker, Custom Inputs:
    """
    return list(client.list_aggs(**inputs))

def polygon_data(poly_rest_api_key, tickers, start_date, end_date, get_returns = True, key = 'close'):    
    client = RESTClient(api_key = poly_rest_api_key)
    data = pd.DataFrame()
    # for default plan we must batch data, limit of 5 API calls per minute 
    for i in range(0, len(tickers), 5):
        batch = tickers[i: i+5]
        for ticker in batch:
            aggs = []
            try:
                for a in client.list_aggs(ticker=ticker, multiplier=1, timespan="day", from_= start_date, to = end_date):
                    aggs.append(a)
            except Exception as e:
                print(e)
            data[ticker] = [getattr(aggs, key) for agg in aggs] # filter for key, here for each ticker argument in ticker arguments get the close
        
        # throttle to avoid rate limit
        time.sleep(60) 
    
    # returns data or price data
    if get_returns:
        data = data.pct_change().dropna(how = 'any')
    else:
        pass

    # in/out sample data 
    in_sample_data, out_sample_data = train_test_split(
        np.array(data), 
        test_size=0.3, 
        shuffle=False
    )

    return data, in_sample_data, out_sample_data

        