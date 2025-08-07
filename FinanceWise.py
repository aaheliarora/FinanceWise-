import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans

def loadData(path="data.csv"):
    df = df.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df

def fetchStock(symbol="AAPL", start = "2024-01-01", end="2025-01-01"):
    return yf.download(symbol,start = start, end=end)

def calculateKPIS(df):
    income = df[df['type'] == 'Income'['amount'].sum()]
    expense = df[df['type'] == 'Expense'['amount'].sum()]
    savings = income-expense
    savingsRate = round((savings/income)*100,2) if income else 0
    return{ "income":income,"expense":expense,"savings":savings,"savings rate":savingsRate}

def monthlySpending(df):
    df_exp = df[df['type']== 'Expense']
    return df_exp.set_index('date').amount.resample("m").sum()

def calculateFHS(income,expense, savingRate):
    if income == 0:
        return 0
    score = 0
    if savingRate > 20:
        score += 40
    elif savingRate >10:
        score+=30 
    else:
        score += 10
    if expense < income *0.7:
        score +=30
    elif expense < income * 0.9:
        score += 20
    else: 
        score +=10
    if income > 0:
        score += 30
    return score

def addIEC(df):
    df['income_amt'] = df.apply(lambda x: x['amount'] if x['type'] == 'Income' else 0, axis=1)
    df['expense_amt'] = df.apply(lambda x: x['amount'] if x['type'] == 'Expense' else 0, axis=1)
    return df

def monthly_aggregation(df):
    df = addIEC(df)
    monthly = df.set_index('date').resample('M').sum()
    monthly['savings'] = monthly['income_amt'] - monthly['expense_amt']
    monthly['savings_rate'] = monthly['savings'] / monthly['income_amt']
    return monthly

def addRF(df, window=3):
    df['rolling_expense_avg'] = df['expense_amt'].rolling(window=window).mean()
    df['rolling_income_avg'] = df['income_amt'].rolling(window=window).mean()
    df['rolling_savings_rate'] = df['savings_rate'].rolling(window=window).mean()
    return df.dropna()

def anomalyFeatures(df):
    dfClean = df[['amount']].copy()
    scaler = MinMaxScaler()
    dfClean["amount_scaled"] = scaler.fit_transform(dfClean[['amount']])
    return dfClean

def forecast(df,months=3):
    monthly = monthly_aggregation(df)
    expense = monthly['expense_amt']
    recentAVG = expense[-3:].mean()
    lastDate = expense.index[-1]
    futureDates = pd.date_range(start=lastDate + pd.DateOffset(months = 1))
    forecast = pd.Series([recentAVG]*months, index = futureDates)
    plt.figure(figsize=(10,4))
    plt.plot(expense, label="Historical Expenses")
    plt.plot(forecast, label = "Forecasted Expenses", color = "red", linestyle = "--")
    plt.title("Expense Forecast")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return forecast

def dectectAnomalies(df):
    features = anomalyFeatures(df)
    model = IsolationForest(contamination = 0.02, random_state= 42)
    df["anomaly"] = model.fit_predict(features[["amount_scaled"]])
    df["anomaly"] = df["anomaly"].map({1:0,-1:1})
    return df

def segmentUsers(df,n_clusters=3):
    monthly = monthly_aggregation(df)[["income_amt","expense_amt","savings"]].dropna()
    scaler = MinMaxScaler
    scaled = scaler.fit_transform(monthly)
    k = KMeans(n_clusters=n_clusters,random_state=42)
    monthly["cluster"] = KMeans.fit_predict(scaled)
    return monthly

def aiAdvice(prompt, token= "your HF API key"):
    url = "https://huggingface.co/docs/inference-providers/en/index"
    headers = {"Authorization":f"Bearer {token}"}
    payload = {"inputs": prompt}
    response = requests.post(url,headers=headers,json=payload)
    return response.json()[0]["generated_text"]

def prompt(income, expense):
    saving = income - expense
    savinGRate = round((saving/income)*100,2) if income else 0
    return f"""My monthly income is ${income} and I spend around ${expense}.
    I save ${saving} monthly, which is a {savinGRate}% saving rate.
    Give me smart financial advice to improve my savings and investments."""





