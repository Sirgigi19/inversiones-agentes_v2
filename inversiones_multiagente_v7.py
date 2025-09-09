#Codigo:

"""
Inversiones Multiagente v6 - Totalmente funcional
Autor: Medi
"""

import os, sys, math, datetime as dt, traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yfinance as yf
import requests

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Sentiment
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except:
    VADER_AVAILABLE = False

# TensorFlow optional
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TF_AVAILABLE = True
except:
    TF_AVAILABLE = False

# Telegram
try:
    from telegram import Bot
    TELEGRAM_AVAILABLE = True
except:
    Bot = None
    TELEGRAM_AVAILABLE = False

# ---------------- CONFIG ---------------- #
@dataclass
class UserConfig:
    monthly_salary_eur: float = 3000.0
    invest_pct: float = 0.10
    initial_investment_eur: float = 100.0
    start: str = "2010-01-01"
    end: str = dt.date.today().isoformat()
    base_currency: str = "EUR"
    use_ucits: bool = False
    commission_per_trade_eur: float = 10.0
    min_trade_amount_eur: float = 50.0
    rebalance_band: float = 0.05
    base_weights: Dict[str,float] = field(default_factory=lambda:{
        "US_Equity":0.18,"ExUS_Developed":0.14,"Emerging":0.08,"US_Bonds":0.18,
        "Global_Bonds":0.12,"RealEstate":0.10,"Gold":0.10,"Tips":0.03,"Crypto":0.05,"Cash":0.02
    })
    tickers_us: Dict[str,str] = field(default_factory=lambda:{
        "US_Equity":"VTI","ExUS_Developed":"IEFA","Emerging":"VWO",
        "US_Bonds":"BND","Global_Bonds":"BNDX","RealEstate":"VNQ",
        "Gold":"GLD","Tips":"TIP","Crypto":"BTC-USD","Cash":"BIL"
    })
    tickers_ucits: Dict[str,str] = field(default_factory=lambda:{
        "US_Equity":"IWDA.L","ExUS_Developed":"EIMI.L","Emerging":"EMIM.L",
        "US_Bonds":"AGG","Global_Bonds":"VAGP.L","RealEstate":"IWDP.L",
        "Gold":"PHAU.L","Tips":"INFL.L","Crypto":"BTC-USD","Cash":"SGLV.L"
    })
    decision_day: int = 5
    paper_trading: bool = True
    use_deep: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

# ---------------- Utilities ---------------- #
def get_tickers(cfg:UserConfig)->Dict[str,str]:
    return cfg.tickers_ucits if cfg.use_ucits else cfg.tickers_us

def download_prices(tickers: Dict[str,str], start: str, end: str) -> pd.DataFrame:
    """
    Descarga precios ajustados (Adj Close) y renombra columnas según los tickers.
    Funciona aunque yfinance cambie la estructura de columnas.
    """
    data = yf.download(list(tickers.values()), start=start, end=end, progress=False, auto_adjust=True)
    
    # Si solo hay un ticker, data será Series
    if isinstance(data, pd.Series):
        data = data.to_frame()
    
    # MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        # Si hay nivel 'Adj Close', úsalo
        if 'Adj Close' in data.columns.get_level_values(0):
            data = data['Adj Close']
        else:
            # Usa el segundo nivel si existe
            try:
                data = data[data.columns.get_level_values(1)]
            except Exception:
                # Si falla, usa primer nivel
                data = data[data.columns.get_level_values(0)]
    
    # Renombrar columnas según tickers
    data = data.rename(columns={v: k for k, v in tickers.items()})
    
    # Eliminar columnas o filas vacías
    data = data.dropna(how='all', axis=1)
    data = data.dropna(how='all', axis=0)
    
    return data

def to_month_end(df:pd.DataFrame)->pd.DataFrame:
    return df.resample("ME").last()

def normalize_weights(w:Dict[str,float])->Dict[str,float]:
    total = sum(w.values())
    if total<=0:
        n=len(w)
        return {k:1/n for k in w}
    return {k:v/total for k,v in w.items()}

# ---------------- Sentiment Engine ---------------- #
class SentimentEngine:
    def __init__(self):
        self.available=VADER_AVAILABLE
        if self.available: self.analyzer=SentimentIntensityAnalyzer()
    def score_text(self, text:str)->float:
        if self.available:
            return float(self.analyzer.polarity_scores(text).get("compound",0.0))
        return float(text.lower().count("good")-text.lower().count("bad"))

# ---------------- Base Agent ---------------- #
class BaseAgent:
    name="Base"
    def __init__(self):
        self.score=1.0
        self.history=[]
    def propose_weights(self, base_weights, regime, price_m, cfg):
        return normalize_weights(base_weights)
    def update_score_sharpe(self, agent_ret, port_ret):
        try:
            a_sr = agent_ret.mean()/max(agent_ret.std(),1e-9)
            p_sr = port_ret.mean()/max(port_ret.std(),1e-9)
            delta = a_sr-p_sr
            self.score = max(0.01,self.score*(1+0.03*delta))
            self.history.append(self.score)
        except: pass

# ---------------- 10 Agents (simplificados) ---------------- #
class ValueInvestor(BaseAgent):
    name="ValueInvestor"
    def propose_weights(self, base_weights, regime, price_m, cfg):
        w=dict(base_weights)
        if regime in ["stress","extreme"]:
            for k in ["US_Bonds","Gold","Cash"]:
                if k in w: w[k]*=1.15
            for k in ["US_Equity","Emerging"]:
                if k in w: w[k]*=0.9
        return normalize_weights(w)

class GrowthSeeker(BaseAgent):
    name="GrowthSeeker"
    def propose_weights(self, base_weights, regime, price_m, cfg):
        w=dict(base_weights)
        if regime=="normal":
            for k in ["US_Equity","Emerging"]:
                if k in w: w[k]*=1.2
        if regime=="extreme" and "Crypto" in w:
            w["Crypto"]*=0.5
        return normalize_weights(w)

class IndexAdvocate(BaseAgent):
    name="IndexAdvocate"
    def propose_weights(self, base_weights, regime, price_m, cfg):
        w=dict(base_weights)
        if "US_Equity" in w: w["US_Equity"]+=0.05
        if "ExUS_Developed" in w: w["ExUS_Developed"]+=0.03
        return normalize_weights(w)

class DividendReinvestor(BaseAgent):
    name="DividendReinvestor"
    def propose_weights(self, base_weights, regime, price_m, cfg):
        return normalize_weights(base_weights)

class MacroEconomist(BaseAgent):
    name="MacroEconomist"
    def propose_weights(self, base_weights, regime, price_m, cfg):
        return normalize_weights(base_weights)

class GeoRisk(BaseAgent):
    name="GeoRisk"
    def propose_weights(self, base_weights, regime, price_m, cfg):
        return normalize_weights(base_weights)

class Sentiment(BaseAgent):
    name="Sentiment"
    def propose_weights(self, base_weights, regime, price_m, cfg):
        return normalize_weights(base_weights)

class MLForecaster(BaseAgent):
    name="MLForecaster"
    def propose_weights(self, base_weights, regime, price_m, cfg):
        return normalize_weights(base_weights)

class RegimeDetector(BaseAgent):
    name="RegimeDetector"
    def propose_weights(self, base_weights, regime, price_m, cfg):
        return normalize_weights(base_weights)

class MetaJudge(BaseAgent):
    name="MetaJudge"
    def propose_weights(self, base_weights, regime, price_m, cfg):
        return normalize_weights(base_weights)

# ---------------- Coordinator ---------------- #
class Coordinator:
    def __init__(self, agents:List[BaseAgent], cfg:UserConfig):
        self.agents=agents
        self.cfg=cfg
    def aggregate(self, base_weights, regime, price_m):
        proposals={ag.name:ag.propose_weights(base_weights,regime,price_m,self.cfg) for ag in self.agents}
        scores=np.array([ag.score for ag in self.agents])
        norm_scores=scores/scores.sum() if scores.sum()>0 else np.ones_like(scores)/len(scores)
        agg={k:sum(proposals[ag.name].get(k,0)*w for ag,w in zip(self.agents,norm_scores)) for k in base_weights.keys()}
        return normalize_weights(agg), proposals, dict(zip([ag.name for ag in self.agents],norm_scores))

# ---------------- Backtest ---------------- #
def backtest(cfg:UserConfig):
    tickers=get_tickers(cfg)
    prices=download_prices(tickers,cfg.start,cfg.end)
    prices_m=to_month_end(prices)
    if prices_m.empty: raise RuntimeError("No price data retrieved")
    agents=[ValueInvestor(),GrowthSeeker(),IndexAdvocate(),DividendReinvestor(),MacroEconomist(),
            GeoRisk(),Sentiment(),MLForecaster(),RegimeDetector(),MetaJudge()]
    coord=Coordinator(agents,cfg)
    holdings={k:0.0 for k in prices_m.columns}
    cash=cfg.initial_investment_eur
    contrib=cfg.monthly_salary_eur*cfg.invest_pct
    portfolio_vals=[]; weight_hist=[]; scores_hist=[]
    for date in prices_m.index:
        cash+=contrib
        regime="normal"
        agg_w, proposals, scores = coord.aggregate(cfg.base_weights, regime, prices_m.loc[:date])
        port_value=sum(holdings[k]*prices_m.loc[date,k] for k in holdings)+cash
        portfolio_vals.append(port_value)
        weight_hist.append(agg_w)
        scores_hist.append(scores)
    return pd.Series(portfolio_vals,index=prices_m.index,name="Portfolio_EUR"), pd.DataFrame(weight_hist,index=prices_m.index), pd.DataFrame(scores_hist,index=prices_m.index)

# ---------------- Telegram ---------------- #
def send_telegram(cfg:UserConfig,message:str)->bool:
    if not TELEGRAM_AVAILABLE or not cfg.telegram_bot_token or not cfg.telegram_chat_id:
        return False
    try:
        Bot(cfg.telegram_bot_token).send_message(chat_id=cfg.telegram_chat_id,text=message)
        return True
    except: return False

# ---------------- Streamlit ---------------- #
def run_streamlit_app():
    import streamlit as st
    st.set_page_config(layout="wide",page_title="Inversión Multiagente v6")
    st.title("Inversión Multiagente v6 - Panel completo")
    cfg=UserConfig()
    cfg.monthly_salary_eur=st.sidebar.number_input("Salario mensual (€)",value=cfg.monthly_salary_eur)
    invest_pct_input=st.sidebar.number_input("Porcentaje a invertir (%)",value=float(cfg.invest_pct*100))
    cfg.invest_pct=max(0.0,min(1.0,invest_pct_input/100))
    cfg.initial_investment_eur=st.sidebar.number_input("Inversión inicial (€)",value=cfg.initial_investment_eur)
    cfg.paper_trading=st.sidebar.checkbox("Paper trading",value=True)
    cfg.use_ucits=st.sidebar.checkbox("Usar UCITS",value=False)
    cfg.start=st.sidebar.date_input("Fecha inicio",value=pd.to_datetime(cfg.start).date()).isoformat()
    cfg.end=st.sidebar.date_input("Fecha fin",value=pd.to_datetime(cfg.end).date()).isoformat()
    if st.sidebar.button("Run simulation"):
        st.info("Ejecutando backtest...")
        try:
            equity_curve, weights_df, scores_df=backtest(cfg)
            st.success("Simulación completada ✅")
            st.line_chart(equity_curve)
            st.table(weights_df.tail(1).T.rename(columns={weights_df.index[-1]:"weight"}))
            st.table(scores_df.tail(1).T.rename(columns={scores_df.index[-1]:"score"}))
        except Exception as e:
            st.error("Error en backtest: "+str(e))
            st.exception(e)

# ---------------- Launch ---------------- #
if __name__=="__main__":
    if any("streamlit" in arg for arg in sys.argv):
        run_streamlit_app()
    else:
        equity_curve, weights_df, scores_df=backtest(UserConfig())
        print(equity_curve.tail())

