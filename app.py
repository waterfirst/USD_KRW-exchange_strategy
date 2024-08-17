import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from scipy import stats
from scipy.optimize import minimize
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
from fredapi import Fred

# 경고 메시지 무시
warnings.filterwarnings("ignore")

# FRED API 키 설정 (실제 사용 시 본인의 API 키로 대체해야 합니다)

fred = Fred(api_key=st.secrets["8c945b914a00faf01f18cdc2e1368861"])


# 페이지 설정
st.set_page_config(page_title="환율 페어트레이딩 투자 전략", layout="wide")

# CSS 스타일
st.markdown("""
<style>
    .reportview-container .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 5px;
        padding-right: 5px;
    }
    .stPlotlyChart {
        height: 70vh !important;
    }
</style>
""", unsafe_allow_html=True)

# 데이터 로드 및 전처리 함수
@st.cache_data
def load_and_process_data(start_date, end_date):
    tickers = {
        "USD/KRW": "KRW=X",
        "USD/JPY": "JPY=X",
        "S&P500": "^GSPC",
        "US_IR": "^TNX",  # 10-year Treasury Yield
    }
    data = yf.download(list(tickers.values()), start=start_date, end=end_date)['Close']
    data.columns = tickers.keys()
    
    # Get CPI data from FRED
    cpi_data = fred.get_series('CPIAUCSL', start_date, end_date)
    cpi_data = cpi_data.reindex(data.index).fillna(method='ffill')
    
    # Calculate inflation rate
    data['US_Inflation'] = cpi_data.pct_change(12)  # 12-month inflation rate
    
    data['Returns'] = np.log(data['USD/KRW'] / data['USD/KRW'].shift(1))
    return data.dropna().replace([np.inf, -np.inf], np.nan).dropna()



# 시계열 그래프 생성 함수
def create_time_series_plot(data, title, y1_label, y2_label):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=data.index, y=data[y1_label], name=y1_label),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=data.index, y=data[y2_label], name=y2_label),
        secondary_y=True,
    )

    fig.update_layout(
        title_text=title,
        xaxis_title="Date"
    )

    fig.update_yaxes(title_text=y1_label, secondary_y=False)
    fig.update_yaxes(title_text=y2_label, secondary_y=True)

    return fig



# 복합 전략 함수
def combined_strategy(data, ma_short, ma_long, corr_window, corr_threshold, pair_window, pair_threshold):
    ma_data = moving_average_strategy(data.copy(), ma_short, ma_long)
    corr_data = correlation_strategy(data.copy(), corr_window, corr_threshold)
    pair_data = pair_trading_strategy(data.copy(), pair_window, pair_threshold)
    
    data['Combined_Signal'] = (ma_data['Signal'] + corr_data['Signal'] + pair_data['Signal']) / 3
    data['Buy_Signal'] = np.where(data['Combined_Signal'] > 0.66, 1, 0)
    data['Sell_Signal'] = np.where(data['Combined_Signal'] < -0.33, 1, 0)
    
    return data



# USD/KRW 그래프와 신호 생성 함수
def create_usd_krw_plot_with_signals(data):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=data.index, y=data['USD/KRW'], name='USD/KRW', line=dict(color='blue')))
    
    buy_signals = data[data['Buy_Signal'] == 1]
    sell_signals = data[data['Sell_Signal'] == 1]
    
    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['USD/KRW'], 
                             mode='markers', marker=dict(symbol='triangle-up', size=10, color='green'), 
                             name='매수 신호'))
    
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['USD/KRW'], 
                             mode='markers', marker=dict(symbol='triangle-down', size=10, color='red'), 
                             name='매도 신호'))
    
    fig.update_layout(title='USD/KRW 환율과 매수/매도 신호',
                      xaxis_title='날짜',
                      yaxis_title='USD/KRW 환율')
    
    return fig


# 이동평균 교차 전략 함수
def moving_average_strategy(data, short_window, long_window):
    data['SMA_short'] = data['USD/KRW'].rolling(window=short_window).mean()
    data['SMA_long'] = data['USD/KRW'].rolling(window=long_window).mean()
    data['Signal'] = np.where(data['SMA_short'] > data['SMA_long'], 1, 0)
    data['Strategy_Returns'] = data['Signal'].shift(1) * data['Returns']
    return data

# 상관관계 기반 전략 함수
def correlation_strategy(data, corr_window, threshold):
    data['Corr_SP500'] = data['USD/KRW'].rolling(window=corr_window).corr(data['S&P500'])
    data['Corr_US_IR'] = data['USD/KRW'].rolling(window=corr_window).corr(data['US_IR'])
    data['Signal'] = np.where((data['Corr_SP500'] > threshold) | (data['Corr_US_IR'] > threshold), 1, 0)
    data['Strategy_Returns'] = data['Signal'].shift(1) * data['Returns']
    return data

# 페어 트레이딩 전략 함수
def pair_trading_strategy(data, window, threshold):
    data['Ratio'] = data['USD/KRW'] / data['USD/JPY']
    data['Ratio_MA'] = data['Ratio'].rolling(window=window).mean()
    data['Ratio_Std'] = data['Ratio'].rolling(window=window).std()
    data['Z_Score'] = (data['Ratio'] - data['Ratio_MA']) / data['Ratio_Std']
    data['Signal'] = np.where(data['Z_Score'] > threshold, -1, np.where(data['Z_Score'] < -threshold, 1, 0))
    data['Strategy_Returns'] = data['Signal'].shift(1) * data['Returns']
    return data

# 그랜저 인과성 테스트 함수
def granger_causality(data, max_lag=5):
    variables = ['USD/KRW', 'S&P500', 'US_IR', 'USD/JPY']
    results = {}
    for v1 in variables:
        for v2 in variables:
            if v1 != v2:
                test_result = grangercausalitytests(data[[v1, v2]], maxlag=max_lag, verbose=False)
                min_p_value = min([test_result[i+1][0]['ssr_ftest'][1] for i in range(max_lag)])
                results[f"{v2} -> {v1}"] = min_p_value
    return results

# 시계열 그래프 생성 함수
def create_time_series_plot(data, title, y1_label, y2_label):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=data.index, y=data[y1_label], name=y1_label),
        secondary_y=False,
    )

    

    fig.add_trace(
        go.Scatter(x=data.index, y=data[y2_label], name=y2_label),
        secondary_y=True,
    )

    fig.update_layout(
        title_text=title,
        xaxis_title="Date"
    )

    fig.update_yaxes(title_text=y1_label, secondary_y=False)
    fig.update_yaxes(title_text=y2_label, secondary_y=True)

    return fig


# 메인 앱 함수
def main():
    st.title("USD/KRW 트레이딩 전략 비교 분석")
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("시작 날짜", value=datetime(2022, 1, 1))
    with col2:
        end_date = st.date_input("종료 날짜", value=datetime.now())
    
    data = load_and_process_data(start_date, end_date)
    
    if data.empty:
        st.warning("선택한 기간에 대한 데이터가 없습니다. 다른 기간을 선택해주세요.")
        return
    
    st.subheader("시계열 그래프")
    
    # 미국 이자율과 S&P 500 그래프
    fig1 = create_time_series_plot(data, "미국 이자율과 S&P 500", "US_IR", "S&P500")
    st.plotly_chart(fig1)

    # 미국 물가상승률과 이자율 그래프
    fig3 = create_time_series_plot(data, "미국 물가상승률과 이자율", "US_Inflation", "US_IR")
    st.plotly_chart(fig3)
    
    # USD/KRW와 USD/JPY 그래프
    fig2 = create_time_series_plot(data, "USD/KRW와 USD/JPY", "USD/KRW", "USD/JPY")
    st.plotly_chart(fig2)


     # 복합 전략 적용
    combined_data = combined_strategy(data.copy(), 20, 50, 30, 0.5, 20, 2)
    
    # USD/KRW 그래프와 신호 표시
    st.subheader("USD/KRW 환율과 매수/매도 신호")
    fig_signals = create_usd_krw_plot_with_signals(combined_data)
    st.plotly_chart(fig_signals)


    # 전략 비교
    st.subheader("전략 비교")
    
    ma_data = moving_average_strategy(data.copy(), 20, 50)
    corr_data = correlation_strategy(data.copy(), 30, 0.5)
    pair_data = pair_trading_strategy(data.copy(), 20, 2)
    
    strategies = {
        "이동평균 교차 전략": ma_data['Strategy_Returns'],
        "상관관계 기반 전략": corr_data['Strategy_Returns'],
        "USD/JPY 페어 트레이딩 전략": pair_data['Strategy_Returns']
    }
    
    cumulative_returns = pd.DataFrame({name: (1 + ret).cumprod() for name, ret in strategies.items()})
    cumulative_returns['Buy & Hold'] = (1 + data['Returns']).cumprod()
    
    fig = go.Figure()
    for col in cumulative_returns.columns:
        fig.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns[col], name=col))
    
    fig.update_layout(title="전략별 누적 수익률 비교", xaxis_title="날짜", yaxis_title="누적 수익률")
    st.plotly_chart(fig)
    
    # 성과 지표 계산
    performance = pd.DataFrame({
        "총 수익률": cumulative_returns.iloc[-1] - 1,
        "연간 수익률": (cumulative_returns.iloc[-1] ** (252 / len(cumulative_returns)) - 1),
        "샤프 비율": [np.sqrt(252) * ret.mean() / ret.std() for ret in strategies.values()] + [np.sqrt(252) * data['Returns'].mean() / data['Returns'].std()]
    })
    
    st.write(performance)

    st.subheader("결론")
    best_strategy = performance['총 수익률'].idxmax()
    st.write(f"분석 결과, {best_strategy}이(가) 가장 높은 수익률을 보였습니다.")
    
    st.write("그러나 다음 사항을 고려해야 합니다:")
    st.write("1. 과거 성과가 미래 성과를 보장하지 않습니다.")
    st.write("2. 각 전략의 리스크 프로필이 다를 수 있으므로, 단순 수익률 비교만으로는 충분하지 않을 수 있습니다.")
    st.write("3. 실제 거래에서는 거래 비용, 슬리피지 등이 발생하여 백테스트 결과와 차이가 날 수 있습니다.")
    st.write("4. 상관관계와 인과관계 분석 결과를 함께 고려하여 전략을 선택해야 합니다.")
    
    st.write("최종적인 전략 선택은 투자자의 리스크 성향, 투자 목적, 그리고 시장에 대한 견해를 종합적으로 고려하여 이루어져야 합니다.")

    st.subheader("추가 분석")
    st.subheader("데이터")
    st.dataframe(data)
    
    # 상관관계 분석
    st.subheader("상관관계 분석")
    correlation = data.corr()
    st.write(correlation)
    
    # 그랜저 인과성 테스트
    st.subheader("그랜저 인과성 테스트")
    granger_results = granger_causality(data)
    st.write(pd.DataFrame(granger_results, index=['p-value']).T)

if __name__ == "__main__":
    main()