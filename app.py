import streamlit as st
from textblob import TextBlob
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


# Page config
st.set_page_config(
    page_title="Advanced Sentiment & Stock Analyzer",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Advanced Sentiment & Stock Analyzer")

# Sidebar for navigation
st.sidebar.title("Navigation")
analysis_type = st.sidebar.selectbox(
    "Choose Analysis Type:",
    ["Single Text Analysis", "Batch Analysis", "Stock + Sentiment Correlation", "Historical Analysis"]
)

def analyze_sentiment(text):
    """Analyze sentiment with enhanced categorization"""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    if polarity > 0.5:
        sentiment_label = "Very Positive"
        color = "green"
    elif polarity > 0.1:
        sentiment_label = "Positive"
        color = "lightgreen"
    elif polarity > -0.1:
        sentiment_label = "Neutral"
        color = "gray"
    elif polarity > -0.5:
        sentiment_label = "Negative"
        color = "orange"
    else:
        sentiment_label = "Very Negative"
        color = "red"
    
    return {
        'polarity': polarity,
        'subjectivity': subjectivity,
        'label': sentiment_label,
        'color': color
    }

def get_stock_data(ticker, period="1mo"):
    """Get stock data with error handling"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        info = stock.info
        return hist, info
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None, None

def calculate_stock_metrics(hist_data):
    """Calculate additional stock metrics"""
    if hist_data is None or hist_data.empty:
        return None
    
    current_price = hist_data['Close'].iloc[-1]
    previous_price = hist_data['Close'].iloc[-2] if len(hist_data) > 1 else current_price
    change = current_price - previous_price
    change_percent = (change / previous_price) * 100
    
    return {
        'current_price': current_price,
        'change': change,
        'change_percent': change_percent,
        'high_52w': hist_data['High'].max(),
        'low_52w': hist_data['Low'].min(),
        'avg_volume': hist_data['Volume'].mean()
    }

# SINGLE TEXT ANALYSIS
if analysis_type == "Single Text Analysis":
    st.header("🔍 Single Text Sentiment Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_input = st.text_area("Enter text to analyze:", height=100)
        
        if user_input:
            sentiment = analyze_sentiment(user_input)
            
            st.markdown(f"### Sentiment: <span style='color:{sentiment['color']}'>{sentiment['label']}</span>", unsafe_allow_html=True)
            
            # Create gauge chart for sentiment
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = sentiment['polarity'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Sentiment Score"},
                delta = {'reference': 0},
                gauge = {
                    'axis': {'range': [-1, 1]},
                    'bar': {'color': sentiment['color']},
                    'steps': [
                        {'range': [-1, -0.5], 'color': "red"},
                        {'range': [-0.5, 0], 'color': "orange"},
                        {'range': [0, 0.5], 'color': "lightgreen"},
                        {'range': [0.5, 1], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 0
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if user_input:
            st.metric("Polarity", f"{sentiment['polarity']:.3f}")
            st.metric("Subjectivity", f"{sentiment['subjectivity']:.3f}")
            
            # Interpretation
            st.write("**Interpretation:**")
            if sentiment['subjectivity'] > 0.5:
                st.write("📝 Subjective (opinion-based)")
            else:
                st.write("📊 Objective (fact-based)")

# BATCH ANALYSIS
elif analysis_type == "Batch Analysis":
    st.header("📊 Batch Sentiment Analysis")
    
    batch_input = st.text_area("Enter multiple texts (one per line):", height=200)
    
    if batch_input and st.button("Analyze All"):
        texts = [text.strip() for text in batch_input.split('\n') if text.strip()]
        
        results = []
        for i, text in enumerate(texts):
            sentiment = analyze_sentiment(text)
            results.append({
                'Text': text[:50] + "..." if len(text) > 50 else text,
                'Sentiment': sentiment['label'],
                'Polarity': sentiment['polarity'],
                'Subjectivity': sentiment['subjectivity']
            })
        
        df = pd.DataFrame(results)
        st.dataframe(df)
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            avg_polarity = df['Polarity'].mean()
            st.metric("Average Polarity", f"{avg_polarity:.3f}")
        with col2:
            positive_count = len(df[df['Polarity'] > 0])
            st.metric("Positive Texts", positive_count)
        with col3:
            negative_count = len(df[df['Polarity'] < 0])
            st.metric("Negative Texts", negative_count)
        
        # Visualization
        fig = px.histogram(df, x='Sentiment', title="Sentiment Distribution")
        st.plotly_chart(fig, use_container_width=True)

# STOCK + SENTIMENT CORRELATION
elif analysis_type == "Stock + Sentiment Correlation":
    st.header("📈 Stock & Sentiment Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        ticker = st.text_input("Enter stock ticker (e.g., AAPL, TSLA):").upper()
        period = st.selectbox("Time Period:", ["1d", "5d", "1mo", "3mo", "6mo", "1y"])
        
    with col2:
        sentiment_text = st.text_area("Enter sentiment text about this stock:", height=100)
    
    if ticker:
        hist_data, stock_info = get_stock_data(ticker, period)
        
        if hist_data is not None:
            metrics = calculate_stock_metrics(hist_data)
            
            # Stock metrics display
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Price", f"${metrics['current_price']:.2f}",
                         f"{metrics['change']:+.2f} ({metrics['change_percent']:+.1f}%)")
            with col2:
                st.metric("52W High", f"${metrics['high_52w']:.2f}")
            with col3:
                st.metric("52W Low", f"${metrics['low_52w']:.2f}")
            with col4:
                st.metric("Avg Volume", f"{metrics['avg_volume']:,.0f}")
            
            # Stock chart
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=('Stock Price', 'Volume'),
                row_width=[0.7, 0.3]
            )
            
            fig.add_trace(go.Candlestick(
                x=hist_data.index,
                open=hist_data['Open'],
                high=hist_data['High'],
                low=hist_data['Low'],
                close=hist_data['Close'],
                name="Price"
            ), row=1, col=1)
            
            fig.add_trace(go.Bar(
                x=hist_data.index,
                y=hist_data['Volume'],
                name="Volume",
                marker_color='blue',
                opacity=0.6
            ), row=2, col=1)
            
            fig.update_layout(
                title=f"{ticker} Stock Analysis",
                xaxis_rangeslider_visible=False,
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Sentiment analysis if provided
            if sentiment_text:
                sentiment = analyze_sentiment(sentiment_text)
                
                st.subheader("Sentiment Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Sentiment:** <span style='color:{sentiment['color']}'>{sentiment['label']}</span>",
                               unsafe_allow_html=True)
                    st.write(f"**Polarity:** {sentiment['polarity']:.3f}")
                    st.write(f"**Subjectivity:** {sentiment['subjectivity']:.3f}")
                
                with col2:
                    # Correlation insight
                    price_trend = "📈 Rising" if metrics['change'] > 0 else "📉 Falling" if metrics['change'] < 0 else "➡️ Stable"
                    
                    st.write("### Quick Insight")
                    st.write(f"**Stock Trend:** {price_trend}")
                    st.write(f"**Sentiment:** {sentiment['label']}")
                    
                    if (metrics['change'] > 0 and sentiment['polarity'] > 0) or (metrics['change'] < 0 and sentiment['polarity'] < 0):
                        st.success("✅ Sentiment aligns with stock movement")
                    else:
                        st.warning("⚠️ Sentiment contrasts with stock movement")

# HISTORICAL ANALYSIS
elif analysis_type == "Historical Analysis":
    st.header("📅 Historical Analysis")
    
    ticker = st.text_input("Enter stock ticker:").upper()
    
    if ticker:
        # Get longer historical data
        hist_data, _ = get_stock_data(ticker, "1y")
        
        if hist_data is not None:
            # Price trend analysis
            hist_data['Returns'] = hist_data['Close'].pct_change()
            hist_data['MA20'] = hist_data['Close'].rolling(window=20).mean()
            hist_data['MA50'] = hist_data['Close'].rolling(window=50).mean()
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=hist_data.index,
                y=hist_data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                x=hist_data.index,
                y=hist_data['MA20'],
                mode='lines',
                name='20-day MA',
                line=dict(color='orange')
            ))
            
            fig.add_trace(go.Scatter(
                x=hist_data.index,
                y=hist_data['MA50'],
                mode='lines',
                name='50-day MA',
                line=dict(color='red')
            ))
            
            fig.update_layout(
                title=f"{ticker} - 1 Year Historical Analysis",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                volatility = hist_data['Returns'].std() * 100
                st.metric("Volatility", f"{volatility:.2f}%")
            
            with col2:
                avg_return = hist_data['Returns'].mean() * 100
                st.metric("Avg Daily Return", f"{avg_return:.3f}%")
            
            with col3:
                total_return = ((hist_data['Close'].iloc[-1] / hist_data['Close'].iloc[0]) - 1) * 100
                st.metric("Total Return (1Y)", f"{total_return:.1f}%")

# Footer
st.markdown("---")
st.markdown("### 💡 Tips for Better Analysis")
st.markdown("""
- **Sentiment Analysis:** More text generally provides better accuracy
- **Stock Analysis:** Consider multiple timeframes for comprehensive view
- **Correlation:** Remember that sentiment is just one factor affecting stock prices
- **Historical Data:** Look for patterns but remember past performance doesn't guarantee future results
""")

st.markdown("**Disclaimer:** This tool is for educational purposes only. Not financial advice.")
