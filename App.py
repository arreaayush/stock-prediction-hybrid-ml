import streamlit as st
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objs as go
from tensorflow.keras.models import load_model
import requests
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon
nltk.download('vader_lexicon')

# Custom CSS
st.markdown("""
    <style>
        .reportview-container .main .block-container {
            max-width: 90%;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .center-align {
            text-align: center;
        }
        .news-article {
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 10px;
            background-color: #f0f2f6;
        }
        .metric-box {
            padding: 1rem;
            border-radius: 10px;
            background-color: #f0f2f6;
            margin: 1rem 0;
        }
        .sentiment-positive { color: #4CAF50; }
        .sentiment-neutral { color: #FF9800; }
        .sentiment-negative { color: #F44336; }
    </style>
""", unsafe_allow_html=True)

# Cache functions
@st.cache_data
def fetch_data(api_key, symbol):
    ts = TimeSeries(key=api_key, output_format='pandas')
    data, _ = ts.get_daily(symbol=symbol, outputsize='full')
    data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    data.index = pd.to_datetime(data.index)
    data.sort_index(inplace=True)
    data['MA_50'] = data['Close'].rolling(50).mean()
    data['MA_200'] = data['Close'].rolling(200).mean()
    return data.dropna()

@st.cache_resource
def load_assets():
    scaler = joblib.load('stock_scaler.pkl')
    lstm_model = load_model('lstm_model.keras')
    rf_model = joblib.load('rf_model.pkl')
    # Replace these with your actual validation scores
    lstm_r2 = 0.93  # Example LSTM R² score
    rf_r2 = 0.89    # Example Random Forest R² score
    return scaler, lstm_model, rf_model, lstm_r2, rf_r2

@st.cache_data(ttl=3600)
def fetch_news(api_key, symbol):
    url = f"https://newsapi.org/v2/everything?q={symbol}&language=en&sortBy=publishedAt&apiKey={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        news_data = response.json()
        return news_data.get('articles', [])[:5]
    except Exception as e:
        st.error(f"News API Error: {str(e)}")
        return []

# App Interface
st.title("📈 Stock Prediction Web App")
st.markdown("<div class='center-align'>", unsafe_allow_html=True)

# User Inputs
api_key = st.secrets["AV_API_KEY"]
news_api_key = st.secrets["NEWS_API_KEY"]
symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, MSFT):", "AAPL").upper()

if st.button('Analyze') and symbol:
    try:
        # Load data and models
        data = fetch_data(api_key, symbol)
        scaler, lstm, rf, lstm_r2, rf_r2 = load_assets()

        # Display data
        st.subheader("📜 Historical Data (First 5 Days)")
        st.dataframe(data.head(5).style.format("{:.2f}"), height=150)
        st.subheader("📅 Recent Data (Latest 5 Days)")
        st.dataframe(data.tail(5).style.format("{:.2f}"), height=150)

        # Predictions
        st.subheader("🔮 Next Day Predictions")
        n_days = 60
        scaled_data = scaler.transform(data[['Open', 'High', 'Low', 'Close', 'Volume', 'MA_50', 'MA_200']])
        last_sequence = scaled_data[-n_days:]

        lstm_pred = lstm.predict(last_sequence[np.newaxis, ...])[0][0]
        rf_pred = rf.predict(last_sequence.reshape(1, -1))[0]

        dummy_row = np.zeros((1, 7))
        dummy_row[0, 3] = lstm_pred
        lstm_price = scaler.inverse_transform(dummy_row)[0, 3]
        hybrid_pred = (lstm_price + rf_pred) / 2

        # Display predictions
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**LSTM Prediction**\n\n${lstm_price:.2f}")
        with col2:
            st.success(f"**Random Forest Prediction**\n\n${rf_pred:.2f}")
        with col3:
            st.warning(f"**Hybrid Prediction**\n\n${hybrid_pred:.2f}")

        # Model accuracies
        st.subheader("📊 Model Performance (R² Scores)")
        hybrid_r2 = (lstm_r2 + rf_r2) / 2
        cols = st.columns(3)
        with cols[0]:
            st.markdown(f"""
            <div class="metric-box">
                <h4>LSTM</h4>
                <h2>{lstm_r2:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        with cols[1]:
            st.markdown(f"""
            <div class="metric-box">
                <h4>Random Forest</h4>
                <h2>{rf_r2:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        with cols[2]:
            st.markdown(f"""
            <div class="metric-box">
                <h4>Hybrid</h4>
                <h2>{hybrid_r2:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)

        # Visualizations
        #st.subheader("📈 Moving Averages Analysis")
        #fig_ma = go.Figure()
        #fig_ma.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close Price'))
        #fig_ma.add_trace(go.Scatter(x=data.index, y=data['MA_50'], name='50-Day MA'))
        #fig_ma.add_trace(go.Scatter(x=data.index, y=data['MA_200'], name='200-Day MA'))
        #fig_ma.update_layout(height=500, xaxis_rangeslider_visible=False)
        #st.plotly_chart(fig_ma, use_container_width=True)
#------------------
        # 3. Technical Indicators Visualization
        st.subheader("📈 Moving Averages Analysis (50Days & 200Days)")
        fig_ma = go.Figure()
        fig_ma.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close Price'))
        fig_ma.add_trace(go.Scatter(x=data.index, y=data['MA_50'], name='50-Day MA'))
        fig_ma.add_trace(go.Scatter(x=data.index, y=data['MA_200'], name='200-Day MA'))
        fig_ma.update_layout(
            height=500,
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig_ma, use_container_width=True)

        # 4. Actual vs Predicted Visualization
        st.subheader("🔄 Actual vs Predicted Values")
        actual_prices = data['Close'].values[-n_days:]
        next_day = data.index[-1] + pd.offsets.BDay(1)

        fig_compare = go.Figure()
        fig_compare.add_trace(go.Scatter(
            x=data.index[-n_days:], 
            y=actual_prices, 
            name='Actual Prices',
            line=dict(color='blue', width=2)
        ))
        fig_compare.add_trace(go.Scatter(
            x=[next_day], 
            y=[lstm_price], 
            name='LSTM Prediction',
            mode='markers',
            marker=dict(color='green', size=12)
        ))
        fig_compare.add_trace(go.Scatter(
            x=[next_day], 
            y=[rf_pred], 
            name='RF Prediction',
            mode='markers',
            marker=dict(color='orange', size=12)
        ))
        fig_compare.add_trace(go.Scatter(
            x=[next_day], 
            y=[hybrid_pred], 
            name='Hybrid Prediction',
            mode='markers',
            marker=dict(color='purple', size=12)
        ))
        fig_compare.update_layout(
            height=500,
            showlegend=True,
            xaxis_title='Date',
            yaxis_title='Price'
        )
        st.plotly_chart(fig_compare, use_container_width=True)







#-------------
        # News and Sentiment Analysis
        st.subheader("📰 Latest News & Market Sentiment")
        news_articles = fetch_news(news_api_key, symbol)
        sia = SentimentIntensityAnalyzer()
        sentiment_scores = []

        if news_articles:
            for article in news_articles:
                text = f"{article.get('title', '')} {article.get('description', '')}"
                sentiment = sia.polarity_scores(text)
                sentiment_scores.append(sentiment['compound'])
                
                # Determine sentiment color
                sentiment_class = "sentiment-neutral"
                if sentiment['compound'] >= 0.05:
                    sentiment_class = "sentiment-positive"
                elif sentiment['compound'] <= -0.05:
                    sentiment_class = "sentiment-negative"

                # Display article
                publish_date = pd.to_datetime(article['publishedAt']).strftime('%b %d, %Y')
                st.markdown(f"""
                <div class="news-article">
                    <div class="{sentiment_class}" style="float: right; font-weight: bold;">
                        {sentiment['compound']:.2f}
                    </div>
                    <h4>{article['title']}</h4>
                    <p><b>{article['source']['name']}</b> | {publish_date}</p>
                    <p>{article['description']}</p>
                    <a href="{article['url']}" target="_blank">Read full article →</a>
                </div>
                """, unsafe_allow_html=True)

            # Overall sentiment
            avg_sentiment = np.mean(sentiment_scores)
            sentiment_label = "Neutral"
            sentiment_color = "#FF9800"
            if avg_sentiment >= 0.05:
                sentiment_label = "Positive"
                sentiment_color = "#4CAF50"
            elif avg_sentiment <= -0.05:
                sentiment_label = "Negative"
                sentiment_color = "#F44336"

            st.subheader("Overall Market Sentiment")
            st.markdown(f"""
            <div style="text-align: center; padding: 2rem;">
                <div style="font-size: 3rem; color: {sentiment_color};">
                    {avg_sentiment:.2f}
                </div>
                <div style="font-size: 1.5rem; color: {sentiment_color};">
                    {sentiment_label} Sentiment
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("No recent news articles found for this stock")

    except Exception as e:
        st.error(f"Error processing {symbol}: {str(e)}")

st.markdown("</div>", unsafe_allow_html=True)
