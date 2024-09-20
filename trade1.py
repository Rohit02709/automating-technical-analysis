from app.data_sourcing import Data_Sourcing, data_update
from app.indicator_analysis import Indications
from app.graph import Visualization
from tensorflow.keras.models import load_model
import streamlit as st 
import gc
import nselib  # Ensure this is installed

gc.collect()

def main(app_data):
    st.set_page_config(layout="wide")
    indication = 'Predicted'

    st.sidebar.subheader('Asset:')
    asset_options = sorted(['Indexes', 'Stocks', 'Futures', 'Options'])
    asset = st.sidebar.selectbox('', asset_options, index=1)

    exchange = 'NSE India'
    app_data.exchange_data(exchange)

    if asset == 'Stocks':
        # Fetch all NSE stocks
        assets = nselib.get_all_stocks()  # Adjust according to nselib function
        st.sidebar.subheader('NSE Stocks:')
        equity = st.sidebar.selectbox('', assets)

    elif asset == 'Indexes':
        # Fetch all NSE indexes
        index_options = nselib.get_indexes()  # Fetch available indexes
        market = st.sidebar.selectbox('Select Index:', index_options)
        assets = nselib.get_index_stocks(market)
        st.sidebar.subheader(f'{market} Stocks:')
        equity = st.sidebar.selectbox('', assets)

    elif asset == 'Futures':
        assets = nselib.get_futures()  # Fetch available futures
        st.sidebar.subheader('Futures:')
        equity = st.sidebar.selectbox('', assets)

    elif asset == 'Options':
        assets = nselib.get_options()  # Fetch available options
        st.sidebar.subheader('Options:')
        equity = st.sidebar.selectbox('', assets)

    # Interval selection remains unchanged
    st.sidebar.subheader('Interval:')
    interval = st.sidebar.selectbox('', ('5 Minute', '15 Minute', '30 Minute', '1 Hour', '1 Day', '1 Week'), index=4)
    volatility_index = 0     

    st.sidebar.subheader('Trading Volatility:')
    risk = st.sidebar.selectbox('', ('Low', 'Medium', 'High'), index=volatility_index)

    st.title(f'Automated Technical Analysis.')
    st.subheader(f'{asset} Data Sourced from {exchange}.')
    st.info(f'Predicting...')

    analysis = Visualization(exchange, interval, equity, indication, action_model, price_model)
    requested_date = analysis.df.index[-1]
    current_price = float(analysis.df['Adj Close'][-1])
    change = float(analysis.df['Adj Close'].pct_change()[-1]) * 100
    requested_prediction_price = float(analysis.requested_prediction_price)
    requested_prediction_action = analysis.requested_prediction_action

    risks = {
        'Low': [analysis_day.df['S1'].values[-1], analysis_day.df['R1'].values[-1]], 
        'Medium': [analysis_day.df['S2'].values[-1], analysis_day.df['R2'].values[-1]],   
        'High': [analysis_day.df['S3'].values[-1], analysis_day.df['R3'].values[-1]],
    }
    buy_price = float(risks[risk][0])
    sell_price = float(risks[risk][1])

    change_display = f'A **{change:.2f}%** gain' if change > 0 else f'A **{change:.2f}%** loss' if change < 0 else 'UNCH'

    current_price = f'{current_price:,.2f}'
    requested_prediction_price = f'{requested_prediction_price:,.2f}'
    buy_price = f'{buy_price:,.2f}'
    sell_price = f'{sell_price:,.2f}'

    present_statement_prefix = 'off from taking any action with' if requested_prediction_action == 'Hold' else ''
    present_statement_suffix = ' at this time' if requested_prediction_action == 'Hold' else ''
    
    accuracy_threshold = {analysis.score_action: 75., analysis.score_price: 75.}
    confidence = {score: f'*({score}% confident)*' if float(score) >= threshold else '' for score, threshold in accuracy_threshold.items()}

    forecast_prefix = int(interval.split()[0])
    forecast_suffix = str(interval.split()[1]).lower() + ('s' if forecast_prefix > 1 else '')

    asset_suffix = 'price'

    st.markdown(f'**Prediction Date & Time (UTC):** {str(requested_date)}.')
    st.markdown(f'**Current Price:** ₹ {current_price}.')
    st.markdown(f'**{interval} Price Change:** {change_display}.')
    st.markdown(f'**Recommended Trading Action:** You should **{requested_prediction_action.lower()}** {present_statement_prefix} this {asset.lower()[:6]}{present_statement_suffix}. {confidence[analysis.score_action]}')
    st.markdown(f'**Estimated Forecast Price:** The {asset.lower()[:6]} {asset_suffix} for **{equity}** is estimated to be **₹ {requested_prediction_price}** in the next **{forecast_prefix} {forecast_suffix}**. {confidence[analysis.score_price]}')
    
    if requested_prediction_action == 'Hold':
        st.markdown(f'**Recommended Trading Margins:** You should consider buying more **{equity}** {asset.lower()[:6]} at **₹ {buy_price}** and sell it at **₹ {sell_price}**.')

    prediction_fig = analysis.prediction_graph(asset)
    
    st.success(f'Historical {asset[:6]} Price Action.')
    st.plotly_chart(prediction_fig, use_container_width=True)

    technical_analysis_fig = analysis.technical_analysis_graph()
    st.plotly_chart(technical_analysis_fig, use_container_width=True) 

if __name__ == '__main__':
    import warnings
    import gc
    warnings.filterwarnings("ignore") 
    gc.collect()
    action_model = load_model("models/action_prediction_model.h5")
    price_model = load_model("models/price_prediction_model.h5")
    app_data = Data_Sourcing()
    main(app_data=app_data)
