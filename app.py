import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation
from prophet.plot import plot_cross_validation_metric
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.plot import add_changepoints_to_plot
from prophet.plot import plot_yearly
import base64

st.title('Автоматизований прогноз числових показників')

"""
### Крок 1: Імпорт набору даних
"""
df = st.file_uploader('Завантажте ваш набір даних тут. Датасет повинен містити дату записів і бажаний показник. Колонка з датою повинна мати назву "ds" та відповідати формату: РРРР-ММ-ДД (Наприклад: 2019-05-20). Колонка з показником повинна називатися "y" і представляти числове значення, яке ви хочете прогнозувати. Допустимий формат файлу: csv.', type='csv')

st.info(
            f"""
                👆 Завантажуй файл csv. [Приклад](https://raw.githubusercontent.com/BohdanTarchanin/healthcare-metrics-prediction/master/example_pedestrians_covid.csv)
                """
        )

if df is not None:
    data = pd.read_csv(df)
    data['ds'] = pd.to_datetime(data['ds'], errors='coerce') 
    st.write(data)
    
    max_date = data['ds'].max()
    st.write(max_date)
    st.success('Це остання дата у вашому наборі даних')

"""
### Крок 2: Вказіть тривалість прогнозу
"""

periods_input = st.number_input('На скільки днів ви хотіли б прогноз? Введіть число від 1 до 365 і натисніть Enter.',
min_value=1, max_value=365)

if df is not None and periods_input > 0:
    m = Prophet()
    m.fit(data)

    """
    ### Крок 3: Візуалізація прогнозованих даних
    """
    future = m.make_future_dataframe(periods=periods_input)
    forecast = m.predict(future)
    fcst = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    fcst_filtered = fcst[fcst['ds'] > max_date]    
    st.write(fcst_filtered)
    
    fig1 = m.plot(forecast)
    st.write(fig1)

    fig2 = m.plot_components(forecast)
    st.write(fig2)
        
    fig3 = plot_plotly(m, forecast)
    st.write(fig3)

    fig4 = plot_components_plotly(m, forecast)
    st.write(fig4)

"""
### Крок 4: Завантажте ваш прогноз
"""
if df is not None:
    csv_exp = fcst_filtered.to_csv(index=False)
    b64 = base64.b64encode(csv_exp.encode()).decode()  # Кодування CSV
    st.download_button(label="Завантажити прогноз", data=csv_exp, file_name='forecast.csv', mime='text/csv')
