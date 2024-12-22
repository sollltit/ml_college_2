import streamlit as st
from datetime import datetime, time
import pickle
import pandas as pd
import numpy as np
import catboost

# путь к модели
#model_path = 'model_cb2.pkl'
# модель
with open('model_cb2.pkl', 'rb') as file:
    model = pickle.load(file)

# загрузка site.pkl
with open('site.pkl', 'rb') as f:
    link_dict = pickle.load(f)


# текст
st.title('Определение мошенничества в интернете 🌐')

st.html("<p><span style='font-size: 22px;'>Поведение пользователей интернета разнообразно: они посещают самые разные сайты, проявляют активность в различное время суток. Кто-то может за день перейти на десятки сайтов и провести на каждом совсем мало времени, а кто-то будет подолгу сидеть на одном сайте и тщательно изучать информацию на нём. Такое разнообразие поведения в сети помогает идентифицировать человека, а так же даёт возможность определить, <i>является пользователь мошенником или нет</i>.</span></p>")
st.html("<p><span style='font-size: 20px; background-color: #3176A5;'>Сессия содержит историю посещения последних десяти сайтов. Введите информацию о ней, чтобы определить, принадлежит ли эта сессия мошеннику.</span></p>")
st.html("<p><span style='font-size: 20px;'>Если сессия содержит менее 10-ти сайтов, оставьте оставшиеся поля ввода пустыми.</span></p>")
st.markdown('────────────────────────────────────────────────────────────')
st.html("<p><tt><span style='font-size: 24px;'>Введите данные о сессии: </span></tt></p>")

# текущая дата
date_now = datetime.now().date()

# проверка что ссылка это ссылка
#def site_proverka(link):
#    if not (link.startswith("http://") or link.startswith("https://")): return st.error('Ошибка! Введите ссылку.')

# проверка что время введено правильно
def chek_time(time_s):
    try:
        # преобразование того что ввёл пользователь в тип данных datetime
        input_time = datetime.strptime(time_s, "%H:%M:%S").time()
        
        # проверка диапозона времнеи
        
        if input_time < time(0, 0, 0) or input_time > time(23, 59, 59):
            st.error('Ошибка! Время должно быть в диапазоне от 00:00:00 до 23:59:59')
        else: return True
    except ValueError:
        st.error('Ошибка! Введите время в формате "HH:MM:SS" (пример: 12:30:15). Время должно быть в диапазоне от 00:00:00 до 23:59:59')


# обработка ссылки
def link_id(url):
    if url != 0:
        if url in link_dict: return link_dict[url]
        # если ссылки в словаре нет, просто добавляем её туда
        new_id = int(max(link_dict.values()) + 1)
        link_dict[url] = new_id
        return new_id
    else: return url


# преобразовываем дф так, чтобы его могля принять модель
def mod_df(df_predict):
    col_site = ['site1', 'site2', 'site3', 'site4', 'site5', 
                'site6', 'site7', 'site8', 'site9', 'site10']
    times_col = ['time1', 'time2', 'time3', 'time4', 'time5',
                  'time6', 'time7', 'time8', 'time9', 'time10']
    df_predict[col_site] = df_predict[col_site].applymap(link_id) 
    # время
    for i in range(1, 10):
        time_col1 = f'time{i}'
        time_col2 = f'time{i+1}'
        # разница между посещением
        diff = pd.to_datetime(df_predict[time_col2]) - pd.to_datetime(df_predict[time_col1])
        diff_sec = diff.dt.total_seconds()

        # если пропуск, то заполняется нулём
        diff_sec[df_predict[time_col1].isna() | df_predict[time_col2].isna()] = 0
        # новая колонка
        df_predict.insert(df_predict.columns.get_loc(time_col1) + 1, f'time_diff{i}', diff_sec)

    df_predict[times_col] = df_predict[times_col].apply(pd.to_datetime)
    for col in times_col:
            df_predict.insert(df_predict.columns.get_loc(col) + 1, f'{col}_evening', (df_predict[col].dt.hour >= 18) & (df_predict[col].dt.hour < 24))
            df_predict.insert(df_predict.columns.get_loc(col) + 1, f'{col}_day', (df_predict[col].dt.hour >= 12) & (df_predict[col].dt.hour < 18))
            df_predict.insert(df_predict.columns.get_loc(col) + 1, f'{col}_morning', (df_predict[col].dt.hour >= 6) & (df_predict[col].dt.hour < 12))
            df_predict.insert(df_predict.columns.get_loc(col) + 1, f'{col}_night', (df_predict[col].dt.hour >= 0) & (df_predict[col].dt.hour < 6))
    
    # день недели
    for col in times_col:
        # "+ 1", чтобы отсчёт был с единицы (пн - 1, вт - 2 и тд) 
        df_predict.insert(df_predict.columns.get_loc(col) + 1, f'{col}_dayofweek', df_predict[col].dt.dayofweek + 1)

    # добавляем колонки с датой
    for col in times_col:
        # год*100+месяц, -> 2014*100+20 = 201402 будет в строке
        year_month = (df_predict[col].dt.year * 100 + df_predict[col].dt.month)
        df_predict.insert(df_predict.columns.get_loc(col) + 1, f'{col}_date', year_month)        

    df_predict['time1_date'] = df_predict['time1_date'].astype('int64')
    df_predict['time1_dayofweek'] = df_predict['time1_dayofweek'].astype('int64')
    int_col = ['time2_dayofweek', 'time3_dayofweek', 'time4_dayofweek','time5_dayofweek', 
               'time6_dayofweek', 'time7_dayofweek', 'time8_dayofweek', 'time9_dayofweek', 'time10_dayofweek',
               'time2_date', 'time3_date', 'time4_date', 'time5_date', 'time6_date', 'time7_date', 'time8_date',
               'time9_date', 'time10_date']
    
    df_predict[int_col] = df_predict[int_col].astype('int64')
    bool_cols = ['time1_night', 'time1_day', 'time1_morning', 'time1_evening',
           'time2_night', 'time2_day', 'time2_morning', 'time2_evening',
           'time3_night', 'time3_day', 'time3_morning', 'time3_evening',
           'time4_night', 'time4_day', 'time4_morning', 'time4_evening',
           'time5_night', 'time5_day', 'time5_morning', 'time5_evening',
           'time6_night', 'time6_day', 'time6_morning', 'time6_evening',
           'time7_night', 'time7_day', 'time7_morning', 'time7_evening',
           'time8_night', 'time8_day', 'time8_morning', 'time8_evening',
           'time9_night', 'time9_day', 'time9_morning', 'time9_evening',
           'time10_night', 'time10_day', 'time10_morning', 'time10_evening']
    df_predict[bool_cols] = df_predict[bool_cols].astype('int64')
    
    df_predict = df_predict.drop(columns=['time1', 'time2', 'time3', 'time4', 'time5', 'time6', 'time7', 'time8', 'time9', 'time10'])

    return df_predict

# использование модели
def model_predict(model, df_pr):
    predict = model.predict_proba(df_pr)[:, 1]
    predict_rounded = np.round(predict, 3)
    return predict_rounded * 100




# 1ый сайт:
st.html("<p><u><span style='font-size: 22px;'>Сайт №1: </span></u></p>")
url1 = st.text_input('Введите ссылку на сайт: ', key = 'li1')
#if url1:site_proverka(url1)
data_s1 = st.date_input('Введите дату посещения: ', value=None, min_value=datetime(2000, 1, 1).date(), max_value = date_now, key = 'd1')
time_s1 = st.text_input('Введите время: ', placeholder='HH:MM:SS', key = 'vr1')
if data_s1 and time_s1: 
    if chek_time(time_s1):
       time_mod1 = datetime.strptime(time_s1, '%H:%M:%S').time()
       datetime_1 = datetime.combine(data_s1, time_mod1)
else:
    time_mod1 = 0
    datetime_1 = 0
st.markdown('-------------------')
# 2ой сайт:
st.html("<p><u><span style='font-size: 22px;'>Сайт №2: </span></u></p>")
url2 = st.text_input('Введите ссылку на сайт: ', key = 'li2')
#if url2: site_proverka(url2)
data_s2 = st.date_input('Введите дату посещения: ', value=None, min_value=datetime(2000, 1, 1).date(), max_value = date_now, key = 'd2')
time_s2 = st.text_input('Введите время: ', placeholder='HH:MM:SS', key = 'vr2')
if data_s2 and time_s2:
    if chek_time(time_s2):
      time_mod2 = datetime.strptime(time_s2, '%H:%M:%S').time()
      datetime_2 = datetime.combine(data_s2, time_mod2)
else:
    time_mod2 = 0
    datetime_2 = 0
st.markdown('-------------------')
# 3ий сайт:
st.html("<p><u><span style='font-size: 22px;'>Сайт №3: </span></u></p>")
url3 = st.text_input('Введите ссылку на сайт: ', key = 'li3')
#if url3: site_proverka(url3)
data_s3 = st.date_input('Введите дату посещения: ', value=None, min_value=datetime(2000, 1, 1).date(), max_value = date_now, key = 'd3')
time_s3 = st.text_input('Введите время: ', placeholder='HH:MM:SS',key = 'vr3')
if data_s3 and time_s3:
    if chek_time(time_s3):
      time_mod3 = datetime.strptime(time_s3, '%H:%M:%S').time()
      datetime_3 = datetime.combine(data_s3, time_mod3)
else:
    time_mod3 = 0
    datetime_3 = 0
st.markdown('-------------------')
# 4ый
st.html("<p><u><span style='font-size: 22px;'>Сайт №4: </span></u></p>")
url4 = st.text_input('Введите ссылку на сайт: ', key = 'li4')
#if url4: site_proverka(url4)
data_s4 = st.date_input('Введите дату посещения: ', value=None, min_value=datetime(2000, 1, 1).date(), max_value = date_now, key = 'd4')
time_s4 = st.text_input('Введите время: ', placeholder='HH:MM:SS',key = 'vr4')
if data_s4 and time_s4:
    if chek_time(time_s4):
      time_mod4 = datetime.strptime(time_s4, '%H:%M:%S').time()
      datetime_4 = datetime.combine(data_s4, time_mod4)
else:
    time_mod4 = 0
    datetime_4 = 0
st.markdown('-------------------')
# 5ый
st.html("<p><u><span style='font-size: 22px;'>Сайт №5: </span></u></p>")
url5 = st.text_input('Введите ссылку на сайт: ', key = 'li5')
#if url5: site_proverka(url5)
data_s5 = st.date_input('Введите дату посещения: ', value=None, min_value=datetime(2000, 1, 1).date(), max_value = date_now, key = 'd5')
time_s5 = st.text_input('Введите время: ', placeholder='HH:MM:SS',key = 'vr5')
if data_s5 and time_s5:
    if chek_time(time_s5):
        time_mod5 = datetime.strptime(time_s5, '%H:%M:%S').time()
        datetime_5 = datetime.combine(data_s5, time_mod5)
else:
    time_mod5 = 0
    datetime_5 = 0
st.markdown('-------------------')
# 6ой
st.html("<p><u><span style='font-size: 22px;'>Сайт №6: </span></u></p>")
url6 = st.text_input('Введите ссылку на сайт: ', key = 'li6')
#if url6: site_proverka(url6)
data_s6 = st.date_input('Введите дату посещения: ', value=None, min_value=datetime(2000, 1, 1).date(), max_value = date_now, key = 'd6')
time_s6 = st.text_input('Введите время: ', placeholder='HH:MM:SS',key = 'vr6')
if data_s6 and time_s6:
    if chek_time(time_s6):
        time_mod6 = datetime.strptime(time_s6, '%H:%M:%S').time()
        datetime_6 = datetime.combine(data_s6, time_mod6)
else:
    time_mod6 = 0
    datetime_6 = 0
st.markdown('-------------------')
# 7ой
st.html("<p><u><span style='font-size: 22px;'>Сайт №7: </span></u></p>")
url7 = st.text_input('Введите ссылку на сайт: ', key = 'li7')
#if url7: site_proverka(url7)
data_s7 = st.date_input('Введите дату посещения: ', value=None, min_value=datetime(2000, 1, 1).date(), max_value = date_now, key = 'd7')
time_s7 = st.text_input('Введите время: ', placeholder='HH:MM:SS',key = 'vr7')
if data_s7 and time_s7:
    if chek_time(time_s7):
        time_mod7 = datetime.strptime(time_s7, '%H:%M:%S').time()
        datetime_7 = datetime.combine(data_s7, time_mod7)
else:
    time_mod7 = 0
    datetime_7 = 0
st.markdown('-------------------')
# 8ой
st.html("<p><u><span style='font-size: 22px;'>Сайт №8: </span></u></p>")
url8 = st.text_input('Введите ссылку на сайт: ', key = 'li8')
#if url8: site_proverka(url8)
data_s8 = st.date_input('Введите дату посещения: ', value=None, min_value=datetime(2000, 1, 1).date(), max_value = date_now, key = 'd8')
time_s8 = st.text_input('Введите время: ', placeholder='HH:MM:SS',key = 'vr8')
if data_s8 and time_s8:
    if chek_time(time_s8):
        time_mod8 = datetime.strptime(time_s8, '%H:%M:%S').time()
        datetime_8 = datetime.combine(data_s8, time_mod8)
else:
    time_mod8 = 0
    datetime_8 = 0
st.markdown('-------------------')
# 9ый
st.html("<p><u><span style='font-size: 22px;'>Сайт №9: </span></u></p>")
url9 = st.text_input('Введите ссылку на сайт: ', key = 'li9')
#if url9: site_proverka(url9)
data_s9 = st.date_input('Введите дату посещения: ', value=None, min_value=datetime(2000, 1, 1).date(), max_value = date_now, key = 'd9')
time_s9 = st.text_input('Введите время: ', placeholder='HH:MM:SS',key = 'vr9')
if data_s9 and time_s9:
    if chek_time(time_s9):
        time_mod9 = datetime.strptime(time_s9, '%H:%M:%S').time()
        datetime_9 = datetime.combine(data_s9, time_mod9)
else:
    time_mod9 = 0
    datetime_9 = 0
st.markdown('-------------------')
# 10ый
st.html("<p><u><span style='font-size: 22px;'>Сайт №10: </span></u></p>")
url10 = st.text_input('Введите ссылку на сайт: ', key = 'li10')
#if url10: site_proverka(url10)
data_s10 = st.date_input('Введите дату посещения: ', value=None, min_value=datetime(2000, 1, 1).date(), max_value = date_now, key = 'd10')
time_s10 = st.text_input('Введите время: ', placeholder='HH:MM:SS',key = 'vr10')
if data_s10 and time_s10:
    if chek_time(time_s10):
        time_mod10 = datetime.strptime(time_s10, '%H:%M:%S').time()
        datetime_10 = datetime.combine(data_s10, time_mod10)
else:
    time_mod10 = 0
    datetime_10 = 0
st.markdown('-------------------')

# настройки кнопки
st.html(
    """
    <style>
    .stButton > button {
        display: block;
        margin: 0 auto;
        width: 400px;
        height: 60px;
        text-align: center;
        text-decoration: none;
        font-size: 20px;
        border-radius: 12px;
    }
    </style>
    """
)
# кнопка
button = st.button('Определить', type="primary")
if button:
    # дф
    df_dict = {
        'site1': url1,
        'time1': datetime_1,
        'site2': url2,
        'time2': datetime_2,
        'site3': url3,
        'time3': datetime_3,
        'site4': url4,
        'time4': datetime_4,
        'site5': url5,
        'time5': datetime_5,
        'site6': url6,
        'time6': datetime_6,
        'site7': url7,
        'time7': datetime_7,
        'site8': url8,
        'time8': datetime_8,
        'site9': url9,
        'time9': datetime_9,
        'site10': url10,
        'time10': datetime_10,
    }
    
    df_user = pd.DataFrame([df_dict])
    df_user[df_user == ''] = 0
    st.dataframe(df_user)

    df_predict = df_user
    df_predict = mod_df(df_predict)
    # st.dataframe(df_predict)   
    
    predict_val = model_predict(model, df_predict)
    res = np.array2string(predict_val, separator=', ', prefix='', suffix='').strip('[]')
    st.write(f'<p><u><span style="font-size: 26px;">Сессия принадлежит мошеннику с вероятностью {res}%.</span></u></p>', unsafe_allow_html = True)


