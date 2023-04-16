import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

#Стиль графика
plt.style.use('dark_background')
df = pd.read_csv("ses3ready.csv", encoding="utf-8-sig", nrows=100000)
#Заголовок прописаный тегами html
html_temp = """
<div style="background-color:tomato;padding:15px">
<h1 style="color:white;text-align:center;">Предсказание признаков</h1>
</div>"""
st.markdown(html_temp,unsafe_allow_html=True)
st.write(f" ")
st.write(
    '''
    Данное веб приложение содержит 5 интерактивных полей:
1)	Поле ввода с датой – туда можно ввести год (например “2024”) или полную дату (“2024-01-01”).
2)	Ползунок, который отвечает за отрисовку графика в нижней части экрана, он показывает предсказанную тенденцию по дням (на него можно нажать мышкой и нажимая стрелочки будет мотаться по 1му дню).
3)	Следующие это выпадающий список, там можно выбрать желаемый для предсказания кластер.
4)	Предпоследнее это выбор атрибута для предсказания (на данный момент их там два – total_amount и trip_distance)
5)	Внизу экрана расположен график, немного справа от него сверху есть кнопка, если ее нажать график откроется шире и его можно будет рассмотреть.
Прямо под графиком расположено предсказываемое число.
    '''
)
st.write(f" ")


#Пользовательский воод
user_input = st.text_input("Введите дату на которую хотите получить предсказание в формате\n2024-01-01", 2024)
#Тут задаются параметры кнопок и ползунков
steps=st.slider("Количество дней для предсказания", 10, 1600, step=1)
#Выпадающий список кластеров
selected_cluster=st.selectbox("Выберите 1 из кластеров", (df['clustername'].unique()))

selected_cluster=df[df["clustername"] == selected_cluster]["BMCLUSTS"].values[0]
#Выпадающий список атрибутов
selected_attribute=st.selectbox("Выберите 1 из атрибутов", ("total_amount", "trip_distance"))

#Алгоритм, который определяет какую модель нужно подключить в зависимости от кластера или атрибута
strfmodel = "models/"
#2 условия, которые определяют путь до регрессионной модели
if selected_attribute == "total_amount":
    strfmodel += "amountC" + str(selected_cluster) +".sav"
    # Загрузка модели
    loaded_model = pickle.load(open(strfmodel, 'rb'))
elif selected_attribute == "trip_distance":
    strfmodel += "distanceC" + str(selected_cluster) +".sav"
    # Загрузка модели
    loaded_model = pickle.load(open(strfmodel, 'rb'))


#Импортирую API и преминяю его функции
from api import API
y = API.attributor(selected_attribute, selected_cluster, df)
#Получаю список с предсказаниями и линией прогноза
preds = API.modelfitter(y, steps,user_input,loaded_model)


forecasted = preds[1]


# Простая визуализация предсказания временных рядов
ax = y.plot(label='observed', figsize=(20, 15))
# Построение графика осуществляется на основе характеристики forecast (это итоговое предсказание)
preds[0].predicted_mean.plot(ax=ax, label='Forecast')
pred_ci = preds[0].conf_int()

fig, ax = plt.subplots()
#Ось графика с отрисованной линией
ax = y.plot(label='observed', figsize=(20, 15))
#Индексы для построения графиков
preds[0].predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
pred_ci.iloc[:, 0],
pred_ci.iloc[:, 1], color='y', alpha=.25)
ax.set_xlabel('Date')
#Заголовки
ax.set_ylabel(f"Предсказание {selected_attribute} по кластеру {selected_cluster}")
ax.set_title('Предсказание тенденции изменения характеристик поездки такси с течением времени для United States')
plt.legend()
st.pyplot(plt)

st.write(f"Предсказание на выбранный вами отрезок времени - {round(preds[1][0])}")
st.write(f" ")