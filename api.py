import warnings
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
pd.options.display.max_columns=50

class API():
    def attributor(attribute, claster, df):
        # to_datetime метод, который автоматически преобразует дату в првавильный формат
        df[["tpep_pickup_datetime", "tpep_dropoff_datetime"]] = pd.to_datetime(
            df[["tpep_pickup_datetime", "tpep_dropoff_datetime"]].stack(), format='%Y-%m-%d %H:%M:%S').unstack()
        # Создание столбца с уникальными датами, это необходимо для работы модели строящей времянные ряды
        df["tpep_pickup_datetime"] = df["tpep_pickup_datetime"].map(lambda x: x.date())
        # Удаление дубликатов при помощи drop_duplicates
        df = df.drop_duplicates(subset=['tpep_pickup_datetime'])
        # Сброс индексов
        df.reset_index(drop=True, inplace=True)

        df = df.sort_values(by='tpep_pickup_datetime')
        # df.reset_index(inplace=True)
        # Создания копии колонки с датой
        df['datecopy'] = df['tpep_pickup_datetime']
        # Установка временных индексов вместо обычных
        df.set_index('datecopy', inplace=True)
        # Привести индексы к формату datetime
        df.index = pd.to_datetime(df.index)
        # Создание не большого набора дыннх для дальнейших пердсказаний атрибута
        chi = df[df['BMCLUSTS'] == claster]
        chi = chi[[attribute]]
        # Переимоновывание атрибута
        chi = chi.rename(columns={attribute: "co2"})
        # Автоматическое заполнение пропусков дат (без этого мадоль не будет работать)
        y = chi.asfreq('D')
        # Заполнение пустых значений нулями
        y.fillna(0, inplace=True)

        # 'MS' группирует месячные данные
        y = y['co2']

        # bfill значит, что нужно использовать значение до заполнения пропущенных значений
        y = y.fillna(y.bfill())
        # Построение графика распределения
        y.plot(figsize=(15, 6))
        plt.show()
        return y

    def modelfitter(y, steps, date, mod):
        results = mod
        print(results.summary().tables[1])
        # Предсказание для выявление средней квадратичной ошибки
        pred_dynamic = results.get_prediction(start=pd.to_datetime(y.index[0]), dynamic=True, full_results=True)
        pred_dynamic_ci = pred_dynamic.conf_int()

        # Извлечь прогнозируемые и истинные значения временного ряда
        y_forecasted = pred_dynamic.predicted_mean
        y_truth = y[y.index[0]:]  # Вычислить среднеквадратичную ошибку
        mse = ((y_forecasted - y_truth) ** 2).mean()
        print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

        # Предсказание по выбраной дате
        pred_dynamic = results.get_prediction(start=pd.to_datetime(date), dynamic=True, full_results=True)
        pred_dynamic_ci = pred_dynamic.conf_int()
        y_forecasted = pred_dynamic.predicted_mean

        # Получить прогноз
        pred_uc = results.get_forecast(steps=steps)
        # Получить интервал прогноза
        return [pred_uc, y_forecasted]