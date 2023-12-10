                                  Задача от «IaC Bank of China»

 Прогнозирование Финансового Инструмента с Использованием Датасайнс Методов



Цель:
Определить применимость методов датасайнса для прогнозирования значений финансового инструмента на основе предоставленных данных.

Данная работа выполнена на основе статьи по прогнозированию временных рядов , а именно на работе по предсказанию температуры воздуха (https://www.tensorflow.org/tutorials/structured_data/time_series?hl=ru#setup).
Вводные данные:
- Предоставлен CSV-файл с данными о нескольких финансовых инструментах и соответствующими фичами, которые, предположительно, коррелируют с изменениями значений финансовых инструментов.
Цель:
Определить применимость методов датасайнса для прогнозирования значений финансового инструмента на основе предоставленных данных, а именно прогнозирование курса валюты .


Это руководство представляет собой введение в прогнозирование временных рядов с использованием TensorFlow. Он строит несколько разных стилей моделей, включая сверточные и рекуррентные нейронные сети (CNN и RNN).

Он состоит из двух основных частей с подразделами:

### Прогноз для одного временного шага:
Единственная особенность.
Все функции.
### Прогноз нескольких шагов:
Single-shot: Делайте прогнозы сразу.
Авторегрессия: делайте по одному прогнозу за раз и отправляйте выходные данные обратно в модель.

Установим необходимые библиотеки для решения нашей задачи.
#импорт библиотек
import tensorflow as tf

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os  
Набор данных предоставленный заказчиком «IaC Bank of China»
В этом руководстве используется набор данных временных рядов курса валют и т.д., записанный банком на протяжении длительного времени.

Этот набор данных содержит 27 различных столбцов, таких как курсы валют по отношению к доллару,форвард поинт, всевозможные коэффициенты и индексы, а так же  3892 строки. Они собирались  начиная с 2007 года. Для эффективности вы будете использовать только данные, собранные в период с 2007 по 2019 год. Данные с 2019 по 2021  используются для тестовой выборки данных
#variables
data='brl'

#импорт данных, сделать циклом
data = pd.read_csv('data/all_data.csv', sep=',')

#конвертация данных
data['Date'] = pd.to_datetime(data['Date'], dayfirst=False)

### Давайте взглянем на  данные. Вот первые несколько строк:
data
# создадим копию данных
drop_data = data.copy()
### Осмотр и очистка данных
#удаление пустых строчек
drop_data = drop_data.dropna(how='all')
#удаление пропусков спота из-за выходных и праздников(требование заказчика)
drop_data = drop_data.dropna(subset=['USDBRL Curncy'])

#добавление предыдущих значений признаков в пустые места(требование заказчика)
for col in drop_data.select_dtypes(include=['int', 'float']):
    while drop_data[col].isnull().any(): 
        drop_data[col] = drop_data[col].fillna(method='ffill')

#вычисление изменения цены

drop_data['devprice'] = (drop_data['USDBRL Curncy'].shift(-1) - (drop_data['USDBRL Curncy'] + (drop_data['BCN1W BGN Curncy'] / 50000))) / drop_data['USDBRL Curncy'].shift(-1)

#перенос последнего столбца на 4-тое место
new_columns = drop_data.columns[:-1].to_list()
new_columns.insert(1, 'devprice')
t_data_0 = drop_data[new_columns]
#удаление 'USDBRL Curncy' и 'BCN1W BGN Curncy' 
t_data = t_data_0.copy()
t_data.drop(['USDBRL Curncy', 'BCN1W BGN Curncy'], axis=1, inplace=True)
t_data.drop(t_data.tail(1).index, inplace=True)
Проверка наших данных на изменения , которые мы хотели сделать
t_data
### Использование инструментов визуалиазации для анализа данных
# построение графиков распределения признаков для различных классов

# Перебор столбцов и построение графиков
for column in t_data.columns:
    if column not in ['Date']:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
        axes.scatter(
        x=t_data['Date'],
        y=t_data[column],
        s=1,
        marker='o',
        c = 'red',
        label=column
        );

    
                # Настройка осей и заголовка
        plt.xlabel('Date')
        #axes[0].set_ylabel(column)
        #axes[1].set_ylabel(column)
            
        # Добавление легенды
        axes.legend()
                
                    # Отображение графика
        plt.show()
Визуализация данных , которые нам предоставили, не имеет сильных локальных выбросов, что позволяет нам продолжить дальнейшую обработку и подготовку данных для наших моделей.
#вычисление таргета
#t_data['target']=(t_data['devprice']-t_data['devprice'].mean())/t_data['devprice'].std()
t_data['target']=(t_data['devprice']-t_data['devprice'].min())*(10/(t_data['devprice'].max()-t_data['devprice'].min()))-5
#t_data['target'] = np.where(t_data['target'] == -0.0, 0.0, t_data['target'])
new_columns = t_data.columns[:-1].to_list()
new_columns.insert(2, 'target')
t_data = t_data[new_columns]
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
axes.scatter(
    x=t_data['Date'],
    y=t_data['target'],
    s=1,
    marker='o',
    c = 'red',
    label=column
);
## Статистический анализ
Этот код выполняет описательную статистику для каждого столбца в DataFrame t_data и выводит результаты в транспонированной форме. 

Описательная статистика включает в себя следующие значения: 

- count: количество непустых значений в столбце
- mean: среднее значение
- std: стандартное отклонение
- min: минимальное значение
- 25%: нижний квартиль (25-й процентиль)
- 50%: медиана (50-й процентиль)
- 75%: верхний квартиль (75-й процентиль)
- max: максимальное значение

Транспонирование (transpose()) меняет столбцы на строки и наоборот, чтобы результаты описательной статистики были легко читаемыми.



t_data.describe().transpose()
Создание тестого набора, который составляет 20 процентов от общего набора данных.
#выделение тестового набора для сохранения ненормализованных значений
n = len(t_data)
test_df_0 = t_data[int(n*0.8):]
# Разработка функций
Прежде чем погрузиться в построение модели, важно понять наши данные и убедиться, что мы  передаете модели данные в соответствующем формате.
# Преобразование datetime в дни
data_start=t_data['Date'].min()
t_data['Date'] = (t_data['Date'] - data_start).dt.days
#нормализация данных к диапазону от -1 до 1
t_data=t_data.drop('devprice', axis=1)
for column in t_data.columns:
    if column not in ['target']:
        t_data[column]=(t_data[column]-t_data[column].min())*(10/(t_data[column].max()-t_data[column].min()))-5
обучающие, проверочные и тестовые данные
#разбиение данных
column_indices = {name: i for i, name in enumerate(t_data.columns)}


train_df = t_data[0:int(n*0.6)]
val_df = t_data[int(n*0.6):int(n*0.8)]
test_df = t_data[int(n*0.8):]

### Окно данных
Модели в этом руководстве будут делать набор прогнозов на основе окна последовательных выборок из данных.

Основные особенности окон ввода:

  -Ширина (количество временных шагов) окон ввода и метки.

  -Смещение времени между ними.

  -Какие функции используются в качестве входных данных, меток или того и другого.

В этом руководстве создаются различные модели (включая линейные модели, модели DNN, CNN и RNN) и используются они для обеих целей:

  -Прогнозы с одним выходом и несколькими выходами .

  -Прогнозы с одним и несколькими временными шагами .

В этом разделе основное внимание уделяется реализации окна данных, чтобы его можно было повторно использовать для всех этих моделей.

В зависимости от задачи и типа модели может потребоваться создание различных окон данных. Вот некоторые примеры:

 1.Например, чтобы сделать один прогноз на 24 часа вперед, учитывая 24-часовую историю, вы можете определить окно следующим образом:
 
 
 
   
![image.png](attachment:image.png)













  Одно предсказание на 24 часа вперед.
  

  2.Модель, которая делает прогноз на один час вперед, учитывая шесть часов истории, нуждалась бы в таком окне:
  
  
  ![image-2.png](attachment:image-2.png)
  
  
  
  
  
  
  
  
  
  





  Одно предсказание на час вперед


В оставшейся части этого раздела определяется класс WindowGenerator . Этот класс может:

  1.Обрабатывайте индексы и смещения, как показано на диаграммах выше.
  
  2.Разделить окна функций на пары (features, labels) .
  
  3.Постройте содержимое получившихся окон.
  
  4.Эффективно генерируйте пакеты этих окон из обучающих, оценочных и тестовых данных,
    используя      tf.data.Dataset s.
  
### Индексы и смещения
Начните с создания класса WindowGenerator . Метод __init__ включает всю необходимую логику для индексов ввода и меток.

Он также принимает обучающие, оценочные и тестовые кадры данных в качестве входных данных. Позже они будут преобразованы в
   tf.data.Dataset окон.   
#создание класса окна
class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df=train_df, val_df=val_df, test_df=test_df,
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])
###  Сплит

Учитывая список последовательных входных данных, метод split_window преобразует их в окно входных данных и окно меток.

Пример , который мы определили ранее, будет разбит следующим образом:




![split_window.png](attachment:split_window.png)









Начальное окно представляет собой все последовательные выборки, это разбивает его на пары (входы, метки).

На этой диаграмме не показана ось features данных, но эта функция split_window также обрабатывает label_columns , поэтому ее можно использовать как для примеров с одним выходом, так и для примеров с несколькими выходами.
Теперь в объекте WindowGenerator доступен метод split_window, который можно использовать для разделения окон на входы и метки в заданных размерностях.
#сплит с учетом разделения окон и меток
def split_window(self, features):
  inputs = features[:, self.input_slice, :]
  labels = features[:, self.labels_slice, :]
  if self.label_columns is not None:
    labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        axis=-1)

  # Slicing doesn't preserve static shape information, so set the shapes
  # manually. This way the `tf.data.Datasets` are easier to inspect.
  inputs.set_shape([None, self.input_width, None])
  labels.set_shape([None, self.label_width, None])

  return inputs, labels

WindowGenerator.split_window = split_window
#### Вот метод построения графика, который позволяет легко визуализировать разделенное окно:
    "Эта функция plot добавляется в класс WindowGenerator. Она позволяет визуализировать данные и (опционально) предсказания модели на графиках. Вот что делает эта функция:
    Принимает входные параметры self, model, plot_col и max_subplots
    Получает входы и метки (labels) из self.example, которое предполагается, 
    что было установлено заранее
    Создает график с заданными размерами (12 на 8 дюймов)
    Определяет индекс столбца plot_col в массиве данных (plot_col_index)
    Определяет максимальное количество подграфиков, которые могут быть созданы (max_n), 
    ограниченное заданным значением max_subplots и количеством доступных входов 
    Для каждого подграфика в пределах max_n  
    Создает отдельный подграфик с меткой оси y, соответствующей plot_col.    
    Рисует график входов (inputs) на подграфике  
    Если есть метки (labels), рисует точки меток на графике  
    Если есть модель, рисует точки предсказаний на графике  
    Добавляет легенду на первом подграфике
    Добавляет метку оси x, соответствующую 'days'
    
    
    
  Теперь в объекте WindowGenerator доступна функция plot, которую можно использовать для визуализации данных и предсказаний модели.
def plot(self, model=None, plot_col='target', max_subplots=3):
  inputs, labels = self.example
  plt.figure(figsize=(12, 8))
  plot_col_index = self.column_indices[plot_col]
  max_n = min(max_subplots, len(inputs))
  for n in range(max_n):
    plt.subplot(max_n, 1, n+1)
    plt.ylabel(f'{plot_col} [normed]')
    plt.plot(self.input_indices, inputs[n, :, plot_col_index],
             label='Inputs', marker='.', zorder=-10)

    if self.label_columns:
      label_col_index = self.label_columns_indices.get(plot_col, None)
    else:
      label_col_index = plot_col_index

    if label_col_index is None:
      continue

    plt.scatter(self.label_indices, labels[n, :, label_col_index],
                edgecolors='k', label='Labels', c='#2ca02c', s=64)
    if model is not None:
      predictions = model(inputs)
      plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                  marker='X', edgecolors='k', label='Predictions',
                  c='#ff7f0e', s=64)

    if n == 0:
      plt.legend()

  plt.xlabel('days')

WindowGenerator.plot = plot
### Создайтем tf.data.Dataset s
Этот метод make_dataset возьмет временной ряд DataFrame и преобразует его в tf.data.Dataset из пар (input_window, label_window) с помощью функции tf.keras.utils.timeseries_dataset_from_array :



# создание tf.data.Dataset
def make_dataset(self, data):
  data = np.array(data, dtype=np.float32)
  ds = tf.keras.utils.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=True,
      batch_size=20,)

  ds = ds.map(self.split_window)

  return ds

WindowGenerator.make_dataset = make_dataset
Объект WindowGenerator содержит обучающие, проверочные и тестовые данные.

Добавьте свойства для доступа к ним как tf.data.Dataset с помощью метода make_dataset , который вы определили ранее. Кроме того, добавьте стандартный пакет примеров для быстрого доступа и построения графиков:
@property
def train(self):
  return self.make_dataset(self.train_df)

@property
def val(self):
  return self.make_dataset(self.val_df)

@property
def test(self):
  return self.make_dataset(self.test_df)

@property
def example(self):
  """Get and cache an example batch of `inputs, labels` for plotting."""
  result = getattr(self, '_example', None)
  if result is None:
    # No example batch was found, so get one from the `.train` dataset
    result = next(iter(self.train))
    # And cache it for next time
    self._example = result
  return result

WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example
### Одноступенчатые модели
Самая простая модель, которую вы можете построить на такого рода данных, — это модель, которая предсказывает значение одной функции — 1 временной шаг (один час) в будущее, основываясь только на текущих условиях.

Итак, начните с построения моделей для прогнозирования значения 'target' на один час вперед.


![narrow_window.png](attachment:narrow_window.png)














Предсказать следующий временной шаг

Настройте объект WindowGenerator для создания этих одношаговых пар (input, label)
#Одноступенчатые модели
single_step_window = WindowGenerator(
    input_width=1, label_width=1, shift=1,
    label_columns=['target'])
single_step_window
Объект window создает tf.data.Dataset из обучающих, проверочных и тестовых наборов, что позволяет легко перебирать пакеты данных.
for example_inputs, example_labels in single_step_window.train.take(1):
  print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
  print(f'Labels shape (batch, time, features): {example_labels.shape}')

### Базовый уровень
Перед созданием обучаемой модели было бы неплохо иметь базовый уровень производительности в качестве точки для сравнения с более поздними более сложными моделями.

Эта первая задача состоит в том, чтобы предсказать 'target' на один час вперед, учитывая текущее значение всех признаков. Текущие значения включают текущий 'target'.

Итак, начнем с модели, которая просто возвращает текущуй 'target' в качестве прогноза, прогнозируя «без изменений». Это разумная базовая линия, поскольку 'target' изменяется медленно. Конечно, эта базовая линия будет работать хуже, если вы сделаете прогноз в будущем.











![baseline.png](attachment:baseline.png)


#базовый прогноз, возращает текущую цену в качестве прогноза, прогнозируя «без изменений»
class Baseline(tf.keras.Model):
  def __init__(self, label_index=None):
    super().__init__()
    self.label_index = label_index

  def call(self, inputs):
    if self.label_index is None:
      return inputs
    result = inputs[:, :, self.label_index]
    return result[:, :, tf.newaxis]

Создайте экземпляр и оцените эту модель:
baseline = Baseline(label_index=column_indices['target'])

baseline.compile(loss=tf.losses.MeanSquaredError(),
                 metrics=[tf.metrics.MeanAbsoluteError()])

val_performance = {}
performance = {}
val_performance['Baseline'] = baseline.evaluate(single_step_window.val)
performance['Baseline'] = baseline.evaluate(single_step_window.test, verbose=0)

Это напечатало некоторые показатели производительности, но они не дают вам представления о том, насколько хорошо работает модель.

В WindowGenerator есть метод plot, но графики будут не очень интересными только с одним образцом.

Итак, создайте более широкий WindowGenerator , который генерирует окна 24 часа последовательных входных данных и меток за раз. Новая переменная wide_window не меняет способ работы модели. Модель по-прежнему делает прогнозы на один час вперед на основе одного входного временного шага. Здесь time ось действует как batch ось: каждый прогноз делается независимо, без взаимодействия между временными шагами:
wide_window = WindowGenerator(
    input_width=50, label_width=50, shift=1,
    label_columns=['target'])

test_window = WindowGenerator(
    input_width=1, label_width=1, shift=1,
    label_columns=['target'])

wide_window
Это расширенное окно можно передать непосредственно в ту же baseline модель без каких-либо изменений кода. Это возможно, потому что входы и метки имеют одинаковое количество временных шагов, а базовая линия просто перенаправляет вход на выход:









![last_window.png](attachment:last_window.png)
print('Input shape:', wide_window.example[0].shape)
print('Output shape:', baseline(wide_window.example[0]).shape)

Построив прогнозы базовой модели, обратите внимание, что это просто метки, сдвинутые вправо на один час:
wide_window.plot(baseline)
#это просто метки, сдвинутые вправо на один день
    #Синяя линия Inputs показывает входную температуру на каждом временном шаге. Модель получает все функции, этот график показывает только температуру.
    #Зеленые точки Labels показывают целевое значение прогноза. Эти точки отображаются во время прогнозирования, а не во время ввода. Поэтому диапазон меток смещен на 1 шаг относительно входов.
    #Оранжевые кресты Predictions — это прогнозы модели для каждого выходного временного шага. Если бы модель предсказывала идеально, прогнозы попадали бы прямо в Labels .
На приведенных выше графиках трех примеров одноэтапная модель работает в течение 24 часов. Это заслуживает некоторого пояснения:

- Синяя линия Inputs показывает входную температуру на каждом временном шаге. Модель получает все функции, этот график показывает только температуру.
- Зеленые точки Labels показывают целевое значение прогноза. Эти точки отображаются во время прогнозирования, а не во время ввода. Поэтому диапазон меток смещен на 1 шаг относительно входов.
- Оранжевые кресты Predictions — это прогнозы модели для каждого выходного временного шага. Если бы модель предсказывала идеально, прогнозы попадали бы прямо в Labels .
### Линейная модель
Самая простая обучаемая модель, которую вы можете применить к этой задаче, — это вставить линейное преобразование между входом и выходом. В этом случае результат временного шага зависит только от этого шага:










![narrow_window.png](attachment:narrow_window.png)

Слой tf.keras.layers.Dense без набора activation является линейной моделью. Слой преобразует только последнюю ось данных из (batch, time, inputs) в (batch, time, units) ; он применяется независимо к каждому элементу по осям batch и time .
# линейная модель
linear = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1)
])

print('Input shape:', single_step_window.example[0].shape)
print('Output shape:', linear(single_step_window.example[0]).shape)

В этом руководстве обучается множество моделей, поэтому упакуйте процедуру обучения в функцию:
MAX_EPOCHS = 100

def compile_and_fit(model, window, patience=30):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
  return history
Обучим модель и оценим ее производительность:
history = compile_and_fit(linear, single_step_window)

val_performance['Linear'] = linear.evaluate(single_step_window.val)
performance['Linear'] = linear.evaluate(single_step_window.test, verbose=0)

Как и baseline модель, линейную модель можно вызывать для пакетов широких окон. При таком использовании модель делает набор независимых прогнозов на последовательных временных шагах. Ось time действует как другая ось batch . Между прогнозами на каждом временном шаге нет взаимодействий.









![wide_window.png](attachment:wide_window.png)
print('Input shape:', wide_window.example[0].shape)
print('Output shape:', baseline(wide_window.example[0]).shape)

Вот график его примерных прогнозов для wide_window , обратите внимание, что во многих случаях прогноз явно лучше, чем просто возврат входной температуры, но в некоторых случаях он хуже:
wide_window.plot(linear)

Одним из преимуществ линейных моделей является то, что их относительно просто интерпретировать. 
test_df
target_ln=linear.predict(test_df)
target_ln.shape
target_ln
df_ln = test_df_0.iloc[0:, 0:3]
df_ln
df_ln.loc[:,'target_ln']=target_ln
#df_ln['Date']=df_ln['Date'].apply(lambda x: data_start + pd.Timedelta(days=x))
df_ln = df_ln[['Date', 'devprice', 'target', 'target_ln']]
df_ln['pnl']=(df_ln['target_ln']/5)*df_ln['devprice']
df_ln['sum']=df_ln["pnl"].cumsum()
sharp=(df_ln['sum'].mean()*255**0.5)/df_ln['sum'].std()
print(df_ln, sharp)

### Многоступенчатый плотный
Одношаговая модель не имеет контекста для текущих значений входных данных. Он не может видеть, как входные объекты меняются с течением времени. Чтобы решить эту проблему, модели требуется доступ к нескольким временным шагам при прогнозировании:















![conv_window.png](attachment:conv_window.png)

baseline , linear и dense модели обрабатывали каждый временной шаг независимо. Здесь модель будет принимать несколько временных шагов в качестве входных данных для получения одного вывода.

Создайте WindowGenerator , который будет создавать пакеты трехчасовых входных данных и одночасовых меток:

Обратите внимание, что параметр shift Window относится к концу двух окон.
# Многоступенчатый плотный слой
CONV_WIDTH = 7
conv_window = WindowGenerator(
    input_width=CONV_WIDTH,
    label_width=1,
    shift=1,
    label_columns=['target'])

conv_window

conv_window.plot()
plt.title("Given 3 days of inputs, predict 1 day into the future.")
Вы можете обучить dense модель в окне с несколькими входными шагами, добавив tf.keras.layers.Flatten в качестве первого слоя модели:
multi_step_dense = tf.keras.Sequential([
    # Shape: (time, features) => (time*features)
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1),
    # Add back the time dimension.
    # Shape: (outputs) => (1, outputs)
    tf.keras.layers.Reshape([1, -1]),
])

print('Input shape:', conv_window.example[0].shape)
print('Output shape:', multi_step_dense(conv_window.example[0]).shape)

history = compile_and_fit(multi_step_dense, conv_window)

val_performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.val)
performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.test, verbose=0)

from IPython.display import clear_output
conv_window.plot(multi_step_dense)

target_msd=multi_step_dense.predict(test_df)
target_msd.shape
target_msd
### Сверточная нейронная сеть
Слой свертки ( tf.keras.layers.Conv1D ) также использует несколько временных шагов в качестве входных данных для каждого прогноза.

Ниже представлена ​​та же модель, что и в multi_step_dense , переписанная с помощью свертки.

Обратите внимание на изменения:

tf.keras.layers.Flatten и первый tf.keras.layers.Dense заменяются tf.keras.layers.Conv1D .
tf.keras.layers.Reshape больше не нужен, так как свертка сохраняет ось времени в своих выходных данных.
#сверточная нейронная сеть
conv_model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=20,
                           kernel_size=(CONV_WIDTH,),
                           activation='relu'),
    tf.keras.layers.Dense(units=20, activation='relu'),
    tf.keras.layers.Dense(units=1),
])

Запустите его на примере пакета, чтобы убедиться, что модель выдает выходные данные с ожидаемой формой:
print("Conv model on `conv_window`")
print('Input shape:', conv_window.example[0].shape)
print('Output shape:', conv_model(conv_window.example[0]).shape)

Обучите и оцените его в conv_window и он должен дать производительность, аналогичную модели multi_step_dense .
history = compile_and_fit(conv_model, conv_window)

#IPython.display.clear_output()
val_performance['Conv'] = conv_model.evaluate(conv_window.val)
performance['Conv'] = conv_model.evaluate(conv_window.test, verbose=0)

Разница между этой conv_model и моделью multi_step_dense заключается в том, что conv_model можно запускать на входах любой длины. Сверточный слой применяется к скользящему окну входных данных:




















![wide_conv_window.png](attachment:wide_conv_window.png)

Если вы запустите его на более широком вводе, он выдаст более широкий вывод:
print("Wide window")
print('Input shape:', wide_window.example[0].shape)
print('Labels shape:', wide_window.example[1].shape)
print('Output shape:', conv_model(wide_window.example[0]).shape)

Обратите внимание, что вывод короче ввода. Чтобы обучение или построение графика работали, вам нужно, чтобы метки и прогноз имели одинаковую длину. Поэтому создайте WindowGenerator для создания широких окон с несколькими дополнительными временными шагами ввода, чтобы длина метки и предсказания совпадала:
LABEL_WIDTH = 50
INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
wide_conv_window = WindowGenerator(
    input_width=INPUT_WIDTH,
    label_width=LABEL_WIDTH,
    shift=1,
    label_columns=['target'])

wide_conv_window
print("Wide conv window")
print('Input shape:', wide_conv_window.example[0].shape)
print('Labels shape:', wide_conv_window.example[1].shape)
print('Output shape:', conv_model(wide_conv_window.example[0]).shape)

Теперь вы можете построить прогнозы модели в более широком окне. Обратите внимание на 3 шага входного времени перед первым прогнозом. Каждый прогноз здесь основан на 3 предыдущих временных шагах:
wide_conv_window.plot(conv_model)

target_msd=conv_model.predict(test_df)
target_msd.shape
target_msd
### Рекуррентная нейронная сеть
Рекуррентная нейронная сеть (RNN) — это тип нейронной сети, хорошо подходящий для данных временных рядов. RNN обрабатывают временной ряд шаг за шагом, сохраняя внутреннее состояние от шага к шагу.

Вы можете узнать больше о создании текста с помощью учебника по RNN и о рекуррентных нейронных сетях (RNN) с руководством по Keras .

В этом руководстве вы будете использовать слой RNN под названием Long Short-Term Memory ( tf.keras.layers.LSTM ).

Важным аргументом конструктора для всех слоев Keras RNN, таких как tf.keras.layers.LSTM , является аргумент return_sequences . Этот параметр может настроить слой одним из двух способов:

Если False по умолчанию, слой возвращает только выходные данные последнего временного шага, давая модели время, чтобы прогреть свое внутреннее состояние, прежде чем делать один прогноз:

![lstm_1_window.png](attachment:lstm_1_window.png)

















Если True , слой возвращает результат для каждого входа. Это полезно для:
Укладка слоев RNN.
Обучение модели на нескольких временных шагах одновременно.


#рекуррентная нейронная сеть lstm
lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(20, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])

С return_sequences=True модель можно обучать на данных за 24 часа за раз.
print('Input shape:', wide_window.example[0].shape)
print('Output shape:', lstm_model(wide_window.example[0]).shape)

history = compile_and_fit(lstm_model, wide_window)

#IPython.display.clear_output()
val_performance['LSTM'] = lstm_model.evaluate(wide_window.val)
performance['LSTM'] = lstm_model.evaluate(wide_window.test, verbose=0)

wide_window.plot(lstm_model)

target_msd=lstm_model.predict(test_window.test)
target_msd.shape
#Performance
x = np.arange(len(performance))
width = 0.3
metric_name = 'mean_absolute_error'
metric_index = lstm_model.metrics_names.index('mean_absolute_error')
val_mae = [v[metric_index] for v in val_performance.values()]
test_mae = [v[metric_index] for v in performance.values()]

plt.ylabel('mean_absolute_error [target, normalized]')
plt.bar(x - 0.17, val_mae, width, label='Validation')
plt.bar(x + 0.17, test_mae, width, label='Test')
plt.xticks(ticks=x, labels=performance.keys(),
           rotation=45)
_ = plt.legend()

for name, value in performance.items():
  print(f'{name:12s}: {value[1]:0.4f}')
