import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

uploaded_file = st.file_uploader('Выберите файл для анализа данных с расширением csv')

df = pd.read_csv(uploaded_file)  
df.rename(columns={'Количество больничных дней': 'work_days', 'Возраст': 'age', 'Пол' : 'sex'}, inplace=True)
df['sex'].replace(['М', 'Ж'], [0, 1], inplace=True)

male = df[df['sex'] == 0]['work_days']
female = df[df['sex'] == 1]['work_days']

st.title('Проверка гипотез')
st.markdown('Гипотеза 1: Мужчины пропускают в течение года более 2 рабочих дней по болезни значимо чаще женщин')

fig = plt.figure(figsize=(15, 10))
plt.title('Histogram Density Function')
plt.hist(male, density=True, alpha=0.5, label='Sex = М', bins=9)
plt.hist(female, density=True, alpha=0.5, label='Sex = Ж', bins=9)
plt.xlabel('work_days')
plt.ylabel('Density')
plt.legend()

st.pyplot(fig)
