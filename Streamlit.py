import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

from scipy.stats import mannwhitneyu, kstest, ttest_ind, beta

sns.set()
rcParams['figure.figsize'] = 10, 6
np.random.seed(42)

st.title('Проверка гипотез о частоте пропуска более 2 рабочих дней')

uploaded_file = st.file_uploader('Выберите файл для анализа данных с расширением csv')

if uploaded_file is not None:
  df = pd.read_csv(uploaded_file, sep=',', encoding='cp1251')
  df.rename(columns={'Количество больничных дней': 'work_days', 'Возраст': 'age', 'Пол' : 'sex'}, inplace=True)
  df['sex'].replace(['М', 'Ж'], [0, 1], inplace=True)

  male = df[df['sex'] == 0]['work_days']
  female = df[df['sex'] == 1]['work_days']

  st.markdown('Гипотеза 1: Мужчины пропускают в течение года более n рабочих дней по болезни значимо чаще женщин')
  
  number_of_days = np.max(df['work_days'])
    days = st.slider('Задайте количество дней n в гипотезе', 0, 10, 10)

  fig = plt.figure(figsize=(15, 10))
  plt.title('Histogram Density Function')
  plt.hist(male, density=True, alpha=0.5, label='Sex = М', bins=9)
  plt.hist(female, density=True, alpha=0.5, label='Sex = Ж', bins=9)
  plt.xlabel('work_days')
  plt.ylabel('Density')
  plt.legend()

  st.pyplot(fig)
  
  df['more_2_days'] = np.where(df['work_days'] > days, 1, 0)

  male = df[df['sex'] == 0]['more_n_days']
  female = df[df['sex'] == 1]['more_n_days']
  
  st.write("**Частота пропуска для**")
  st.write(f"мужчин : {sum(male) / len(male):.4f}")
  st.write(f"женщин : {sum(female) / len(female):.4f}")
  st.write("##")
  
  stat, p_value = mannwhitneyu(male, female, alternative='greater', method='exact')
  st.write("**Mann–Whitney U Test**")
  st.write(f"statistic = {stat:.4f}, p-value = {p_value:.4f}")
  st.write("##")
  
  stat, p_value = kstest(male, female, alternative='greater', method='exact')
  st.write("**Kolmogorov-Smirnov Test**")
  st.write(f"statistic = {stat:.4f}, p-value = {p_value:.4f}")
  st.write("##")
  
  sample_stat = np.mean(male) - np.mean(female)
  stats = np.zeros(1000)
  for k in range(1000):
    labels = np.random.permutation((df['sex'] == 0).values)
    stats[k] = np.mean(df.more_n_days[labels]) - np.mean(df.more_n_days[labels==False])
  p_value = np.mean(stats > sample_stat)
  st.write("**Permutation Test**")
  st.write(f"p-value = {p_value:.4f}")
  st.write("##")
