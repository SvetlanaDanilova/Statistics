import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from math import lgamma

from scipy.stats import mannwhitneyu, kstest, ttest_ind, beta

def h(a, b, c, d):
    num = lgamma(a + c) + lgamma(b + d) + lgamma(a + b) + lgamma(c + d)
    den = lgamma(a) + lgamma(b) + lgamma(c) + lgamma(d) + lgamma(a + b + c + d)
    return np.exp(num - den)

def g0(a, b, c):    
    return np.exp(lgamma(a + b) + lgamma(a + c) - (lgamma(a + b + c) + lgamma(a)))

def hiter(a, b, c, d):
    while d > 1:
        d -= 1
        yield h(a, b, c, d) / d

def g(a, b, c, d):
    return g0(a, b, c) + sum(hiter(a, b, c, d))

def calc_prob_between(beta1, beta2):
    return g(beta1.args[0], beta1.args[1], beta2.args[0], beta2.args[1])

def ab_test(old, new):
    #This is the known data: impressions and conversions for the Control and Test set
    imps_ctrl, convs_ctrl = len(old), sum(old)
    imps_test, convs_test = len(new), sum(new)

    #here we create the Beta functions for the two sets
    a_C, b_C = convs_ctrl+1, imps_ctrl-convs_ctrl+1
    beta_C = beta(a_C, b_C)
    a_T, b_T = convs_test+1, imps_test-convs_test+1
    beta_T = beta(a_T, b_T)

    #calculating the lift
    lift=(beta_T.mean()-beta_C.mean())/beta_C.mean()

    #calculating the probability for Test to be better than Control
    prob=calc_prob_between(beta_T, beta_C)

    st.write(f"Test option lift Conversion Rates by {lift*100:2.2f}% with {prob*100:2.1f}% probability.")
    st.write("##")

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

  fig = plt.figure(figsize=(15, 10))
  plt.title('Histogram Density Function')
  plt.hist(male, density=True, alpha=0.5, label='Sex = М', bins=9)
  plt.hist(female, density=True, alpha=0.5, label='Sex = Ж', bins=9)
  plt.xlabel('work_days')
  plt.ylabel('Density')
  plt.legend()

  st.pyplot(fig)
  
  number_of_days = max(df['work_days'])
  days = st.slider('Задайте количество дней n в гипотезе', 0, number_of_days, number_of_days)
  
  df['more_n_days'] = np.where(df['work_days'] > days, 1, 0)

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

  ab_test(female, male)
