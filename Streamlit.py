import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from math import lgamma

from scipy.stats import mannwhitneyu, kstest, ttest_ind, beta

def load_data(uploaded_file):
  df = pd.read_csv(uploaded_file, sep=',', encoding='cp1251')
  df.rename(columns={'Количество больничных дней': 'work_days', 'Возраст': 'age', 'Пол' : 'sex'}, inplace=True)
  df['sex'].replace(['М', 'Ж'], [0, 1], inplace=True)
  return df

def sex_checker(p_value, alpha):
  if p_value < alpha:
    st.write("Принимаем альтернативную гипотезу о том, что мужчины пропускают рабочие дни чаще")
  else:
    st.write("Не отвергаем гипотезу о том, что частота пропусков мужчин и женщин одинаковая")
    
def age_checker(p_value, alpha):
  if p_value < alpha:
    st.write("Принимаем альтернативную гипотезу о том, что люди постарше пропускают рабочие дни чаще")
  else:
    st.write("Не отвергаем гипотезу о том, что частота пропусков людей разных возрастов одинаковая")  
    
def all_tests(pridicted, observed, alpha):
  stat, p_value = mannwhitneyu(pridicted, observed, alternative='greater', method='exact')
  st.write("**Mann–Whitney U Test**")
  st.write(f"statistic = {stat:.4f}, p-value = {p_value:.4f}")
  sex_checker(p_value, alpha)
  st.write("##")
  
  stat, p_value = kstest(pridicted, observed, alternative='greater', method='exact')
  st.write("**Kolmogorov-Smirnov Test**")
  st.write(f"statistic = {stat:.4f}, p-value = {p_value:.4f}")
  sex_checker(p_value, alpha)
  st.write("##")

  st.write("**A/B Test**")
  ab_test(observed, pridicted)
  st.write("##")

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

sns.set()
rcParams['figure.figsize'] = 10, 6
np.random.seed(42)

st.title('Проверка гипотез о частоте пропуска более 2 рабочих дней')

uploaded_file = st.file_uploader('Выберите файл для анализа данных с расширением csv')

if uploaded_file is not None:
  df = load_data(uploaded_file)

  male = df[df['sex'] == 0]['work_days']
  female = df[df['sex'] == 1]['work_days']

  st.markdown('Гипотеза 1: Мужчины пропускают в течение года более n рабочих дней по болезни значимо чаще женщин.')

  fig = plt.figure(figsize=(15, 10))
  plt.title('Histogram Density Function')
  plt.hist(male, density=True, alpha=0.5, label='Sex = М', bins=9)
  plt.hist(female, density=True, alpha=0.5, label='Sex = Ж', bins=9)
  plt.xlabel('work_days')
  plt.ylabel('Density')
  plt.legend()
  st.pyplot(fig)
  
  number_of_days = max(df['work_days'])
  s_days = st.slider('Задайте количество дней n в гипотезе', 0, number_of_days-1, 2)
  df['more_n_days'] = np.where(df['work_days'] > s_days, 1, 0)

  male = df[df['sex'] == 0]['more_n_days']
  female = df[df['sex'] == 1]['more_n_days']
  
  st.write("**Частота пропуска для**")
  st.write(f"мужчин : {sum(male) / len(male):.4f}")
  st.write(f"женщин : {sum(female) / len(female):.4f}")
  st.write("##")
    
  s_alpha = st.slider('Задайте уровень значимости для проверки гипотозы', 0.0, 0.2, 0.05)
  all_tests(male, female, s_alpha)
  
  
  st.markdown('Гипотеза 2: Работники старше m лет пропускают в течение года более n рабочих дней по болезни значимо чаще своих более молодых коллег.')
  
  a_days = st.slider('Задайте количество дней n в гипотезе', 0, number_of_days-1, 2)
  df['more_n_days'] = np.where(df['work_days'] > a_days, 1, 0)
  
  max_age = max(df['age'])
  age = st.slider('Задайте граничное количество лет m в гипотезе', 0, max_age-1, 35)
  old = df[df['age'] > age]['work_days']
  young = df[df['age'] <= age]['work_days']
  
  plt.title('Histogram Density Function')
  plt.hist(old, density=True, alpha=0.5, label='age > ' + str(age), bins=9)
  plt.hist(young, density=True, alpha=0.5, label='age <= ' + str(age), bins=9)
  plt.xlabel('work_days')
  plt.ylabel('Density')
  plt.legend()
  plt.show()
  
  st.write("**Частота пропуска для**")
  st.write(f"более взрослых людей : {sum(old) / len(old):.4f}")
  st.write(f"менее взрослых людей : {sum(young) / len(young):.4f}")
  st.write("##")
  
  a_alpha = st.slider('Задайте уровень значимости для проверки гипотозы', 0.0, 0.2, 0.05)
  all_tests(old, young, a_alpha)
