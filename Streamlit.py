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

def checker(p_value, alpha):
    if p_value < alpha:
        return False
    else:
        return True

def print_answer(answer, type):
    if type == 'sex':
        if answer:
            st.write("Не отвергаем гипотезу о том, что частота пропусков мужчин и женщин одинаковая")
        else:
            st.write("Принимаем альтернативную гипотезу о том, что мужчины пропускают рабочие дни чаще")
    if type == 'age':
        if answer:
            st.write("Не отвергаем гипотезу о том, что частота пропусков людей разных возрастов одинаковая")
        else:
            st.write("Принимаем альтернативную гипотезу о том, что люди постарше пропускают рабочие дни чаще")       

def draw_hist(array1, label1, array2, label2):
    bin_size = 1
    min_edge = 0
    max_edge = 8
    N = int((max_edge - min_edge) / bin_size)
    Nplus1 = N + 1
    bin_list = np.linspace(min_edge, max_edge, Nplus1)

    fig = plt.figure(figsize=(15, 10))
    plt.title('Histogram Density Function')
    plt.hist(array1, density=True, alpha=0.5, label=label1, bins=bin_list)
    plt.hist(array2, density=True, alpha=0.5, label=label2, bins=bin_list)
    plt.xlabel('work_days')
    plt.ylabel('Density')
    plt.legend()
    return fig

def print_frequency(array1, label1, array2, label2):
    st.write("**Частота пропуска для**")
    st.write(f"{label1} : {sum(array1) / len(array1):.4f}")
    st.write(f"{label2} : {sum(array2) / len(array2):.4f}")
    st.write("##")

def do_test(test, test_name, pridicted, observed, alpha, type):
    stat, p_value = test(pridicted, observed, alternative='greater')
    st.write(f"**{test_name}**")
    st.write(f"statistic = {stat:.4f}, p-value = {p_value:.4f}")
    check = checker(p_value, alpha)
    print_answer(check, type)
    st.write("##")
        
def all_tests(pridicted, observed, alpha, type):
    
    do_test(mannwhitneyu, 'Mann–Whitney U Test', pridicted, observed, alpha, type)

    do_test(kstest, 'Kolmogorov-Smirnov Test', pridicted, observed, alpha, type)

    ab_test(observed, pridicted)

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

    st.write("**A/B Test**")
    st.write(f"Увеличение Conversion Rate на {lift*100:2.2f}% с вероятностью {prob*100:2.1f}%.")
    st.write("##")

def main():

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

    fig1 = draw_hist(male, 'Sex = М', female, 'Sex = Ж')
    st.pyplot(fig1)
    
    number_of_days = max(df['work_days'])
    s_days = st.slider('Задайте количество дней n в гипотезе 1', 0, number_of_days-1, 2)
    df['more_n_days'] = np.where(df['work_days'] > s_days, 1, 0)

    male = df[df['sex'] == 0]['more_n_days']
    female = df[df['sex'] == 1]['more_n_days']

    print_frequency(male, 'мужчин', female, 'женщин')
      
    s_alpha = st.slider('Задайте уровень значимости для проверки гипотозы 1', 0.0, 0.2, 0.05)
    all_tests(male, female, s_alpha, 'sex')
    
    
    st.markdown('Гипотеза 2: Работники старше m лет пропускают в течение года более n рабочих дней по болезни значимо чаще своих более молодых коллег.')
    
    a_days = st.slider('Задайте количество дней n в гипотезе 2', 0, number_of_days-1, 2)
    df['more_n_days'] = np.where(df['work_days'] > a_days, 1, 0)
    
    max_age = max(df['age'])
    min_age = min(df['age'])
    age = st.slider('Задайте граничное количество лет m в гипотезе 2', min_age, max_age-1, 35)

    old = df[df['age'] > age]['work_days']
    young = df[df['age'] <= age]['work_days']
    
    fig2 = draw_hist(old, 'age > ' + str(age), young, 'age <= ' + str(age))
    st.pyplot(fig2)
    
    old = df[df['age'] > 35]['more_n_days']
    young = df[df['age'] <= 35]['more_n_days']

    print_frequency(old, 'более взрослых людей', young, 'менее взрослых людей')
    
    a_alpha = st.slider('Задайте уровень значимости для проверки гипотозы 2', 0.0, 0.2, 0.05)
    all_tests(old, young, a_alpha, 'age')

if __name__ == "__main__":
    main()
