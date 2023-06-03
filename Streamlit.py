import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import Streamlit as st
from scipy.stats import mannwhitneyu, kstest, ttest_ind, beta

df = pd.read_csv('data.csv', sep=',', encoding='cp1251')  
df.rename(columns={'Количество больничных дней': 'work_days', 'Возраст': 'age', 'Пол' : 'sex'}, inplace=True)
df['sex'].replace(['М', 'Ж'], [0, 1], inplace=True)

male = df[df['sex'] == 0]['work_days']
female = df[df['sex'] == 1]['work_days']

st.title('Анализ гипотез о частоте пропуска более 2 рабочих дней')
st.markdown('Гипотеза 1: Мужчины пропускают в течение года более 2 рабочих дней по болезни значимо чаще женщин.')

plt.title('Histogram Density Function')
plt.hist(male, density=True, alpha=0.5, label='Sex = М', bins=9)
plt.hist(female, density=True, alpha=0.5, label='Sex = Ж', bins=9)
plt.xlabel('work_days')
plt.ylabel('Density')
plt.legend()
plt.show()