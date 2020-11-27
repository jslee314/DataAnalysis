import pandas as pd
import numpy as np
import random

import matplotlib.pyplot as plt
# data_1 = [117, 108, 105, 89, 101, 93, 96, 108, 108, 94, 93, 112, 92, 91, 100, 96, 120, 86, 96, 95]
# data_2 = [121, 101, 102, 114, 103, 105, 101, 131, 96, 109, 109, 113, 115, 94, 108, 96, 110, 112, 120, 100]



xles_name = 'totalData_id.xlsx'
# serise = pd.read_excel(xles_name, sheet_name='tTest_kidney')
#
# first_x = 'normal_egfr'
# second_x = 'd_l_yes_egfr'
# y = 'egfr'

# serise = pd.read_excel(xles_name, sheet_name='tTest_kidney')
#
# first_x = 'normal_creatinine'
# second_x = 'd_l_Yes_creatinine'
# y = 'creatinine'

serise = pd.read_excel(xles_name, sheet_name='tTest_brain')

first_x = 'spoke_no_stress'
second_x = 'spoke_yes_stress'
y = 'stress'

original_data_1 = serise[first_x].dropna().tolist()
original_data_2 = serise[second_x].dropna().tolist()
plt.figure(figsize=(6, 6))
plt.grid()
plt.boxplot([original_data_1, original_data_2])
plt.xlabel(first_x + "  /  " + second_x)
plt.ylabel(y)
plt.title('original box plot')
plt.show()
print(first_x+"_original_data_1: " + str(np.mean(original_data_1)))
print(second_x+"+original_data_2: " + str(np.mean(original_data_2)))


if len(original_data_1) > len(original_data_2):
    data_1 = random.sample(original_data_1, len(original_data_2))
    data_2 = original_data_2
else:
    data_1 = original_data_1
    data_2 = random.sample(original_data_2, len(original_data_1))

# 두 집단의 평균 차이가 통계적으로 유의미한지 t-검증
# a, b: 두 집단
# equal_var: 두 집단의 variance가 같은지, 다른지를 측정함. True일 경우는 같다고, False일 경우에는 다르다고 하며, 다른 테스트를 수행함.
# Returns : statistic : The calculated t-statistic.
#           p-value   : The two-tailed p-value.

# H0(귀무가설 = 영가설 = 차이가 없다라는 가설)
# p-value가 클수록 귀무가설이 타당할 가능성이 높다. 즉 두 집단간에 차이가 없을 가능성 높음
import scipy.stats
statistic, p_value = scipy.stats.ttest_rel(
    a=data_1, b=data_2)
print("=="*30)
print("=="*30)
print(f"statistic: {statistic:.3f}")
print(f"p_value  : {p_value:.3f}")
print("=="*30)
print("유의수준(오차허용범위)를 5%로 정하였을 때 ")
print("p-value 가 0.05 보다 크기때문에 '두 집단간에 차이가 없다'라는 귀무가설은 '타당'하다. (귀무가설 채택)")
print("p-value 가 0.05 보다 크기때문에 홍채 패턴에 따른 평균 차이는 유의미하지 않다고 할 수 있다.")

print("data_1: " + str(np.mean(data_1)))
print("data_2: " + str(np.mean(data_2)))
print(str(np.std(data_1)))
print(str(np.std(data_2)))

plt.figure(figsize=(6, 6))
plt.grid()
plt.boxplot([data_1, data_2])
plt.xlabel("no pattern" + "                             " + "Lacuna/Defect pattern")
plt.ylabel(y)
plt.title('Iris Box Plot')
plt.show()




