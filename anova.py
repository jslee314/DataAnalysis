import scipy.stats as stats
import pandas as pd
import urllib
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt
import numpy as np

xles_name = 'totalData_id.xlsx'
brain = pd.read_excel(xles_name, sheet_name='kidney_anova')


colors = 'brg'

data_1 = brain['no_dl'].tolist()
data_2 = brain['yes_dl'].tolist()

data = list(range(0, len(data_1)))

plt.plot(data, data_1, '.', color=colors[0])
plt.plot(data, data_2, '.', color=colors[1])


plt.show()
print(data_1)

F_statistic, pVal = stats.f_oneway(data_1, data_2)

print('데이터의 일원분산분석 결과 : F={0:.1f}, p={1:.5f}'.format(F_statistic, pVal))
if pVal < 0.05:
    print('P-value 값이 충분히 작음으로 인해 그룹의 평균값이 통계적으로 유의미하게 차이납니다.')







#
# df = pd.DataFrame(data, columns=['value', 'treatment'])
# model = ols('value ~ C(treatment)', brain).fit()
# print(anova_lm(model))