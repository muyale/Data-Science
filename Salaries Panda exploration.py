import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sal = pd.read_csv("C:\Datasets\Salaries.csv")
# sal.head(3)# sal.info()
base_pay = sal['BasePay']
# print(base_pay]
over_time_pay = sal['OvertimePay']
max_over_pay = over_time_pay.max()
print(max_over_pay)
print(sal.columns)
# lowest paid person
lowest = sal['TotalPayBenefits'].argmin
print(lowest)
by_year = sal.groupby('Year')
yearly_average = by_year.mean()
print(yearly_average)
unique = sal['JobTitle'].nunique()

# five most common jobs
print(sal['JobTitle'].value_counts().head(5))


# PEOPLE WITH THE WORD CHIEF IN THEIR TITLE
def chief_string(title):
    if 'chief ' not in title.lower:
        return False
    else:
        return True


chief = sum(sal['JobTitle'].apply(lambda x: chief_string(x)))

print(chief)
