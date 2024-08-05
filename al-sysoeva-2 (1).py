#!/usr/bin/env python
# coding: utf-8

# # ФИНАЛЬНЫЙ ПРОЕКТ

# # ЗАДАНИЕ 1

# В ходе A/B–тестирования одной гипотезы целевой группе была предложена новая механика оплаты услуг на сайте, у контрольной группы оставалась базовая механика. В качестве задания необходимо проанализировать итоги эксперимента и сделать вывод, стоит ли запускать новую механику оплаты на всех пользователей.
# 
# Имеестся 4 csv-файла:
# 
# - groups.csv - файл с информацией о принадлежности пользователя к контрольной или экспериментальной группе (А – контроль, B – целевая группа).
# - groups_add.csv - дополнительный файл с пользователями, который вам прислали спустя 2 дня после передачи данных.
# - active_studs.csv - файл с информацией о пользователях, которые зашли на платформу в дни проведения эксперимента. 
# - checks.csv - файл с информацией об оплатах пользователей в дни проведения эксперимента. 

# В ходе выполнения проекта планируется ответить на следующие вопросы:
# 
# - На какие метрики Вы смотрите в ходе анализа и почему?
# - Имеются ли различия в показателях и с чем они могут быть связаны?
# - Являются ли эти различия статистически значимыми?
# - Стоит ли запускать новую механику на всех пользователей?

# ## 1. Подготовка библиотек

# In[96]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
import scikit_posthocs as sp
import pingouin as pg
from tqdm.auto import tqdm
from scipy.stats import norm

import requests
from urllib.parse import urlencode
import json


# ## 2. Чтение файлов

# In[2]:


groups = pd.read_csv('Проект_2_groups.csv', sep = ";")


# In[3]:


groups


# In[4]:


groups_add = pd.read_csv('Проект_2_group_add.csv', sep = ",")


# In[5]:


groups_add


# In[6]:


active_studs = pd.read_csv('Проект_2_active_studs.csv')


# In[7]:


active_studs


# In[8]:


checks = pd.read_csv('Проект_2_checks.csv', sep = ";")


# In[9]:


checks


# In[10]:


# по результатам ревью проекта - вариант с загрузкой файлов через API

base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
public_key = 'https://disk.yandex.ru/client/disk?source=main-loginmenu'  
            
public_key_groups = 'https://disk.yandex.ru/d/bCDO-nYqqALthw'
public_key_active_studs = 'https://disk.yandex.ru/d/lJIahE6Y_mpQsw'
public_key_checks = 'https://disk.yandex.ru/d/9Rab4HMISynx_A'

final_url_groups = base_url + urlencode(dict(public_key=public_key_groups))  
response_groups = requests.get(final_url_groups)  
download_url_groups =  response_groups.json()['href']  

groups_ = pd.read_csv(download_url_groups, sep=';')  
final_url_active_studs = base_url + urlencode(dict(public_key=public_key_active_studs))
response_active_studs = requests.get(final_url_active_studs)
download_url_active_studs = response_active_studs.json()['href']

active_studs_ = pd.read_csv(download_url_active_studs, sep=';')
final_url_checks = base_url + urlencode(dict(public_key=public_key_checks))
response_checks = requests.get(final_url_checks)
download_url_checks = response_checks.json()['href']
checks_ = pd.read_csv(download_url_checks, sep=';')


# ## 3. Предобработка

# In[11]:


# соединим два файла с общим содержанием 
groups = groups.merge(groups_add, how='outer')


# In[12]:


groups


# In[13]:


#посчитаем количество наблюдений в каждой группе
groups.groupby('grp').size()


# In[14]:


# посчитаем количество уникальных пользователей по группам
groups.groupby('grp').id.nunique()


# Можем уже сразу сказать о неравномерности выборок, а также о том, что все наши пользователи уникальные.

# In[15]:


# уберем дубликаты


# In[16]:


groups = groups.drop_duplicates()
groups.shape


# In[17]:


active_studs = active_studs.drop_duplicates()
active_studs.shape


# In[18]:


checks = checks.drop_duplicates()
checks.shape


# In[19]:


# дубликаты не обнаружились


# In[20]:


# найдем пропуски в данных


# In[21]:


groups.isna().sum()


# In[22]:


active_studs.isna().sum()


# In[23]:


checks.isna().sum()


# In[24]:


# пропусков в данных так же нет


# In[25]:


# поищем аномалии в данных


# In[26]:


groups.info()


# In[27]:


active_studs.info()


# In[28]:


checks.info()


# In[29]:


# сделаем проверку на аномалии в данных
groups['grp'].unique()


# In[30]:


checks['rev'].unique()


# Аномалий так же не нашлось: групп тестирования две, как и должно быть, оплат с нулевой или отрицательной суммой также нет

# In[31]:


# узнаем общую статистическую информацию об оплатах в checks
round(checks.rev.describe(), 2)


# In[32]:


checks.rev.hist(figsize=(10, 10));


# Видим очень неравномерное распределение оплат, при этом большая часть оплат лежит в границах до 2000. Среднее оплаты находится на отметке в 1059,75. При этом 50% данных находятся на отметке 840, что говорит о том, что данные имеют выбросы.

# In[33]:


# построим боксплот чтобы исследовать данные оплат
fig, ax = plt.subplots(figsize = (17,2))
ax = checks[['rev']].boxplot(vert = False, ax =ax)
ax.set_title('Диаграмма размаха значений оплат пользователей в дни проведения эксперимента');


# Видим, что имеется крупный выброс в районе 4500, который тянет за собой смещение уса (доверительного интервала) диаграммы.

# ## 4. Проведение исследования

# #### 4.1. Объединим датасеты для проведения исследования

# In[34]:


# соберем данные генеральной совокупности: объединим сведения из датасетов с сохранением информации по принадлежности к группе
# при этом мы сохраняем только те данные, которые соотносятся с информацией в active_studs, то есть относятся к пользователям,
# которые заходили на сайт во время проведения эксперимента. Таким образом мы создаем стратифицированную выборку
df = pd.merge(groups, active_studs, how='right', left_on='id', right_on='student_id')  
df = pd.merge(df, checks, how='left', on = 'student_id')


# In[35]:


df


# In[36]:


# проверим количество пустых значений
df.isna().sum()


# Видим большое количество пропусков в графе с оплатой. Это означает, что посетитель зашел на сайт, но не совершил оплату, при этом он  участвует в эксперименте. Также количество информации об оплате уменьшилось на этапе слияния датасетов. Видимо, часть информации об оплате относилась к клиентам, которые выполнили операции вне эксперимента.

# ##### 4.1.1. Сделаем проверку на наличие неактивных пользователей с покупкой

# In[37]:


# создадим столбец с отметкой активности пользователя 
df = df.assign(action=df.apply(lambda x: 'active' if (x.id in active_studs['student_id'].to_list()) else 'passive', axis=1))


# In[38]:


# проверим активность в получившемся столбце
df['action'].unique()


# Получается, что у нас нет данных о неактивных пользователях, совершивших оплату.

# #### 4.2. Сделаем разбивку данных по контрольной и экспериментальной группе

# In[39]:


# отсеим данные для контрольной группы
df_a = df.query('grp == "A"')


# In[40]:


df_a


# In[41]:


# теперь отберем данные для тестовой группы
df_b = df.query('grp == "B"')


# In[42]:


df_b


# In[43]:


# посмотрим на общем графике соотношение полученной прибыли между группами
sns.histplot(data= df, x='rev', hue='grp');


# Как и раньше, видим большой перевес в сторону количества пользователей, участвовавших в тестировании. Скорее всего, здесь имеются ошибка в сплитовании пользователей, так как группы по объему выборки различаются почти в 5 раз. Сделать вывод о том, что новая механика приносит больше прибыли мы пока не можем.

# #### 4.3. Посмотрим распределение оплат по группам

# In[44]:


round(df_a.rev.describe(), 2)


# In[45]:


round(df_b.rev.describe(), 2)


# Видим, что в контрольной группе средний чек немного больше, а так же что максимальное значение, представляющее собой выброс так же находится в тестовой группе. Кроме того, значительно увеличилось количество оплат в тестовой группе.

# In[46]:


# построим графики


# In[47]:


df_a.rev.hist(figsize=(10, 10));


# In[48]:


df_b.rev.hist(figsize=(10, 10));


# In[49]:


# потроим боксплот распределения покупок для каждой группы
plt.figure(figsize=(20, 10))
sns.boxplot(x='grp', y='rev', data=df.query('rev!=0'))
plt.xlabel('Группы')
plt.ylabel('Прибыль')
plt.title('Распределение прибыли по группам')
plt.show()


# Данные все так же распределены неравномерно, графики подверждают сведения, полученные при статистическом описании.

# #### 4.4. Рассчитаем основные метрики

# Имея те данные, которые у нас есть, можем рассчитать следующие метрики продукта: CR, ARPAU и ARPPU.
# 
# **CR** представляет собой конверсию пользователей в целевое действие, в нашем случае это совершение покупки. Рассчитывается как отношение числа пользователей, которые выполнили какое-либо целевое действие к общему числу пользователей.
# 
# **ARPAU** представляет собой средний доход с привлечённого пользователя. Рассчитывается по формуле: **ARPU= Доход/Число пользователей**.
# 
# **ARPPU** представляет собой средний доход на платящего пользователя. Рассчитывается по формуле: **ARPPU=Доход * Число платящих пользователей**.

# In[50]:


# создадим табличку для расчета метрик
# для расчетов сгруппируем данные по контрольным группам, посчитаем общее количество пользователей и количество плативших пользователей
metricks = df.groupby('grp')            .agg({'id':'nunique', 'rev': [lambda x: x.count(),'sum']}).droplevel(0, axis=1).round(2)            .rename(columns={'nunique':'users','<lambda_0>': 'payed_users', 'sum':'revenue'}).reset_index()


# In[51]:


# теперь сделаем собственно расчет этих метрик
metricks = metricks.assign(CR = round((metricks.payed_users/metricks.users)*100,2),
                           ARPAU = round((metricks.revenue/metricks.users),2),
                           ARPPU = round((metricks.revenue/metricks.payed_users),2))


# In[52]:


# проверим что получилось
metricks


# In[53]:


# посмотрим на метрики на графиках
fig.tight_layout(h_pad=5)
fig, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize = (20, 6))
ax_CR = sns.barplot(data=metricks,y='CR', x='grp', ax=ax1)
ax_CR.set_title('Конверсия в покупку')
ax_CR.set_xlabel('Группы')
ax_CR.set_ylabel('CR, %')
ax_ARPAU = sns.barplot(data=metricks,y='ARPAU', x='grp', ax=ax2)
ax_ARPAU.set_title('Средний платеж активных пользователей')
ax_ARPAU.set_xlabel('Группы')
ax_ARPPU = sns.barplot(data=metricks,y='ARPPU', x='grp', ax=ax3)
ax_ARPPU.set_title('Средний платеж покупателей')
ax_ARPPU.set_xlabel('Группы')
sns.despine()


# По результатам расчета метрик можем отметить следующее:
# - конверсия в покупку снизилась после применения нового механизма оплаты
# - при этом как ARPPU, так и ARPAU выросли в тестовой группе.

# #### 4.5. Проверим несколько гипотез относительно имеющихся данных

# ##### 4.5.1.  Гипотеза 1: Но -  конверсия в покупку после изменения механики оплаты осталась прежней, Н1 - конверсия в покупку изменилась

# In[54]:


# создадим столбец на основе столбца с оплатой
df['rvn'] = df['rev']


# In[55]:


# заполним значения пропусков на 0 - отсутствие оплаты
df['rvn'] = df['rvn'].fillna(0) 


# In[56]:


# сделаем значения в столбце 1 - наличие оплаты, 0 - отсутсвие и применим к столбцу

def revenue(x):
    if x > 0:
        return 1
    else:
        return 0
    
    
df['rvn'] = df['rev'].apply(revenue)


# In[57]:


# проверим что получилось
df.head(10)


# In[58]:


# проверим кросстабуляцию
crosstab = pd.crosstab(df.grp, df.rvn)
crosstab


# Видим, что в контрольной группе 1460 значений без проведения оплаты, и только 78 - с оплатой. В тестовой группе без оплаты 6489 значений, с оплатой - 314. Сделать заключение, о том, что соотношение изменилось, все еще рано. 
# 
# С помощью теста на хи-квадрат, который мы выбираем так как у нас имеются две независимые категориальные переменные А и В, узнаем, является ли изменение пропорции статистически значимым.

# In[59]:


exp, obs, stats = pg.chi2_independence(data = df, x='grp', y='rvn')


# In[60]:


stats


# p-value заметно больше 0,05, из чего мы делаем вывод, что статистически значимого изменения соотношения оплаты после проведения эксперимента не случилось или иными словами, у нас нет оснований отвергнуть нулевую гипотезу.

# ##### 4.5.2.  Гипотеза 2: Но -  статистических различий между средним платежем для пользователя между группами нет, Н1 - имеются статистические различия между средним платежом для пользователей между группами (стат проверка ARPAU)

# In[107]:


# посмотрим на данные на графиках
fig.tight_layout(h_pad=5)
fig, (ax1,ax2,ax3) = plt.subplots(3, 1, figsize = (10, 25))

#график распределения прибыли по пользователям обеих групп
ax_1 = sns.histplot(data=df, x='rev', hue = 'grp', ax=ax1)
ax_1.set_title('Распределение прибыли по кол-ву пользователей')
ax_1.set_xlabel('Прибыль')
ax_1.set_ylabel('Кол-во пользователей')
#график распределения прибыли по пользователям контрольной группы
ax_2 = sns.histplot(data=df, x='rev', ax=ax2)
ax_2.set_title('Распределение прибыли по кол-ву пользователей в контрольной группе')
ax_2.set_xlabel('Прибыль')
ax_2.set_ylabel('Кол-во пользователей')
#график распределения прибыли по пользователям тестовой группы
ax_3 = sns.histplot(data=df, x='rev', ax=ax3)
ax_3.set_title('Распределение прибыли по кол-ву пользователей в тестовой группе')
ax_3.set_xlabel('Прибыль')
ax_3.set_ylabel('Кол-во пользователей')

sns.despine()


# In[104]:


# используем бутстрап для оценки наличия различий в метрике ARPAU
def get_bootstrap(
    kontrol, # числовые значения выборки для контрольной группы
    test, # числовые значения выборки для тестовой группы
    boot_it = 1000, # количество подвыборок
    statistic = np.mean, # среднее
    bootstrap_conf_level = 0.95 # уровень значимости
):
    boot_len = max([len(kontrol), len(test)])
    boot_data = []
    for i in tqdm(range(boot_it)): # извлекаем подвыборки
        samples_1 = kontrol.sample(
            boot_len, 
            replace = True # параметр возвращения
        ).values
        
        samples_2 = test.sample(
            boot_len, # чтобы сохранить дисперсию, берем такой же размер выборки
            replace = True
        ).values
        
        boot_data.append(statistic(samples_2-samples_1)) 
    pd_boot_data = pd.DataFrame(boot_data)
        
    left_quant = (1 - bootstrap_conf_level)/2
    right_quant = 1 - (1 - bootstrap_conf_level) / 2
    quants = pd_boot_data.quantile([left_quant, right_quant])
        
    p_1 = norm.cdf(
        x = 0, 
        loc = np.mean(boot_data), 
        scale = np.std(boot_data)
    )
    p_2 = norm.cdf(
        x = 0, 
        loc = -np.mean(boot_data), 
        scale = np.std(boot_data)
    )
    p_value = min(p_1, p_2) * 2
        
    # Визуализация
    _, _, bars = plt.hist(pd_boot_data[0], bins = 50)
    for bar in bars:
        if abs(bar.get_x()) <= quants.iloc[0][0] or abs(bar.get_x()) >= quants.iloc[1][0]:
            bar.set_facecolor('red')
        else: 
            bar.set_facecolor('grey')
            bar.set_edgecolor('black')
    
    
    plt.vlines(quants,ymin=0,ymax=50,linestyle='--')
    plt.xlabel('Выборка')
    plt.ylabel('Частота')
    plt.title("Гистограмма распределения бутстрап")
    plt.show()
       
    return {"quants": quants, 
            "p_value": p_value}


# In[105]:


# создадим набор данных по группам
data_a = df.query('grp == "A"')
data_b = df.query('grp == "B"')


# In[106]:


# проведем само исследование
get_bootstrap(
    data_a.rev, # контрольная группа
    data_b.rev, # тестовая группа
    boot_it = 1000, # количество бутстрап-подвыборок
    statistic = np.mean, # интересующая нас статистика
    bootstrap_conf_level = 0.95 # уровень значимости
)


# Видим, что p-value меньше 0.05. Таким образом, мы можем отклонить нулевую гипотезу и заключить, что имеются статистически значимые различия в ARPAU контрольной и тестовой групп.

# ##### 4.5.3.  Гипотеза 3: Но -  статистических различий между средним платежем для пользователя, совершившего оплату, между группами нет, Н1 - имеются статистические различия между средним платежом таких пользователей (стат проверка ARPPU)

# In[109]:


# проверим данные на нормальность
pg.normality(data=df.query("rvn_mark =='True'"),             dv='rev', group = 'grp', method = 'normaltest')


# In[110]:


# посмотрим на данные на графиках
fig.tight_layout(h_pad=5)
fig, (ax1,ax2,ax3) = plt.subplots(3, 1, figsize = (10, 25))

#график распределения прибыли по пользователям обеих групп
ax_1 = sns.histplot(data=df.query("rvn_mark =='True'"), x='rev', hue = 'grp', ax=ax1)
ax_1.set_title('Распределение прибыли по кол-ву пользователей')
ax_1.set_xlabel('Прибыль')
ax_1.set_ylabel('Кол-во пользователей')
#график распределения прибыли по пользователям контрольной группы
ax_2 = sns.histplot(data=df.query("rvn_mark =='True'"), x='rev', ax=ax2)
ax_2.set_title('Распределение прибыли по кол-ву пользователей в контрольной группе')
ax_2.set_xlabel('Прибыль')
ax_2.set_ylabel('Кол-во пользователей')
#график распределения прибыли по пользователям тестовой группы
ax_3 = sns.histplot(data=df.query("rvn_mark =='True'"), x='rev', ax=ax3)
ax_3.set_title('Распределение прибыли по кол-ву пользователей в тестовой группе')
ax_3.set_xlabel('Прибыль')
ax_3.set_ylabel('Кол-во пользователей')

sns.despine()


# По графикам и проверке на нормальность видим, что данные не распределены нормально, поэтому для проверки гипотез снова воспользуемся Bootstrap

# In[115]:


# создадим набор данных по группам
data_aa = df.query('rvn_mark == "True"')['rev']
data_bb = df.query('rvn_mark == "True"')['rev']


# In[116]:


get_bootstrap(
    data_aa, 
    data_bb, 
    boot_it = 1000, 
    statistic = np.mean, 
    bootstrap_conf_level = 0.95 
)


# Значение доверительного интервала (от 197 до 453) не пересекает ноль ,  поэтому гипотеза Н0 - отклоняется. А p_value - сильно меньше 0,05. Делаем вывод об имеющихся статистически значимых различиях.

# ## Выводы

# Итак, мы провели предобработку данных и в ходе анализа обратили внимание на такие показатели, как уникальный номер пользователя и совершенной покупки, так как эти данные являются ключевыми в вопросе поведения пользователя и получения бизнесом прибыли.
# Для проведения исследования мы сформулировали гипотезы, которые проверяли при помощи p-value. Чем меньше p-значение, тем больше оснований отклонить нулевую гипотезу.
# Коэффициент конверсии пользователей только уменьшился с 5,07% до 4,62%, что говорит от том, что пользователи менее охотно начали делать оплату. Несмотря на это статистически значимо увеличился показатель ARPPU и ARPPU. 
# Учитывая изложенное можно предположить, что отказываться от идеи изменения механики оплаты не стоит, тем не менее, стоит поработать над увеличением конверсии.

# # ЗАДАНИЕ 2

# ##### 2.1 Очень усердные ученики.
# ##### 2.1.2 Задача
# Необходимо написать оптимальный запрос, который даст информацию о количестве очень усердных студентов.NB! Под усердным студентом мы понимаем студента, который правильно решил 20 задач за текущий месяц.

# In[69]:


import pandahouse as ph


# In[70]:


# подключимся к кликхаусу
clickhouse = dict(database='default',
                  host='https://clickhouse.lab.karpov.courses',
                  user='student',
                  password='dpo_python_2020')


# In[71]:


first_task = """ 
SELECT   
        COUNT(st_id) --собственно считаем студентов
        
FROM (
        SELECT st_id  -- создаем подзапрос с условием по месяцу, правильности выполнения задач и их количеству
        FROM default.peas
        WHERE 
                toMonth(timest) = '10' 
                and correct = '1'
        GROUP BY st_id
        HAVING COUNT(correct) >= 20
        )
"""


# In[72]:


students = ph.read_clickhouse(first_task, connection=clickhouse)
print("Количество очень усердных студентов:", students.iat[0, 0])


# ##### 2.2 Оптимизация воронки
# Необходимо в одном запросе выгрузить следующую информацию о группах пользователей:
# - ARPU 
# - ARPAU 
# - CR в покупку 
# - СR активного пользователя в покупку 
# - CR пользователя из активности по математике (subject = ’math’) в покупку курса по математике
# - ARPU считается относительно всех пользователей, попавших в группы.
# 
# Активным считается пользователь, за все время решивший больше 10 задач правильно в любых дисциплинах.
# 
# Активным по математике считается пользователь, за все время решивший 2 или больше задач правильно по математике.

# In[73]:


second_task = """ 
SELECT
    grp AS group,
    (SUM(money) / COUNT(money)) AS ARPU,
    (SUM(money_10) / SUM(score > 10)) AS ARPAU,
    (SUM(money > 0) / COUNT(money)) AS CR,
    (SUM(money > 0 and score > 10) / SUM(score > 10)) AS CR_10,
    (SUM(math_pay = 1) / SUM(math = 1)) AS CR_math,
    --Значения для проверки расчетов метрик
    COUNT(money) AS user_count, --Всего пользователей
    SUM(money > 0) AS user_payed, --Количество пользователей, которые совершили оплату
    SUM(money = 0) AS user_free, --Количество пользователей, которые не совершали оплату
    SUM(score > 10) AS user_10, --Количество пользователей, решивших более 10 задач правильно
    SUM(money > 0 and score > 10) AS user_payed_10, --Количество пользователей, решивших более 10 задач правильно и совершивших оплату
    SUM(money) AS money_sum, --Сумма всех оплат
    SUM(money_10) AS money_10_sum, --Сумма всех оплат пользователей, решивших более 10 задач правильно
    SUM(math = 1) AS user_math, --Количество пользователей, решивших 2 и более задач правильно по математике
    SUM(math_pay = 1) AS user_math_pay --Количество пользователей, оплативших математику
FROM (
    SELECT
        a.st_id AS id,
        a.test_grp AS grp,
        b.money AS money,
        c.score AS score,
        CASE WHEN c.score > 10 THEN b.money ELSE 0 END AS money_10,
        b.math_pay AS math_pay,
        c.math AS math
    FROM default.studs AS a
LEFT JOIN (
        SELECT
            st_id,
            SUM(money) AS money,
            MAX(CASE WHEN subject = 'Math' THEN 1 ELSE 0 END) AS math_pay
        FROM default.final_project_check 
        GROUP BY st_id) AS b 
ON a.st_id = b.st_id
LEFT JOIN (
        SELECT
            st_id,
            SUM(score) AS score,
            SUM(math) AS math
        FROM (
            SELECT
                st_id,
                SUM(correct) AS score,
                CASE WHEN subject = 'Math' and score >= 2 THEN 1 ELSE 0 END AS math
            FROM default.peas
            GROUP BY
                st_id,
                subject)
        GROUP BY st_id) AS c
ON a.st_id = c.st_id)
GROUP BY grp
"""
metrics = ph.read_clickhouse(second_task, connection=clickhouse)
metrics.round(2).T


# # ЗАДАНИЕ 3

# 3.1 Задача
# 
# Реализуйте функцию, которая будет автоматически подгружать информацию из дополнительного файла groups_add.csv (заголовки могут отличаться) и на основании дополнительных параметров пересчитывать метрики.
# 
# 

# В функции мы:
# - загрузим три основных датасета и объединим их
# - оформим названия колонок без повторов
# - подгрузим дополнительный датасет `groups_add.csv` с обновляемой информацией о пользователях
# - сформируем таблицу
# - расчитаем метрики CR и ARPPU

# Для начала снова загрузим файлы из хранилища, дадим им новые имена, а также преобразуем повторяющиеся колонки с одним наименованием для того, чтобы в дальнейшем было удобно делать слияние таблиц. Затем произведем объединение таблиц в общий датасет, создадим столбец с отметкой о наличии оплаты. Из общего датасета соберем таблицу с коротким выводом по эксперименту, на основании которой расчитаем необходимые метрики.

# In[74]:


def groups_add_func():
    df_groups = pd.read_csv('Проект_2_groups.csv', sep = ";")
    df_users  = pd.read_csv('Проект_2_active_studs.csv',sep = ";", names=['id'], header = 0)
    df_checks = pd.read_csv('Проект_2_checks.csv', sep = ";", names=['id', 'rev'], header = 0)

# попробуем обновить данные при наличии нового файла    
    try:
        df_add = pd.read_csv('Проект_2_group_add.csv', sep = ",")
        df_groups = df_groups.merge(df_add, how='outer')
        print('Были добавлены новые данные.\n')
    except:
        print('Нет файла для обновления.\n')
     
 # получим общую таблицу с данными        
    data = pd.merge(df_groups, df_users, how='right')  
    data = pd.merge(data, df_checks, how='left')
    data.fillna(value=0, inplace=True) 
    data['rvn_mark'] = data.rev.apply(lambda x: 'False' if x == 0 else 'True')
    print('Итоговая информация об эксперименте:')
    print(data.head(10))
    #print()

    
 # для автоматического пересчета метрик сгруппируем данные с информацией по эксперименту и сохраним как отдельную таблицу 
    total = data.groupby(['grp', 'rvn_mark'])                 .agg({'rvn_mark': 'count'})                 .rename(columns={'rvn_mark': 'count'})                 .reset_index()
    #print('Сводная информация об эксперименте:')
    #print(total)
    #print()
    
 # посчитаем метрики  
 # ARPPU
    ARPPU = data.query('rev > 0').groupby('grp')['rev'].mean().round(2)
    #print('ARPPU по группам равняется:\n', ARPPU)
    #print()
 # CR 
    # посчитаем количество пользователей, зашедших на сайт и количество пользователей, совершивших оплату, сделаем таблицу
    total_rvn_mark = total.pivot(index='grp', columns='rvn_mark', values='count')
    #print(total_rvn_mark)
    #print()
    # на основании полученной таблицы сделаем расчеты конверсии
    CR = (((total_rvn_mark['True']) / (total_rvn_mark['False']+ total_rvn_mark['True'])) * 100).round(2)
    print('CR по группам равняется:\n', CR)
    #print()
    return ARPPU, CR


# In[75]:


# вызовем новую функцию и посмотрим на результат ее работы
groups_add_func()


# In[76]:


df.fillna(value=0, inplace=True) 
df['rvn_mark'] = df.rev.apply(lambda x: 'False' if x == 0 else 'True')


# In[77]:


total = df.groupby(['grp', 'rvn_mark'])                 .agg({'rvn_mark': 'count'})                 .rename(columns={'rvn_mark': 'count'})                 .reset_index()


# In[78]:


ARPPU = df.query('rev > 0').groupby('grp')['rev'].mean().round(2)


# In[79]:


ARPPU


# In[80]:


total_rvn_mark = total.pivot(index='grp', columns='rvn_mark', values='count')


# In[81]:


CR = (((total_rvn_mark['True']) / (total_rvn_mark['False']+ total_rvn_mark['True'])) * 100).round(2)


# In[82]:


def plot_func(ARPPU, CR):
    plt.figure()
    plt.subplots_adjust(hspace=0.5)
    plt.subplots_adjust(wspace=0.5)
    plt.subplot(2, 2, 1)
    plt.title('Метрика ARPPU')
    ARPPU.hist(figsize=(10, 10)).set(xlabel='student', ylabel='ARPPU')
    plt.subplot(2, 2, 2)
    plt.title('Метрика CR')
    CR.hist(figsize=(10, 10)).set(xlabel='student', ylabel='CR')
plot_func(ARPPU, CR)


# In[ ]:




