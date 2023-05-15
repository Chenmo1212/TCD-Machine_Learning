# Explanation

## Package import

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from category_encoders import TargetEncoder
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
```
## Functions
```python
# 这个函数用来填充原数据中的nan值，具体方法是采用同一个州的其他城市的平均值
def nanfill(data,row, column):
    tag = data.iloc[row,1]
    # print(data[data.iloc[:,1] == tag].iloc[:,column],'\n')
    temp = data[data.iloc[:,1] == tag].iloc[:,column]
    temp = np.average(temp[temp.notna()])
    if (pd.isna(temp)):
        print(row,column)
    data.iloc[row,column] = temp

# 最大最小值归一化
def normalization(data):
    for i in range(data.shape[1]):
        m1 = min(data.iloc[:,i])
        m2 = max(data.iloc[:,i])
        data = (data.iloc[:,i] - m1) / (m2 - m1)
    
# 计算交叉验证结果的均值和标准差， 默认为五折交叉验证，结果为相关系数R2      
def cross_validate(model, cv = 5, scoring = 'r2'):
    scores = cross_val_score(model, X_train, y_train, cv = cv, scoring = scoring)
    return scores.mean(), scores.std(ddof = 1)
```
## Data import and preprocess
导入数据后对nan缺失值进行处理。考虑到不同特征均有数据缺失的情况，直接删除会影响数据量，所以采用替代的方法。又因为同一个州内的不同城市存在一定的相似性，因此用同一个州内其他城市该特征的平均值作为nan值的替代。
```python
raw_data = pd.read_csv('Data.csv')

for i in range(3):
    for j in range(raw_data.shape[0]):
        raw_data.iloc[j,i+2] = np.float64(raw_data.iloc[j,i+2].replace(',',''))
        
for i in range(raw_data.shape[0]):
    for j in range(raw_data.shape[1]):
        if (pd.isna(raw_data.iloc[i,j])):
            nanfill(raw_data,i,j)
```

## Analysis of Correlation
考虑到初步筛选的特征中有些特征之间存在明显的相关性，如一个城市的贫困率和儿童贫困率，因此我们需要对每一组可能存在相关性的变量进行相关性分析，进而筛选可用于训练模型的特征。此处计算每组中不同特征之间的Pearson相关系数以判定其线性相关性，从而实现一个初步的相关性分析。
```python
group = [['LACCESS_POP15','LACCESS_LOWI15','LACCESS_HHNV15','LACCESS_CHILD15','LACCESS_SENIORS15'],
        ['GROCPTH16', 'SUPERCPTH16', 'CONVSPTH16', 'SPECSPTH16', 'WICSPTH16'],
        ['FFRPTH16', 'FSRPTH16'],
        ['FOODINSEC_15_17', 'VLFOODSEC_15_17'],
        ['FMRKT_WIC18', 'FMRKT_WICCASH18'],
        ['POVRATE15', 'CHILDPOVRATE15']]
        
for i in range(len(group)):
    corr = raw_data[group[i]].corr()
    plt.figure(figsize = (12,8), dpi = 300)
    sns.heatmap(corr,linewidths=0.1,vmax=1.0, square=True,linecolor='white', annot=True)
    # plt.savefig('corr_heatmap_{}.jpg'.format(i))

preserve_columns = ['County','State','Population_Estimate_2016','LACCESS_POP15','GROCPTH16',
                    'SUPERCPTH16','CONVSPTH16','SPECSPTH16','WICSPTH16','FFRPTH16','FSRPTH16',
                    'FOODINSEC_15_17','FMARKT_WIC18','POVRATE15','PCT_WIC17']
raw_data = raw_data[preserve_columns]
```

从结果上来看，
- 第一组内均存在很大的线性相关性，因此只保留涉及范围最广的'LACCESS_POP15'这一特征
- 第二组内基本没有线性相关性，可以都保留
- 第三组内存在弱线性相关，可以都保留
- 第四组内存在很强的线性相关性，只保留涉及较广的'FOODINSEC_15_17'这一特征
- 第五组内存在显著的线性相关性，可以只保留'FMRKT_WIC18'
- 第六组内存在很强的线性相关性，只保留涉及较广的'POVRATE15'这一特征

## Data preprocessing
- 数据中'State'这一特征为category variable，所以需要对其向量化。此处采用Target Encoder对其编码。（具体方法在报告中再补充）
- 数据分割采用80%训练集，20%测试集，训练集中再进行交叉验证划分验证集。总共3141条数据，训练集2513条，测试集628条。
- 异常值处理待定  
- 归一化采用最大最小值归一化，公式为$$X_{scale}\frac{X-X_{min}}{X_{max}-X_{min}}$$通过归一化，可以消除特征因为其数据数量级不同对模型参数的影响。
```python
enc = TargetEncoder(cols = ['State']).fit(raw_data['State'],raw_data['PCT_WIC17'])
raw_data['State'] = enc.transform(raw_data['State'])
data = normalization(raw_data)
temp = data.sample(frac = 1).values
X_train = temp[:2512,:-1]
y_train = temp[:2512,-1]
X_test = temp[2512:,:-1]
y_test = temp[2512:,-1]
```

## Model training
选用的模型：
- KernelRidge，Lasso，ElasticNet（可以加入特征多项式化）
- 决策树回归
- 感知机（全连接神经网络）
- ...

## 线性模型

### ~~KernelRidge~~
分别对5个不同的$\alpha$采用了三种不同的核方法进行学习，分别是线性核、sigmoid核和径向基函数核（rbf）  
原理解释部分需要多做一些工作，较为复杂

### ~~Lasso~~



### ElasticNet

综合了l1与l2惩罚项

:star: 时间有限，对于结果差不多的线性模型就暂不考虑了，只留下一个考虑面稍微广泛一点的ElasticNet与剩下两个模型进行对比。

## 决策树
暂时没想到什么参数能改，目前只改了每一次划分特征时的判断方法，但从结果来看没区别
## 神经网络
对四个不同的网络结构分别训练了3个不同学习率的模型 
在我电脑上跑一遍这个流程大概需要5min