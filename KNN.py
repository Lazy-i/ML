import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets

# 读取数据
df = pd.read_csv('train.csv')
#样本数据的提取
feature = df[['age', 'workclass', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']]
target = df['income']

# 读取数据
df = pd.read_csv('test.csv')
#测试数据的提取
testfeature = df[['age', 'workclass', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']]


# 数据转换，将String类型数据(occupation)转换为int
# 数据去重
#s = feature["occupation"].unique()
## 创建一个空字典
#dic = {}
#j = 0
#for i in s:
#    dic[i] = j
#    j += 1
# 替换occupation这一列为数值类型，map映射
# 注意：str类型的数据转换成数值(int/float)类型后，注意转换后的数值与其他参于模型训练的数据数值权重比基本相同
#feature['occupation'] = feature['occupation'].map(dic)
for name in feature:
    s = feature[name].unique()
    dic = {}
    i = 0
    for j in s:
        dic[j] = i
        i = i + 1
    feature[name] = feature[name].map(dic)

#test部分

for name in testfeature:
    s = testfeature[name].unique()
    dic = {}
    i = 0
    for j in s:
        dic[j] = i
        i = i + 1
    testfeature[name] = testfeature[name].map(dic)
###target改变
##s = target["income"].unique()
##dic = {}
##j = 0
##for i in s:
##    dic[i] = j
##    j += 1    
##target['income'] = target['income'].map(dic)

# 设置30000条数据为训练数据
x_train = feature[:30000]
y_train = target[:30000]
x_train = np.array(x_train)
y_train = np.array(y_train)
# 设置测试数据
x_test = testfeature[:10000]
x_test = np.array(x_test)
# KNN模型训练,15为参数
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train, y_train)

##训练
##print('真实的分类结果：',np.array(y_train))
##print('模型的分类结果：',np.array(knn.predict(x_train)))
income = []
for res in np.array(knn.predict(x_test)):
    income.append([res])
chara = ['income']
test=pd.DataFrame(columns=chara,data=income) 
test.to_csv("result.csv",index=False)


