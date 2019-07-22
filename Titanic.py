# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from pylab import *
from matplotlib.font_manager import _rebuild
mpl.rcParams['font.family'] = ['Arial Unicode MS']
mpl.rcParams['axes.unicode_minus'] = False
train_df= pd.read_csv('/Users/shaling/Downloads/titanic/train.csv')
print(train_df.info())
print(train_df.describe())

import matplotlib.pyplot as plt
fig = plt.figure()
plt.subplot2grid((2, 2), (0, 0))

# 生存人数柱状图
ax1 = train_df.Survived.value_counts().plot.bar(ylim=[0, 700])
plt.xticks(rotation=0)
ax1.set_title('生存分布')
plt.xlabel(u'获救情况（1为获救）')
plt.ylabel(u'获救人数')
plt.text(0 - 0.15, train_df.Survived.value_counts()[
         0] + 10, train_df.Survived.value_counts()[0])
plt.text(1 - 0.15, train_df.Survived.value_counts()[
         1] + 10, train_df.Survived.value_counts()[1])

# 乘客等级分布
plt.subplot2grid((2, 2), (0, 1))
ax2 = train_df.Pclass.value_counts().plot(kind='bar', ylim=[0, 700])
ax2.set_title('乘客船票等级分布')
plt.xlabel(u'乘客等级')
plt.ylabel(u'人数')
pc1 = list(train_df.Pclass.value_counts())
x = range((len(pc1)))
for i, j in zip(list(x), pc1):
    plt.text(i - 0.2, j + 10, j)

# 不同等级船票的年龄密度分布
plt.subplot2grid((2, 2), (1, 0), colspan=2)
train_df.Age[train_df.Pclass == 1].plot(kind='kde')
train_df.Age[train_df.Pclass == 2].plot(kind='kde')
train_df.Age[train_df.Pclass == 3].plot(kind='kde')
plt.title('各等级船票年龄密度分布')
plt.xlabel('年龄')
plt.ylabel('密度')
plt.legend((u'头等舱', u'一等舱', u'二等舱'), loc='best')
plt.tight_layout()



# 登船口人数及船票等级分布
pc1 = train_df.Embarked[train_df.Pclass == 1].value_counts()
pc2 = train_df.Embarked[train_df.Pclass == 2].value_counts()
pc3 = train_df.Embarked[train_df.Pclass == 3].value_counts()
df_pc = pd.DataFrame({'高等舱': pc1, '一等舱': pc2, '经济舱': pc3}).stack().unstack()
df_pc.plot(kind='bar', stacked=True)
plt.xlabel('登船口')
plt.ylabel('人数')
plt.xticks(rotation=0)
plt.title('不同港口登船的仓位等级')

# 不同等級船票对获救的影响
survide01 = train_df.Pclass[train_df.Survived == 0].value_counts()
survide02 = train_df.Pclass[train_df.Survived == 1].value_counts()
data_survie = pd.DataFrame({'unsurvie': survide01, 'survie': survide02})
data_survie.plot(kind='bar', stacked=True)
plt.title('不同等級船票对获救的影响')
plt.xlabel(u'各等级船舱')
plt.ylabel(u'人数')
plt.xticks(rotation=0)

# 按性别查看获救情况
survie_sex01 = train_df.Survived[train_df.Sex == 'female'].value_counts(
)
survie_sex02 = train_df.Survived[train_df.Sex == 'male'].value_counts(
)
df_sex = pd.DataFrame({'女性': survie_sex01, '男性': survie_sex02})
df_sex.plot(kind='bar', stacked=True)
plt.xlabel('是否获救')
plt.ylabel('获救人数')
plt.title('不同性别获救情况')
plt.xticks([0,1],['未获救','获救'],rotation=0)


#不同港口登船的获救情况
E0 = train_df.Embarked[train_df.Survived == 0].value_counts()
E1 = train_df.Embarked[train_df.Survived == 1].value_counts()
df_em = pd.DataFrame({'unsurvived': E0, 'survied': E1})
df_em.plot(kind='bar', stacked=True)
plt.title('不同港口登船的获救情况')
plt.xticks(rotation=0)


# 亲属人数对获救的影响
ss = train_df.groupby(['SibSp', 'Survived'])
df = pd.DataFrame(ss.count()['PassengerId'])
print(df)

sp = train_df.groupby(['Parch', 'Survived'])
df1 = pd.DataFrame(sp.count()['PassengerId'])
print(df1)
fig=plt.figure(figsize=(10,8))
ax1=fig.add_subplot(121)
p_s01=train_df.Parch[train_df.Survived==0].value_counts()
p_s02=train_df.Parch[train_df.Survived==1].value_counts()
df_ps=pd.DataFrame({'unsurvie':p_s01,'survie':p_s02})
df_ps.fillna(0,inplace=True)
df_ps.plot(kind='bar',ax=ax1,title='Parch人数对获救的影响')

ax2=fig.add_subplot(122)
ss01=train_df.SibSp[train_df.Survived==0].value_counts()
ss02=train_df.SibSp[train_df.Survived==1].value_counts()
df_ss=pd.DataFrame({'unsurvie':ss01,'survie':ss02})
df_ss.fillna(0,inplace=True)
df_ss.plot(kind='bar',ax=ax2,title='SibSp人数对获救的影响')

#是否有船票号对获救的影响
survide_Cabin = train_df.Survived[pd.notnull(train_df.Cabin)].value_counts()
survide_noCabin = train_df.Survived[pd.isnull(train_df.Cabin)].value_counts()
df3 = pd.DataFrame({'notnull': survide_Cabin, 'isnull': survide_noCabin}).transpose()
df3.plot(kind='bar', stacked=True)
plt.title('是否有船票号对获救的影响')
plt.xticks(x,['有船票','没有船票'],rotation=0)#用plt.xticks(['notnull','isnull'],['有船票','没有船票'],rotation=0)无效
plt.legend(('未获救', '获救'), loc='best')

#查看年龄与获救率的关系
survie_age01=train_df.Age[train_df.Survived==0]
survie_age02=train_df.Age[train_df.Survived==1]
fig=plt.figure()
plt.hist(survie_age01,bins=10,color='#85E6FF',edgecolor='black')
plt.hist(survie_age02,color='#FFD589',alpha=0.6,bins=10)
plt.legend(['未获救','获救'])

# plt.show()


# 补全缺失的年龄
from sklearn.ensemble import RandomForestRegressor #尝试用RF填充
def set_miss_age(df):
    # 把已有的数据值特征取出来放入RFRg
    df_age = df[['Age', 'Pclass', 'SibSp', 'Parch', 'Fare']]
    no_age = df_age[pd.isnull(df_age.Age)].values  # 缺少年龄的数据
    kown_age = df_age[pd.notnull(df_age.Age)].values  # 具有年龄的数据
    x = kown_age[:, 1:]
    y = kown_age[:, 0]
    rf = RandomForestRegressor(n_estimators=1100, n_jobs=-1, random_state=0)
    rf.fit(x, y)
    #看一下预测年龄的准确度
    from sklearn.model_selection import cross_val_score,train_test_split
    score_age=cross_val_score(rf,x,y,cv=5)
    return rf,score_age
print(set_miss_age(train_df))#拟合度很差

#考虑使用姓名的称呼的平均值来判断缺失年龄
def set_age_byname(df):
	df['call']=df.Name.apply(lambda x:x.split(',')[1].split('.')[0].replace(' ',''))#提取姓名称呼
	lost_age=list(set(df.call[df.Age.isnull()]))#缺失年龄所对应的称呼
	for i in lost_age:
		if df[(df.Age.notnull())&(df.call==i)].empty:
			df.loc[(df.Age.isnull())&(df.call==i),'Age']=int(df.Age.mean())#如果缺失年龄的类型没有参考的年龄则填充所有已知平均值
		else:
			df.loc[(df.Age.isnull())&(df.call==i),'Age']=int(df.Age[(df.Age.notnull())&(df.call==i)].mean())#如果缺失年龄的类型有参考的年龄则填充参考类型的平均值
	return df
train_df=set_age_byname(train_df)
#将是否有船票特征化
def set_carbin(df):
    df.loc[(pd.notnull(df.Cabin)), 'Cabin'] = 'YES'
    df.loc[(pd.isnull(df.Cabin)), 'Cabin'] = 'NO'
    return df
train_df = set_carbin(train_df)


#补充embarked缺失值
train_df['Embarked']=train_df['Embarked'].fillna('S')#以众数补充


# 类目特征因子化
cabin_dummies = pd.get_dummies(train_df.Cabin, prefix='Cabin')
Sex_dummies = pd.get_dummies(train_df.Sex, prefix='Sex')
Embarked_dummies = pd.get_dummies(train_df.Embarked, prefix='Embarked')
Pclass_dummies = pd.get_dummies(train_df.Pclass, prefix='Pclass')
train_df = pd.concat([train_df, Pclass_dummies, cabin_dummies,Sex_dummies, Embarked_dummies], axis=1)
train_df.drop(['Pclass','Name','Sex','Ticket','Embarked','call','Cabin'],axis=1,inplace=True)


#增加child,和亲属为1人的特征值（根据之前分析亲属为1人时存活率高，年龄小于10岁时存活率高）
def f_age(x):
	if x<10:
		return 1
	else:
		return 0
train_df['child']=train_df.apply(lambda x:f_age(x.Age),axis=1)#增加10岁以下为child列

def f_parch(x,y):
	if x==1 or y==1:
		return 1
	else:
		return 0
train_df['relative']=train_df.apply(lambda x:f_parch(x.Parch,x.SibSp),axis=1)#增加为亲属为1的列

print(train_df.info())

# age,fare数值特征化
from sklearn import preprocessing
sclaer = preprocessing.StandardScaler()
# 这里series要转行成narray,但是标准化是对列进行，在进行转换时要用reshape(-1,1),而不是reshape(1,-1)
def feature_af(df,*a):
	for i in a:
		sclaerd = sclaer.fit(df[i].values.reshape(-1, 1))
		df[i+'_scalerd'] = sclaerd.transform(df[i].values.reshape(-1, 1))
	return df
train_df=feature_af(train_df,'Age','Fare')
print(train_df.info())
# train_df.to_csv('/Users/shaling/Desktop/train_df_temp.csv',index=False)

#逻辑回归建模
from sklearn import linear_model
# 提取所有特征值
trainfeauter_df = train_df.filter(regex='Survived|Pclass_.*|Sex_.*|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|child|relative|Embarked_.*')
train_np = trainfeauter_df.values
y_survie = train_np[:, 0]
x_survie = train_np[:, 1:]
stf = linear_model.LogisticRegression(C=1.0, penalty='l1', n_jobs=-1, tol=1e-6)
stf.fit(x_survie, y_survie)
print(trainfeauter_df.info())

#查看现有模型性能及系数
from sklearn.model_selection import cross_val_score,train_test_split
score_stf=cross_val_score(stf,x_survie,y_survie,cv=10).mean()
print(score_stf)
table_feature=pd.DataFrame({'feature':list(trainfeauter_df.columns[1:]),'coef':list(stf.coef_.T)})
print(table_feature)

#移除Embarked
# 提取所有特征值
# trainfeauter_df = train_df.filter(regex='Survived|Pclass_.*|Sex_.*|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|child|relative')
# train_np = trainfeauter_df.values
# y_survie = train_np[:, 0]
# x_survie = train_np[:, 1:]
# stf = linear_model.LogisticRegression(C=1.0, penalty='l1', n_jobs=-1, tol=1e-6)
# stf.fit(x_survie, y_survie)
# print(trainfeauter_df.info())

# #查看现有模型性能及系数
# from sklearn.model_selection import cross_val_score,train_test_split
# score_stf=cross_val_score(stf,x_survie,y_survie,cv=10).mean()
# print(score_stf)
# table_feature=pd.DataFrame({'feature':list(trainfeauter_df.columns[1:]),'coef':list(stf.coef_.T)})
# print(table_feature)

train_x,test_x,train_y,test_y=train_test_split(x_survie,y_survie,test_size=0.3,random_state=3)
stf.fit(train_x,train_y)
print(stf.score(train_x,train_y))
print(stf.score(test_x,test_y))



# 预测数据预处理
test_df = pd.read_csv('/Users/shaling/Downloads/titanic/test.csv')
# 补充缺失年龄
test_df=set_age_byname(test_df)
#将是否有船票特征化
test_df=set_carbin(test_df)
# 填充缺失fare值
test_df.loc[test_df.Fare.isnull(), 'Fare'] = test_df.Fare.mean()

# 类目因子化
cabin_dummies = pd.get_dummies(test_df.Cabin, prefix='Cabin')
Sex_dummies = pd.get_dummies(test_df.Sex, prefix='Sex')
Embarked_dummies = pd.get_dummies(test_df.Embarked, prefix='Embarked')
Pclass_dummies = pd.get_dummies(test_df.Pclass, prefix='Pclass')
test_df = pd.concat([test_df, Pclass_dummies, cabin_dummies,Sex_dummies, Embarked_dummies], axis=1)
test_df.drop(['Pclass','Name','Sex','Ticket','Embarked','call','Cabin'],axis=1,inplace=True)

#增加relative为1特征值
test_df['child']=test_df.apply(lambda x:f_age(x.Age),axis=1)#增加10岁以下为child列
test_df['relative']=test_df.apply(lambda x:f_parch(x.Parch,x.SibSp),axis=1)

#Age,Fare数据标准化
test_df=feature_af(test_df,'Age','Fare')

# 提取测试所有特征值
testfeature_df=test_df.filter(regex='Pclass_.*|Sex_.*|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|child|relative|Embarked_.*')

print(testfeature_df.info())

#预测获救率
stf.fit(x_survie, y_survie)
reslut_test=stf.predict(testfeature_df)
reslut_df=pd.DataFrame({'PassengerId':test_df.PassengerId,'Survived':reslut_test.astype(np.int32)})
reslut_df.to_csv('/Users/shaling/Downloads/titanic/logistic_regression_reslut.csv',index=False)