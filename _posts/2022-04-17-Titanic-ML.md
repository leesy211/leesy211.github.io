---
layout: single
title:  "Titanic 생존자 예측"
categories: coding
tag: [python, blog, jupyter]
toc: true
author_profile: false
---

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }
    
    table.dataframe td {
      text-align: center;
      padding: 8px;
    }
    
    table.dataframe tr:hover {
      background: #b8d1f3; 
    }
    
    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>


# Titanic 생존자 예측



타이타닉 호 침몰 사건 당시의 사망자와 생존자를 구분하는 요인 분석을 통해, 승객들의 생존 여부를 예측


필요한 라이브러리를 설치하고, 데이터셋을 불러온다.



```python
from pathlib import Path
import pandas as pd
import numpy as np
import tarfile
import urllib.request

def load_titanic_data():
    tarball_path = Path("datasets/titanic.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/titanic.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as titanic_tarball:
            titanic_tarball.extractall(path="datasets")
    return [pd.read_csv(Path("datasets/titanic") / filename)
            for filename in ("train.csv", "test.csv")]
```


```python
train, test = load_titanic_data()
```

훈련셋 헤드로 데이터를 확인한다.



```python
train.head()
```

<pre>
   PassengerId  Survived  Pclass  \
0            1         0       3   
1            2         1       1   
2            3         1       3   
3            4         1       1   
4            5         0       3   

                                                Name     Sex   Age  SibSp  \
0                            Braund, Mr. Owen Harris    male  22.0      1   
1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   
2                             Heikkinen, Miss. Laina  female  26.0      0   
3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   
4                           Allen, Mr. William Henry    male  35.0      0   

   Parch            Ticket     Fare Cabin Embarked  
0      0         A/5 21171   7.2500   NaN        S  
1      0          PC 17599  71.2833   C85        C  
2      0  STON/O2. 3101282   7.9250   NaN        S  
3      0            113803  53.1000  C123        S  
4      0            373450   8.0500   NaN        S  
</pre>

```python
train.shape
```

<pre>
(891, 12)
</pre>

```python
test.shape
```

<pre>
(418, 11)
</pre>

```python
train.info()
```

<pre>
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  891 non-null    int64  
 1   Survived     891 non-null    int64  
 2   Pclass       891 non-null    int64  
 3   Name         891 non-null    object 
 4   Sex          891 non-null    object 
 5   Age          714 non-null    float64
 6   SibSp        891 non-null    int64  
 7   Parch        891 non-null    int64  
 8   Ticket       891 non-null    object 
 9   Fare         891 non-null    float64
 10  Cabin        204 non-null    object 
 11  Embarked     889 non-null    object 
dtypes: float64(2), int64(5), object(5)
memory usage: 83.7+ KB
</pre>

```python
test.info()
```

<pre>
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 418 entries, 0 to 417
Data columns (total 11 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  418 non-null    int64  
 1   Pclass       418 non-null    int64  
 2   Name         418 non-null    object 
 3   Sex          418 non-null    object 
 4   Age          332 non-null    float64
 5   SibSp        418 non-null    int64  
 6   Parch        418 non-null    int64  
 7   Ticket       418 non-null    object 
 8   Fare         417 non-null    float64
 9   Cabin        91 non-null     object 
 10  Embarked     418 non-null    object 
dtypes: float64(2), int64(4), object(5)
memory usage: 36.0+ KB
</pre>
데이터 값의 분포를 확인하기 위한 라이브러리를 불러온다.



```python
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()
```

범주형 특성 분포를 확인하기 위한 파이차트를 만드는 함수를 정의한다.



```python
def pie_chart(feature):
  feature_ratio = train[feature].value_counts(sort=False)
  feature_size = feature_ratio.size
  feature_index = feature_ratio.index
  survived = train[train['Survived'] == 1][feature].value_counts()
  dead = train[train['Survived'] == 0][feature].value_counts()
  
  plt.plot(aspect='auto')
  plt.pie(feature_ratio, labels=feature_index, autopct='%1.1f%%')
  plt.title(feature + '\'s ratio in total')
  plt.show()

  for i, index in enumerate(feature_index):
    plt.subplot(1, feature_size + 1, i + 1, aspect='equal')
    plt.pie([survived[index], dead[index]], labels=['Survivied', 'Dead'], autopct='%1.1f%%')
    plt.title(str(index) + '\'s ratio')
    
    plt.show()
```

성별비율을 파이차트로 확인한다.

```python
pie_chart('Sex')
```

![sex ratio in total](https://github.com/leesy211/leesy211.github.io/blob/master/assets/images/sex%20ratio%20in%20total.png?raw=true)

![male's ratio.png](https://github.com/leesy211/leesy211.github.io/blob/master/assets/images/male's%20ratio.png?raw=true)

![female's ratio.png](https://github.com/leesy211/leesy211.github.io/blob/master/assets/images/female's%20ratio.png?raw=true)

승객 클래스 비율을 파이차트로 확인한다.



```python
pie_chart('Pclass')
```

![Pclass ratio in total.png](https://github.com/leesy211/leesy211.github.io/blob/master/assets/images/Pclass%20ratio%20in%20total.png?raw=true)

![3's ratio.png](https://github.com/leesy211/leesy211.github.io/blob/master/assets/images/3's%20ratio.png?raw=true)

![1's ratio.png](https://github.com/leesy211/leesy211.github.io/blob/master/assets/images/1's%20ratio.png?raw=true)

![2's ratio.png](https://github.com/leesy211/leesy211.github.io/blob/master/assets/images/2's%20ratio.png?raw=true)



승선지 비율을 파이차트로 확인한다.



```python
pie_chart('Embarked')
```

![Embarked ratio in total.png](https://github.com/leesy211/leesy211.github.io/blob/master/assets/images/Embarked%20ratio%20in%20total.png?raw=true)

![S's ratio.png](https://github.com/leesy211/leesy211.github.io/blob/master/assets/images/S's%20ratio.png?raw=true)

![C's ratio.png](https://github.com/leesy211/leesy211.github.io/blob/master/assets/images/C's%20ratio.png?raw=true)

![Q's ratio.png](https://github.com/leesy211/leesy211.github.io/blob/master/assets/images/Q's%20ratio.png?raw=true)





특성별 생존자와 비생존자를 비교하기 위한 바차트를 만드는 함수를 정의한다.



```python
def bar_chart(feature):
  survived = train[train['Survived']==1][feature].value_counts()
  dead = train[train['Survived']==0][feature].value_counts()
  df = pd.DataFrame([survived,dead])
  df.index = ['Survived','Dead']
  df.plot(kind='bar',stacked=True, figsize=(10,5))
```

Sibsp 특성에 따른 생존자와 비생존자



```python
bar_chart("SibSp")
```

![SibSp.png](https://github.com/leesy211/leesy211.github.io/blob/master/assets/images/SibSp.png?raw=true)



Parch 특성에 따른 생존자와 비생존자



```python
bar_chart("Parch")
```

![Parch.png](https://github.com/leesy211/leesy211.github.io/blob/master/assets/images/Parch.png?raw=true)

### 데이터 전처리 및 특성 추출



훈련셋과 테스트셋 모두에 대해 전처리하기 위해 데이터를 합친다.



```python
train_and_test = [train, test]
```

name 특성에서 새로운 title 특성을 만들어 헤드를 통해 데이터를 확인한다.



```python
for dataset in train_and_test:
  dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.')

train.head(5)
```

<pre>
   PassengerId  Survived  Pclass  \
0            1         0       3   
1            2         1       1   
2            3         1       3   
3            4         1       1   
4            5         0       3   

                                                Name     Sex   Age  SibSp  \
0                            Braund, Mr. Owen Harris    male  22.0      1   
1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   
2                             Heikkinen, Miss. Laina  female  26.0      0   
3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   
4                           Allen, Mr. William Henry    male  35.0      0   

   Parch            Ticket     Fare Cabin Embarked Title  
0      0         A/5 21171   7.2500   NaN        S    Mr  
1      0          PC 17599  71.2833   C85        C   Mrs  
2      0  STON/O2. 3101282   7.9250   NaN        S  Miss  
3      0            113803  53.1000  C123        S   Mrs  
4      0            373450   8.0500   NaN        S    Mr  
</pre>
title을 가진 사람이 몇명인지 성별과 함께 출력한다.



```python
pd.crosstab(train['Title'], train['Sex'])
```

<pre>
Sex       female  male
Title                 
Capt           0     1
Col            0     2
Countess       1     0
Don            0     1
Dr             1     6
Jonkheer       0     1
Lady           1     0
Major          0     2
Master         0    40
Miss         182     0
Mlle           2     0
Mme            1     0
Mr             0   517
Mrs          125     0
Ms             1     0
Rev            0     6
Sir            0     1
</pre>
흔하지 않은 타이틀은 Other 값으로 대체하고, 중복되지 않은 표현은 동일한 값으로 대체한다.



```python
for dataset in train_and_test:
  dataset['Title'] = dataset['Title'].replace(['Capt', 'Col', 'Countess', 'Don','Dona', 'Dr', 'Jonkheer',
                                               'Lady','Major', 'Rev', 'Sir'], 'Other')
  dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
  dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
  dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
```

<pre>
    Title  Survived
0  Master  0.575000
1    Miss  0.702703
2      Mr  0.156673
3     Mrs  0.793651
4   Other  0.347826
</pre>
학습을 위해 Title 특성을 String Data로 변형한다.



```python
for dataset in train_and_test:
  dataset['Title'] = dataset['Title'].astype(str)
```

성별 특성을 String Data로 변형한다.



```python
for dataset in train_and_test:
  dataset['Sex'] = dataset['Sex'].astype(str)
```

Embarked 특성의 데이터에 결측치가 있는지 확인한다.



```python
train.Embarked.value_counts(dropna=False)
```

<pre>
S      644
C      168
Q       77
NaN      2
Name: Embarked, dtype: int64
</pre>
결측치를 가장 많은 비율로 가진 S값으로 채우고, String Data로 변형한다.



```python
for dataset in train_and_test:
  dataset['Embarked'] = dataset['Embarked'].fillna('S')
  dataset['Embarked'] = dataset['Embarked'].astype(str)
```

나이 특성의 결측치는 나머지 모든 승객의 평균으로 대체한다.



수치형 데이터를 범주형 데이터로 변환한다.



같은 길이의 구간을 가지는 5개의 그룹을 생성한다.



```python
for dataset in train_and_test:
  dataset['Age'].fillna(dataset['Age'].mean(), inplace=True)
  dataset['Age'] = dataset['Age'].astype(int)
  train['AgeBand'] = pd.cut(train['Age'], 5)
print (train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean()) # Survivied ratio about Age Band
```

<pre>
         AgeBand  Survived
0  (-0.08, 16.0]  0.550000
1   (16.0, 32.0]  0.344762
2   (32.0, 48.0]  0.403226
3   (48.0, 64.0]  0.434783
4   (64.0, 80.0]  0.090909
</pre>
나이 속성을 위에서 만든 5개의 그룹에 속하도록 바꿔준다.



```python
for dataset in train_and_test:
  dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
  dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
  dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
  dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
  dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
  dataset['Age'] = dataset['Age'].map( { 0: 'Child',  1: 'Young', 2: 'Middle', 3: 'Prime', 4: 'Old'} ).astype(str)
```

각 승객 클래스별 요금과 어느 승객 클래스에 요금의 결측치가 있는지 확인한다.



```python
print (train[['Pclass', 'Fare']].groupby(['Pclass'], as_index=False).mean())
print("")
print(test[test["Fare"].isnull()]["Pclass"])
```

<pre>
   Pclass       Fare
0       1  84.154687
1       2  20.662183
2       3  13.675550

152    3
Name: Pclass, dtype: int64
</pre>
결측치가 있는 Fare를 같은 승객 클래스의 평균 Fare로 대체한다.



```python
for dataset in train_and_test:
  dataset['Fare'] = dataset['Fare'].fillna(13.675) # The only one empty fare data's pclass is 3.
```

수치형 데이터를 범주형 데이터로 변환한다.



```python
for dataset in train_and_test:
  dataset.loc[ dataset['Fare'] <= 7.854, 'Fare'] = 0
  dataset.loc[(dataset['Fare'] > 7.854) & (dataset['Fare'] <= 10.5), 'Fare'] = 1
  dataset.loc[(dataset['Fare'] > 10.5) & (dataset['Fare'] <= 21.679), 'Fare']   = 2
  dataset.loc[(dataset['Fare'] > 21.679) & (dataset['Fare'] <= 39.688), 'Fare']   = 3
  dataset.loc[ dataset['Fare'] > 39.688, 'Fare'] = 4
  dataset['Fare'] = dataset['Fare'].astype(int)
```

Parch와 SibSp 데이터를 합쳐서 Family라는 새로운 특성을 만든다.



```python
for dataset in train_and_test:
  dataset["Family"] = dataset["Parch"] + dataset["SibSp"]
  dataset['Family'] = dataset['Family'].astype(int)
```

학습에서 제외시킬 특성을 drop한다.



```python
train = train.drop(['PassengerId', 'AgeBand'], axis=1)

print(train.head())
print(test.head())
```

<pre>
   Survived  Pclass     Sex     Age  Fare Embarked Title  Family
0         0       3    male   Young     0        S    Mr       1
1         1       1  female  Middle     4        C   Mrs       1
2         1       3  female   Young     1        S  Miss       0
3         1       1  female  Middle     4        S   Mrs       1
4         0       3    male  Middle     1        S    Mr       0
   PassengerId  Pclass     Sex     Age  Fare Embarked Title  Family
0          892       3    male  Middle     0        Q    Mr       0
1          893       3  female  Middle     0        S   Mrs       1
2          894       2    male   Prime     1        Q    Mr       0
3          895       3    male   Young     1        S    Mr       0
4          896       3  female   Young     2        S   Mrs       2
</pre>
범주형 특성에 대해 one-hot encoding하고 Train data와 label을 분리한다.



```python
train = pd.get_dummies(train)
test = pd.get_dummies(test)

train_label = train['Survived']
train_data = train.drop('Survived', axis=1)
test_data = test.drop("PassengerId", axis=1).copy()
```

사용할 예측 모델 라이브러리를 불러온다.



```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.utils import shuffle
```

정확한 학습을 위해 학습 전 데이터를 섞어준다.



```python
train_data, train_label = shuffle(train_data, train_label, random_state = 5)
```

fit()과 predict()를 바로 적용할 함수를 정의한다.



```python
def train_and_test(model):
  model.fit(train_data, train_label)
  prediction = model.predict(test_data)
  accuracy = round(model.score(train_data, train_label) * 100, 2)
  print("Accuracy : ", accuracy, "%")
  return prediction
```

Logistic Regression



SVM



kNN



Random Forest



Navie Bayes



모델을 각각 적용하고 정확도를 비교한다.



```python
# Logistic Regression
log_pred = train_and_test(LogisticRegression())
# SVM
svm_pred = train_and_test(SVC())
#kNN
knn_pred_4 = train_and_test(KNeighborsClassifier(n_neighbors = 4))
# Random Forest
rf_pred = train_and_test(RandomForestClassifier(n_estimators=100))
# Navie Bayes
nb_pred = train_and_test(GaussianNB())
```

<pre>
Accuracy :  82.72 %
Accuracy :  83.5 %
Accuracy :  84.51 %
Accuracy :  88.55 %
Accuracy :  79.8 %
</pre>
정확도가 가장 높은 Random Forest 모델에 대한 결과 파일을 생성한다.



```python
submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": rf_pred})

submission.to_csv('submission_rf.csv', index=False)
```
