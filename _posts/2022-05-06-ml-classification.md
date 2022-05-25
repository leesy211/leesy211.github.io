# Introduction to classification

네 가지 수업에서, 고전적인 머신러닝의 기본 초점인 분류에 대해 살펴볼 것이다. 아시아와 인도의 모든 훌륭한 요리에 대한 데이터셋을 사용하여 다양한 분류 알고리즘을 사용할 것이다.


분류는 회귀 기술과 많은 공통점이 있는 지도 학습의 한 형태이다. 머신 러닝이 데이터 세트를 사용하여 값이나 이름을 예측하는 것이라면 분류는 일반적으로 이진 분류와 다중 클래스 분류의 두 그룹으로 나뉜다.



기억하기
- 선형 회귀 분석을 사용하면 변수 간의 관계를 예측하고 해당 선과 관련하여 새 데이터 점이 포함될 위치를 정확하게 예측할 수 있다. 그래서, 예를 들어 9월과 12월에 호박의 가격이 얼마인지 예측할 수 있다.
- 로지스틱 회귀 분석을 통해 "이진 범주"를 발견할 수 있었다. 이 가격대에서 호박은 주황색인가, 아니면 주황색이 아닌가?

분류는 다양한 알고리즘을 사용하여 데이터 포인트의 레이블이나 클래스를 결정하는 다른 방법을 결정한다. 이 요리 데이터를 사용하여 재료 그룹을 관찰함으로써 음식의 유래를 결정할 수 있는지 알아보자.

## 소개

분류는 머신러닝 연구자와 데이터 과학자의 기본 활동 중 하나이다. 이진 값의 기본 분류("이 이메일은 스팸입니까?")에서 컴퓨터 비전을 사용한 복잡한 이미지 분류 및 분할에 이르기까지 데이터를 클래스로 정렬하고 그에 대한 질문을 할 수 있으면 항상 유용하다.

보다 과학적인 방법으로 과정을 설명하기 위해, 분류 방법에서는 입력 변수와 출력 변수 사이의 관계를 매핑할 수 있는 예측 모형을 만든다.

데이터를 정리하고 시각화하고 머신러닝 작업에 대비하는 프로세스를 시작하기 전에, 머신 러닝을 활용하여 데이터를 분류하는 다양한 방법에 대해 조금 알아보자.

통계학에서 파생된 고전적인 기계 학습을 사용한 분류는 흡연자, 체중, 나이와 같은 특징을 사용하여 X 질환의 발병 가능성을 결정한다. 회귀와 유사한 지도 학습 기법으로, 데이터에 레이블이 지정되고 머신러닝 알고리즘은 이러한 레이블을 사용하여 데이터 세트의 클래스(또는 '특성')를 분류하고 예측하여 그룹 또는 결과에 할당한다.

잠시 요리에 대한 데이터 세트를 상상해 보자. 멀티 클래스 모델이 대답할 수 있는 것은 무엇일까? 2진수 모델은 무엇을 답할 수 있을까? 만약 주어진 요리가 페누그릭을 사용할 가능성이 있는지 여부를 결정하기를 원한다면? 만약 스타 아니스, 아티초크, 콜리플라워, 그리고 고추냉이가 가득한 식료품 봉지를 선물로 받으면, 전형적인 인도 요리를 만들 수 있을까?

## 분류기

이 요리 데이터 세트에 대해 묻고 싶은 질문은 사실 다종다양한 질문이다. 몇 가지 잠재적 국가 요리를 다룰 수 있기 때문이다. 성분 배치가 주어졌을 때, 이 많은 클래스 중 어떤 데이터가 적합할까?

Scikit-learn은 해결하려는 문제의 종류에 따라 데이터를 분류하는 데 사용할 수 있는 몇 가지 다른 알고리즘을 제공한다. 다음 두 가지 수업에서는 이러한 알고리즘 중 몇 가지에 대해 배우게 된다.

### 연습문제 - 데이터 정리 및 균형 조정

이 프로젝트를 시작하기 전에 가장 먼저 해야 할 일은 데이터를 정제하고 균형을 맞춰 더 나은 결과를 얻는 것이다.

가장 먼저 설치해야 할 것은 imblearn이다. 이 패키지는 데이터의 균형을 더 잘 조정할 수 있도록 지원하는 Scikit-learn 패키지이다.


```python
!pip install imblearn
```

    Requirement already satisfied: imblearn in /usr/local/lib/python3.7/dist-packages (0.0)
    Requirement already satisfied: imbalanced-learn in /usr/local/lib/python3.7/dist-packages (from imblearn) (0.8.1)
    Requirement already satisfied: numpy>=1.13.3 in /usr/local/lib/python3.7/dist-packages (from imbalanced-learn->imblearn) (1.21.6)
    Requirement already satisfied: scikit-learn>=0.24 in /usr/local/lib/python3.7/dist-packages (from imbalanced-learn->imblearn) (1.0.2)
    Requirement already satisfied: joblib>=0.11 in /usr/local/lib/python3.7/dist-packages (from imbalanced-learn->imblearn) (1.1.0)
    Requirement already satisfied: scipy>=0.19.1 in /usr/local/lib/python3.7/dist-packages (from imbalanced-learn->imblearn) (1.4.1)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /usr/local/lib/python3.7/dist-packages (from scikit-learn>=0.24->imbalanced-learn->imblearn) (3.1.0)



```python
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from imblearn.over_sampling import SMOTE
```


```python
df  = pd.read_csv('cuisines.csv')
```


```python
df.head()
```





  <div id="df-1b84cf8e-de6b-4318-8527-aa7e079a011d">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>cuisine</th>
      <th>almond</th>
      <th>angelica</th>
      <th>anise</th>
      <th>anise_seed</th>
      <th>apple</th>
      <th>apple_brandy</th>
      <th>apricot</th>
      <th>armagnac</th>
      <th>...</th>
      <th>whiskey</th>
      <th>white_bread</th>
      <th>white_wine</th>
      <th>whole_grain_wheat_flour</th>
      <th>wine</th>
      <th>wood</th>
      <th>yam</th>
      <th>yeast</th>
      <th>yogurt</th>
      <th>zucchini</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>65</td>
      <td>indian</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>66</td>
      <td>indian</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>67</td>
      <td>indian</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>68</td>
      <td>indian</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>69</td>
      <td>indian</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 385 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-1b84cf8e-de6b-4318-8527-aa7e079a011d')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }
    
    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }
    
    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }
    
    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-1b84cf8e-de6b-4318-8527-aa7e079a011d button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';
    
        async function convertToInteractive(key) {
          const element = document.querySelector('#df-1b84cf8e-de6b-4318-8527-aa7e079a011d');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;
    
          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2448 entries, 0 to 2447
    Columns: 385 entries, Unnamed: 0 to zucchini
    dtypes: int64(384), object(1)
    memory usage: 7.2+ MB


요리에 대한 학습

요리 당 데이터의 분포를 알아보자.


```python
#barh()를 호출하여 데이터를 막대로 표시한다.

df.cuisine.value_counts().plot.barh()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f91014e9f10>




![png](https://github.com/leesy211/leesy211.github.io/blob/master/assets/images/bar1.png?raw=true)
    


한정된 수의 요리가 있지만, 데이터의 분포가 고르지 않다. 이것은 고르게 만들 수 있다. 그렇게 하기 전에 조금 더 살펴보자.


```python
#요리당 얼마나 많은 데이터를 사용할 수 있는지 출력

thai_df = df[(df.cuisine == "thai")]
japanese_df = df[(df.cuisine == "japanese")]
chinese_df = df[(df.cuisine == "chinese")]
indian_df = df[(df.cuisine == "indian")]
korean_df = df[(df.cuisine == "korean")]

print(f'thai df: {thai_df.shape}')
print(f'japanese df: {japanese_df.shape}')
print(f'chinese df: {chinese_df.shape}')
print(f'indian df: {indian_df.shape}')
print(f'korean df: {korean_df.shape}')
```

    thai df: (289, 385)
    japanese df: (320, 385)
    chinese df: (442, 385)
    indian df: (598, 385)
    korean df: (799, 385)


재료 발견하기

이제 데이터를 더 깊이 파고들어 요리당 전형적인 재료가 무엇인지 배울 수 있다. 음식 사이에 혼란을 일으키는 반복적인 데이터를 지워야 하는데, 이 문제에 대해 알아보도록 하자.

파이썬에서 create_incomment() 함수를 만들어 성분 데이터 프레임을 만든다. 이 함수는 도움이 되지 않는 열을 떨어뜨리는 것부터 시작하여 재료를 개수에 따라 정렬합니다.


```python
def create_ingredient_df(df):
    ingredient_df = df.T.drop(['cuisine','Unnamed: 0']).sum(axis=1).to_frame('value')
    ingredient_df = ingredient_df[(ingredient_df.T != 0).any()]
    ingredient_df = ingredient_df.sort_values(by='value', ascending=False,
    inplace=False)
    return ingredient_df
```

이제 이 함수를 사용하여 요리별로 가장 인기 있는 10대 식재료에 대한 아이디어를 얻을 수 있다.


```python
#create_interent()를 호출하고 barh()를 호출하여 출력
#태국 데이터
thai_ingredient_df = create_ingredient_df(thai_df)
thai_ingredient_df.head(10).plot.barh()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f9100d2f250>




![png](https://github.com/leesy211/leesy211.github.io/blob/master/assets/images/bar2.png?raw=true)
    



```python
#일본 데이터
japanese_ingredient_df = create_ingredient_df(japanese_df)
japanese_ingredient_df.head(10).plot.barh()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f9100e20c10>




![png](https://github.com/leesy211/leesy211.github.io/blob/master/assets/images/bar3.png?raw=true)
    



```python
#중국 데이터
chinese_ingredient_df = create_ingredient_df(chinese_df)
chinese_ingredient_df.head(10).plot.barh()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f9100e73e10>




![png](https://github.com/leesy211/leesy211.github.io/blob/master/assets/images/bar4.png?raw=true)
    



```python
#인도 데이터
indian_ingredient_df = create_ingredient_df(indian_df)
indian_ingredient_df.head(10).plot.barh()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f9100ccf2d0>




![png](https://github.com/leesy211/leesy211.github.io/blob/master/assets/images/bar5.png?raw=true)
    



```python
#한국 데이터
korean_ingredient_df = create_ingredient_df(korean_df)
korean_ingredient_df.head(10).plot.barh()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f9100d0ee10>




​    
![png](https://github.com/leesy211/leesy211.github.io/blob/master/assets/images/bar6.png?raw=true)
​    


이제 drop()을 호출하여 구별되는 요리 사이에 혼란을 일으키는 가장 일반적인 재료를 삭제한다.


```python
feature_df= df.drop(['cuisine','Unnamed: 0','rice','garlic','ginger'], axis=1)
labels_df = df.cuisine #.unique()
feature_df.head()
```





  <div id="df-1f940346-6067-4b71-9f44-92c057ff96e9">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>almond</th>
      <th>angelica</th>
      <th>anise</th>
      <th>anise_seed</th>
      <th>apple</th>
      <th>apple_brandy</th>
      <th>apricot</th>
      <th>armagnac</th>
      <th>artemisia</th>
      <th>artichoke</th>
      <th>...</th>
      <th>whiskey</th>
      <th>white_bread</th>
      <th>white_wine</th>
      <th>whole_grain_wheat_flour</th>
      <th>wine</th>
      <th>wood</th>
      <th>yam</th>
      <th>yeast</th>
      <th>yogurt</th>
      <th>zucchini</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 380 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-1f940346-6067-4b71-9f44-92c057ff96e9')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }
    
    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }
    
    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }
    
    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-1f940346-6067-4b71-9f44-92c057ff96e9 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';
    
        async function convertToInteractive(key) {
          const element = document.querySelector('#df-1f940346-6067-4b71-9f44-92c057ff96e9');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;
    
          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




### 데이터셋 균형 맞추기

이제 데이터를 정리했으므로 SMOTE - "합성 소수 과표본 기법"("Synthetic Minority Over-sampling Technique")을 사용하여 균형을 맞춘다.

fit_resample()을 호출하면 이 전략은 보간법을 통해 새 샘플을 생성한다.


```python
oversample = SMOTE()
transformed_feature_df, transformed_label_df = oversample.fit_resample(feature_df, labels_df)
```

데이터의 균형을 유지함으로써 데이터를 분류할 때 더 나은 결과를 얻을 수 있다. 이진 분류에 대해 생각해 보자. 대부분의 데이터가 하나의 클래스인 경우 머신러닝 모델은 해당 클래스를 더 자주 예측합니다. 단지 더 많은 데이터가 있기 때문이다. 데이터의 균형을 맞추려면 왜곡된 데이터가 필요하며 이러한 불균형을 제거하는 데 도움이 된다.


```python
#재료당 레이블의 수 확인
print(f'new label count: {transformed_label_df.value_counts()}')
print(f'old label count: {df.cuisine.value_counts()}')
```

    new label count: indian      799
    thai        799
    chinese     799
    japanese    799
    korean      799
    Name: cuisine, dtype: int64
    old label count: korean      799
    indian      598
    chinese     442
    japanese    320
    thai        289
    Name: cuisine, dtype: int64


데이터가 잘 정제되었다.

마지막 단계는 레이블 및 특성을 포함한 균형 잡힌 데이터를 파일로 내보낼 수 있는 새 데이터 프레임에 저장하는 것이다.


```python
transformed_df = pd.concat([transformed_label_df,transformed_feature_df],axis=1, join='outer')
transformed_df
```





  <div id="df-4501aec5-df47-44f0-bf54-caa3643c760c">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cuisine</th>
      <th>almond</th>
      <th>angelica</th>
      <th>anise</th>
      <th>anise_seed</th>
      <th>apple</th>
      <th>apple_brandy</th>
      <th>apricot</th>
      <th>armagnac</th>
      <th>artemisia</th>
      <th>...</th>
      <th>whiskey</th>
      <th>white_bread</th>
      <th>white_wine</th>
      <th>whole_grain_wheat_flour</th>
      <th>wine</th>
      <th>wood</th>
      <th>yam</th>
      <th>yeast</th>
      <th>yogurt</th>
      <th>zucchini</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>indian</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>indian</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>indian</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>indian</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>indian</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3990</th>
      <td>thai</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3991</th>
      <td>thai</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3992</th>
      <td>thai</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3993</th>
      <td>thai</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3994</th>
      <td>thai</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>3995 rows × 381 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-4501aec5-df47-44f0-bf54-caa3643c760c')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }
    
    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }
    
    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }
    
    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-4501aec5-df47-44f0-bf54-caa3643c760c button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';
    
        async function convertToInteractive(key) {
          const element = document.querySelector('#df-4501aec5-df47-44f0-bf54-caa3643c760c');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;
    
          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




transformed_df.head() 및 transformed_df.info()를 사용하여 데이터를 한 번 더 살펴볼 수 있다. 다음 교육에서 사용할 수 있도록 이 데이터 복사본을 저장한다.


```python
transformed_df.head()
```





  <div id="df-a6555b9c-2998-4da7-bcca-290777715994">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cuisine</th>
      <th>almond</th>
      <th>angelica</th>
      <th>anise</th>
      <th>anise_seed</th>
      <th>apple</th>
      <th>apple_brandy</th>
      <th>apricot</th>
      <th>armagnac</th>
      <th>artemisia</th>
      <th>...</th>
      <th>whiskey</th>
      <th>white_bread</th>
      <th>white_wine</th>
      <th>whole_grain_wheat_flour</th>
      <th>wine</th>
      <th>wood</th>
      <th>yam</th>
      <th>yeast</th>
      <th>yogurt</th>
      <th>zucchini</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>indian</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>indian</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>indian</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>indian</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>indian</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 381 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-a6555b9c-2998-4da7-bcca-290777715994')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }
    
    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }
    
    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }
    
    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-a6555b9c-2998-4da7-bcca-290777715994 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';
    
        async function convertToInteractive(key) {
          const element = document.querySelector('#df-a6555b9c-2998-4da7-bcca-290777715994');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;
    
          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
transformed_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3995 entries, 0 to 3994
    Columns: 381 entries, cuisine to zucchini
    dtypes: int64(380), object(1)
    memory usage: 11.6+ MB



```python
transformed_df.to_csv("cleaned_cuisines.csv")
```

# Cuisine Classifiers 1

이 과정에서는 위에서 저장한 모든 음식에 대한 균형 잡힌 깨끗한 데이터로 가득 찬 데이터 세트를 사용한다.

이 데이터 세트를 다양한 분류기와 함께 사용하여 재료 그룹을 기반으로 특정 국가 음식을 예측할 수 있다. 이렇게 하는 동안 분류 작업에 알고리즘을 활용할 수 있는 몇 가지 방법에 대해 자세히 알아볼 수 있다.

## Preparation

Exercise - 국가 음식 예측하기


```python
cuisines_df = pd.read_csv("cleaned_cuisines.csv")
cuisines_df.head()
```





  <div id="df-c5d82790-fb1a-4562-a999-aa979474a388">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>cuisine</th>
      <th>almond</th>
      <th>angelica</th>
      <th>anise</th>
      <th>anise_seed</th>
      <th>apple</th>
      <th>apple_brandy</th>
      <th>apricot</th>
      <th>armagnac</th>
      <th>...</th>
      <th>whiskey</th>
      <th>white_bread</th>
      <th>white_wine</th>
      <th>whole_grain_wheat_flour</th>
      <th>wine</th>
      <th>wood</th>
      <th>yam</th>
      <th>yeast</th>
      <th>yogurt</th>
      <th>zucchini</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>indian</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>indian</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>indian</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>indian</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>indian</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 382 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-c5d82790-fb1a-4562-a999-aa979474a388')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }
    
    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }
    
    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }
    
    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-c5d82790-fb1a-4562-a999-aa979474a388 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';
    
        async function convertToInteractive(key) {
          const element = document.querySelector('#df-c5d82790-fb1a-4562-a999-aa979474a388');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;
    
          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
from sklearn.svm import SVC
import numpy as np
```

훈련을 위해 X 및 y 좌표를 두 개의 데이터 프레임으로 나눈다. 요리는 레이블 데이터 프레임이 될 수 있다.


```python
cuisines_label_df = cuisines_df['cuisine']
cuisines_label_df.head()
```




    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object



drop()을 호출하여 이름이 없는 0번 열과 cuisine 열을 삭제한다.


```python
cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
cuisines_feature_df.head()
```





  <div id="df-8496bc7a-41aa-4652-8eae-2ddad74a2b71">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>almond</th>
      <th>angelica</th>
      <th>anise</th>
      <th>anise_seed</th>
      <th>apple</th>
      <th>apple_brandy</th>
      <th>apricot</th>
      <th>armagnac</th>
      <th>artemisia</th>
      <th>artichoke</th>
      <th>...</th>
      <th>whiskey</th>
      <th>white_bread</th>
      <th>white_wine</th>
      <th>whole_grain_wheat_flour</th>
      <th>wine</th>
      <th>wood</th>
      <th>yam</th>
      <th>yeast</th>
      <th>yogurt</th>
      <th>zucchini</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 380 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-8496bc7a-41aa-4652-8eae-2ddad74a2b71')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }
    
    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }
    
    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }
    
    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-8496bc7a-41aa-4652-8eae-2ddad74a2b71 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';
    
        async function convertToInteractive(key) {
          const element = document.querySelector('#df-8496bc7a-41aa-4652-8eae-2ddad74a2b71');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;
    
          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




모델을 훈련할 준비가 되었다.

## 분류기 선택하기

이제 데이터가 깨끗해지고 훈련 준비가 되었으므로 작업에 사용할 알고리즘을 결정해야 한다.

Scikit-learn "지도 학습"에서 분류를 나누고, 이 범주에서 분류할 수 있는 여러 가지 방법을 찾을 수 있다. 다음 방법에는 모두 분류 기법이 포함된다.

- Linear Models
- Support Vector Machines
- Stochastic Gradient Descent
- Nearest Neighbors
- Gaussian Processes
- Decision Trees
- Ensemble methods (voting Classifier)
- Multiclass and multioutput algorithms (multiclass and multilabel classification, multiclass-multioutput classification)

어떤 분류기를 선택할 것인가?

주로 여러 개를 훑어보고 좋은 결과를 찾는 것이 분류기를 선택하기 위한 테스트 방법이다.

Scikit-learn은 생성된 데이터 세트에 대해 KNeighbors, SVC 두 가지 방법, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB and QuadraticDiscrinationAnalysis를 비교하여 결과를 시각화하여 보여준다.

더 좋은 접근방법

그러나 마구 추측하는 것보다 더 나은 방법은 다운로드 가능한 머신러닝 치트 시트의 아이디어를 따르는 것이다. 여기서, 다중 클래스 문제에 대해 몇 가지 선택사항이 있다는 것을 발견한다.

![cheatsheet.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAtgAAAJuCAYAAACKQHVtAAAKw2lDQ1BJQ0MgUHJvZmlsZQAASImVlwdUk8kWgOf/00NCSQLSCb1Jly4lhBZAercRkkBCCSEhKNiVRQXXgooIKIouiii4FkDWAohiYRGsKOiCLCrKuliwofJ+4BF23zvvvfPuOffMd+5/586dOXP/cwcAMpotEqXBigCkC7PE4X5e9Ni4eDpuEGCAIiAAW4BjcyQiRmhoEEBkZvy7fLgPoMnxjsVkrH///l9FicuTcACAQhFO5Eo46QifQfQlRyTOAgB1ALHrL8sSTXIbwjQxkiDCPZOcPM0jk5w4xWgw5RMZzkSYBgCexGaLkwEg0RE7PZuTjMQheSJsLeQKhAiLEHbn8NlchE8iPDc9PWOS+xA2SfxLnOS/xUyUxWSzk2U8vZcpwXsLJKI0ds7/eRz/W9LTpDNrGCFK4ov9w5GRgpxZT2pGoIyFicEhMyzgTvlPMV/qHzXDHAkzfoa5bO9A2dy04KAZThL4smRxsliRM8yT+ETMsDgjXLZWkpjJmGG2eHZdaWqUzM7nsWTxc/mRMTOcLYgOnmFJakTgrA9TZhdLw2X584R+XrPr+sr2ni75y34FLNncLH6kv2zv7Nn8eULGbExJrCw3Ls/bZ9YnSuYvyvKSrSVKC5X589L8ZHZJdoRsbhZyIWfnhsrOMIUdEDrDgAkEQAh4IB2wAR34A28AsnjLsyY3wswQ5YgFyfwsOgOpMB6dJeRYzqXbWts4AzBZr9PX4S1/qg4hysCsLQeZ72kDAOwza4tdBUANUqcqCbM2gyGkbJB71bSJIxVnT9umagkDiEAB0IAa0Ab6wARYIP8EB+AKPIEPCAAhIBLEgSWAA/hI3mKwDKwE60A+KATbwW5QCirAIXAUnACnQAM4D1rAVXATdIF7oBf0gyHwCoyCD2AcgiAcRIaokBqkAxlC5pAt5AS5Qz5QEBQOxUEJUDIkhKTQSmgDVAgVQaXQQaga+hk6B7VA16Fu6CE0AA1Db6EvMAomwTRYCzaCrWAnmAEHwpHwYjgZzoRz4Tx4K1wCV8LH4Xq4Bb4J34P74VfwGAqg5FAqKF2UBcoJxUSFoOJRSSgxajWqAFWMqkTVoppQ7ag7qH7UCOozGoumouloC7Qr2h8dheagM9Gr0VvQpeij6Hp0G/oOegA9iv6OIWM0MeYYFwwLE4tJxizD5GOKMVWYs5grmHuYIcwHLBargjXGOmL9sXHYFOwK7BbsPmwdthnbjR3EjuFwODWcOc4NF4Jj47Jw+bi9uOO4S7jbuCHcJ7wcXgdvi/fFx+OF+PX4Yvwx/EX8bfxz/DhBkWBIcCGEELiEHMI2wmFCE+EWYYgwTlQiGhPdiJHEFOI6YgmxlniF2Ed8JycnpyfnLBcmJ5BbK1cid1LumtyA3GcShWRGYpIWkaSkraQjpGbSQ9I7MplsRPYkx5OzyFvJ1eTL5CfkT/JUeUt5ljxXfo18mXy9/G351woEBUMFhsIShVyFYoXTCrcURhQJikaKTEW24mrFMsVzig8Ux5SoSjZKIUrpSluUjildV3pBwVGMKD4ULiWPcohymTJIRVH1qUwqh7qBeph6hTpEw9KMaSxaCq2QdoLWSRtVpijPU45WXq5cpnxBuV8FpWKkwlJJU9mmckrlvsqXOVpzGHN4czbPqZ1ze85HVQ1VT1WeaoFqneo91S9qdDUftVS1HWoNao/V0epm6mHqy9T3q19RH9GgabhqcDQKNE5pPNKENc00wzVXaB7S7NAc09LW8tMSae3Vuqw1oq2i7amdor1L+6L2sA5Vx11HoLNL55LOS7oynUFPo5fQ2+ijupq6/rpS3YO6nbrjesZ6UXrr9er0HusT9Z30k/R36bfqjxroGCwwWGlQY/DIkGDoZMg33GPYbvjRyNgoxmijUYPRC2NVY5ZxrnGNcZ8J2cTDJNOk0uSuKdbUyTTVdJ9plxlsZm/GNyszu2UOmzuYC8z3mXfPxcx1niucWzn3gQXJgmGRbVFjMWCpYhlkud6ywfK1lYFVvNUOq3ar79b21mnWh617bSg2ATbrbZps3tqa2XJsy2zv2pHtfO3W2DXavZlnPo83b/+8Hnuq/QL7jfat9t8cHB3EDrUOw44GjgmO5Y4PnGhOoU5bnK45Y5y9nNc4n3f+7OLgkuVyyuVPVwvXVNdjri/mG8/nzT88f9BNz43tdtCt353unuB+wL3fQ9eD7VHp8dRT35PrWeX5nGHKSGEcZ7z2svYSe531+sh0Ya5iNnujvP28C7w7fSg+UT6lPk989XyTfWt8R/3s/Vb4Nftj/AP9d/g/YGmxOKxq1miAY8CqgLZAUmBEYGng0yCzIHFQ0wJ4QcCCnQv6gg2DhcENISCEFbIz5HGocWhm6C9h2LDQsLKwZ+E24SvD2yOoEUsjjkV8iPSK3BbZG2USJY1qjVaIXhRdHf0xxjumKKY/1ip2VezNOPU4QVxjPC4+Or4qfmyhz8LdC4cW2S/KX3R/sfHi5YuvL1FfkrbkwlKFpeylpxMwCTEJxxK+skPYleyxRFZieeIoh8nZw3nF9eTu4g7z3HhFvOdJbklFSS+S3ZJ3Jg/zPfjF/BEBU1AqeJPin1KR8jE1JPVI6kRaTFpdOj49If2ckCJMFbZlaGcsz+gWmYvyRf2ZLpm7M0fFgeIqCSRZLGnMoiGNUYfURPqDdCDbPbss+9Oy6GWnlystFy7vyDHL2ZzzPNc396cV6BWcFa0rdVeuWzmwirHq4GpodeLq1jX6a/LWDK31W3t0HXFd6rpf11uvL1r/fkPMhqY8rby1eYM/+P1Qky+fL85/sNF1Y8Um9CbBps7Ndpv3bv5ewC24UWhdWFz4dQtny40fbX4s+XFia9LWzm0O2/Zvx24Xbr+/w2PH0SKlotyiwZ0Ldtbvou8q2PV+99Ld14vnFVfsIe6R7ukvCSpp3Guwd/ver6X80ntlXmV15Zrlm8s/7uPuu73fc39thVZFYcWXA4IDPQf9DtZXGlUWH8Ieyj707HD04fafnH6qrlKvKqz6dkR4pP9o+NG2asfq6mOax7bVwDXSmuHji453nfA+0VhrUXuwTqWu8CQ4KT358ueEn++fCjzVetrpdO0ZwzPlZ6lnC+qh+pz60QZ+Q39jXGP3uYBzrU2uTWd/sfzlyHnd82UXlC9su0i8mHdx4lLupbFmUfNIS3LLYOvS1t7LsZfvtoW1dV4JvHLtqu/Vy+2M9kvX3K6dv+5y/dwNpxsNNx1u1nfYd5z91f7Xs50OnfW3HG81djl3NXXP77542+N2yx3vO1fvsu7evBd8r/t+1P2eB4se9Pdwe148THv45lH2o/HetX2YvoLHio+Ln2g+qfzN9Le6fof+CwPeAx1PI572DnIGX/0u+f3rUN4z8rPi5zrPq1/Yvjg/7Dvc9XLhy6FXolfjI/l/KP1R/trk9Zk/Pf/sGI0dHXojfjPxdss7tXdH3s973zoWOvbkQ/qH8Y8Fn9Q+Hf3s9Ln9S8yX5+PLvuK+lnwz/db0PfB730T6xISILWZPtQIoROGkJKTHOAIAOQ4AahcAxIXT/fSUQNNvgCkC/4mne+4pcQDgUDMAMYgGIFq+FgBDRKnIp1BPACKbAWxnJ9N/iiTJznY6FgnpubHoiYl3xQDgegH4pjsxMd4yMfGtFUm2EoDLPtN9/KRgkddNkS0MbFY0R7atBf8i/wCcpg/U/44CFAAAAIplWElmTU0AKgAAAAgABAEaAAUAAAABAAAAPgEbAAUAAAABAAAARgEoAAMAAAABAAIAAIdpAAQAAAABAAAATgAAAAAAAACQAAAAAQAAAJAAAAABAAOShgAHAAAAEgAAAHigAgAEAAAAAQAAAtigAwAEAAAAAQAAAm4AAAAAQVNDSUkAAABTY3JlZW5zaG90bP9FmgAAAAlwSFlzAAAWJQAAFiUBSVIk8AAAAdZpVFh0WE1MOmNvbS5hZG9iZS54bXAAAAAAADx4OnhtcG1ldGEgeG1sbnM6eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1wdGs9IlhNUCBDb3JlIDYuMC4wIj4KICAgPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4KICAgICAgPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIKICAgICAgICAgICAgeG1sbnM6ZXhpZj0iaHR0cDovL25zLmFkb2JlLmNvbS9leGlmLzEuMC8iPgogICAgICAgICA8ZXhpZjpQaXhlbFlEaW1lbnNpb24+NjIyPC9leGlmOlBpeGVsWURpbWVuc2lvbj4KICAgICAgICAgPGV4aWY6UGl4ZWxYRGltZW5zaW9uPjcyODwvZXhpZjpQaXhlbFhEaW1lbnNpb24+CiAgICAgICAgIDxleGlmOlVzZXJDb21tZW50PlNjcmVlbnNob3Q8L2V4aWY6VXNlckNvbW1lbnQ+CiAgICAgIDwvcmRmOkRlc2NyaXB0aW9uPgogICA8L3JkZjpSREY+CjwveDp4bXBtZXRhPgrD7i3GAAAAHGlET1QAAAACAAAAAAAAATcAAAAoAAABNwAAATcAALvsHn1+DgAAQABJREFUeAHsXQe8XEX1nigljRJ6AgqEXhTp0pvSQhWkiRJQalAQAkiTLlX4g3RQioUiokhREIEAItJBgtJ7EhI6SaiS/3x337d77nlzd+972bzs5n2TX94tM3PmzHfOzHwzd+7dPhMnTpwSFISAEBACQkAICAEhIASEgBBoCgJ9RLCbgqOECAEhIASEgBAQAkJACAiBDAERbDmCEBACQkAICAEhIASEgBBoIgIi2E0EU6KEgBAQAkJACAgBISAEhIAItnxACAgBISAEhIAQEAJCQAg0EQER7CaCKVFCQAgIASEgBISAEBACQkAEWz4gBISAEBACQkAICAEhIASaiIAIdhPBlCghIASEgBAQAkJACAgBISCCLR8QAkJACAgBISAEhIAQEAJNREAEu4lgSpQQEAJCQAgIASEgBISAEBDBlg8IASEgBISAEBACQkAICIEmIiCC3UQwJUoICAEhIASEgBAQAkJACIhgyweEgBAQAkJACAgBISAEhEATERDBbiKYEiUEhIAQEAJCQAgIASEgBESw5QNCQAgIASEgBISAEBACQqCJCIhgNxFMiRICQkAICAEhIASEgBAQAiLY8gEhIASEgBAQAkJACAgBIdBEBESwmwimRAkBISAEhIAQEAJCQAgIARFs+YAQEAJCQAgIASEgBISAEGgiAiLYTQRTooSAEBACQkAICAEhIASEgAi2fEAICAEhIASEgBAQAkJACDQRgW4R7BcnhHDpXZ+EPl8IYcrnQUfhID9QO1A/oH5A/YD6AfUD6gfash/46iJfCDuuOlMT6XXkxhMnTpzSVYkvTPg8nPu3T7qaTemFgBAQAkJACAgBISAEhEBLIbD6YjOFHVdvEYJ99q0fhz594gp2pOc6Cgf5gdqB+gH1A+oH1A+oH1A/0I79wBpLzBR2Wn3mppL+bq9gn/XXj6IXRV0iyc4dqZ6/z2vFVxAgHv4ofIQPEPB+wWv5h/xD/qH2wf7AH9U/qH9Q/9Ct/mHNJWYOO3+9BQj28+OnhJ/fMimuXPeJK9hTdBQO8gO1A/UD6gfUD6gfUD+gfqAt+4G1lp4lfOfrs3CK2pRjt1awQbBPv3lideGamnBBW9cVBIRHZSIpf5A/AAG1B7UHLLgyyB/kD/IHtgb1j9O7P1g7Euxd15i1ZpAmnHWbYJ9y4wdNKF4ihIAQEAJCQAgIASEgBITA9ENg3aVnDd9dqwUI9nNxBfvkG96vrkhx5qFj8dYfv1VO17UZu/xGfqP2oPagfkD9gPoB9QPTqx9Yb7m+4XtrtgTB/l844Y/vZ3ut4mab2CsU78VWvPCRf6h9FL2rof5B/YP6B/UP6h/S77Kpf+y5/nHD5fuF3dbq29Ql9G5tEXnujc/DcX98t2OqxTln1Ctigc6y4hTmOosw1xkpN9eKj2AYPISPWcqRf1UalvxD/Uv0AfWvGl80Pmh84LCI/kD8KWLQAcNU9I8bRYI9fO1+FVlN+tstgv1sJNjHXPdO7Os58+7o88itq0fFV2bmwqfjQQcn5B1H+Yf8Ays3ah9qH2buUPUH9Q/qH9Q/qH/smfHhmyv0D7u3BsH+LBz9+7iCrSAEhIAQEAJCQAgIASEgBNoYgY0jwd5jnf5NrUGf75337JTTdlwgDOwblw1KBqxgH3HNm/G3MOIKQ/xXXcnWtfCQP7RMe1hvWTzuyvZtVY+jnvpwqvRbeqGZsnb/1GufNKX9r7ssXipB31PT8+7/fByfcMR+Jf5T/1LpX2eNv3+w+uLYH1jDafJHU8KDL+AXdTtWetsEr2UXmiWz7ytvfh4mffy/uvrDP5YdMnOYb7YvxrqH8OSYT8L1D3xY9Y8vz/vFMGDWL4QXJ3wWPvoY3tJe41G7699ueEvf9mofPWmvTb82MHx/3SYT7K8e/MSUxYfMGs7fbUhpkv3MuM/C4Ve/Venq+fyCXb+uK88zhMcM6R/LfalCDjDYV0KFHL75wedh/HsVspDtf4mRlVcSOp53d1wvMt/MoX/uW/Zdyw8yVU++jV8/EuwR3xjYoWcIn0eFdjz3zdL5rf79I4k5fedBYb7Zv5DJe+nNz8Ixf3gvTP4kpsL+hhhs+rLXG3+1X9hr/ZqOn3w2JXzn/KhjN+XZ+ndHn57MD0yXi2Rz4UgSbZj88edh9GufhpciaaQ+/WbpE67cZx6bLJxy8/vh4Rc+mSr8KR+CewKvM3YZFBaZZ6asHuPf/zwc+8f3woT3PsuubfkLzztTOO5bc2TkOYs0f759zoTs6pAt5wirLVppTMDsp9e/F15+839tg0e762/tBYPoGtucyvfPwqu18Np0xdnCnutNA4KNxtEVkg2Cfejv3gxTojP1iYPrlOhTfaK31K55n0fFC58Zwz/+eMC8aC6dwr8i0Tnlxvc62gP9nseK/88z50zh4uFzdcqLG1f/a3K45v7466hZe2I+HrvXfjZcvn/Y3xHs7c8en2injeXv+PWBYafV853Pr+6ZGG56ePJUtf9Nvto37LPBbFVMQLB3PC8+Hcvh0Fi/dmpf8ANgueEy9d9YnxRJ43cvQD8bQv+Z+4Tf7pf3vZNvej88+NxHU4V/T/bfq8YV+MO3mL1qa5zc9NiH4ZejPsiNH8DnrJ3nTJJr5Nn27Alh+YVmDidsNycuq+HO/3wUfnFr/HxsdVyauvZTk9N8/0vpjz7k1D+/2xb6z8jts2b31vUf4Q/e2Vz7DFt5jkiwm/yS4xpHPjll0kefZ53UwH5fCOcNHxKWWCC3xFbtwHjy9LhPw8jfxo4/3oh9v47Codf4wZ9/PD+bQafjzudPCFhJK2oXW67cP/xg3RqZtAKu+tekcNV9E5uK40bL9Q8/2rhWHlawt/2/N5L6DZ1vpmw1dY3F+oYjfv92Jz22jrp/3+n+f7d9EO4cHQn2VPj/Ziv0C/tuWCNdINjb/2J8p/JnlH5m5zUHhp1XH2BNX/d867Mq9howyxfCVSPyBPukOKF7AAR7KvDvSVzx9Odn2w/K1Tfl9wdsOkenyceL8YnJG+//L8w/+xfDAb9+K8Bf/+87c+dk3REJ9tl/fa8l8FhgjpnCMgvOHNZYfNZw9f0Tw4vjP8vpldL//rjd5+QbIsFuE3tKT/Gfnuw/prW/bbHS7GHvDcr3zbnOp+Ciz6PPvT1lv8vGhK6Q7KfjCvaBV75ReRzS0R1U3nauwa3ryl4nTj+Ex4yBx00HL1DQlEI4K66e/f3JSdnwmLL3ubvNGxaNj75T4bf/nJgR7Gb6y8ZfGRAO2KRGXkGwtzpzbE4/fJpolzUGZsSFem3x83GVPa543NnRvgfELQpHbztXtnKIdE/GLQw/ueatanyqvtnj0gb9w2Zfi9tYNpoDIrMAgv2tsyOpjCvYZfJTv3ZIf9Dmc4aNsn3xrG3j4xY/r9gLK9jX/ig/uTsxkrH7n/swCuHXNmr2alU89oyTqa1XrDwJwb5p+NAk7J029r52//kDts8w3P/8x/F3F+Kkz/gj0u8V/Yayxkfyffg170QS/ul0xWNo3AJ2YJwgDDXt/PBr3w5PvFLZK2/99Ttrzx52+XplQG8V/a1+Hm9dt377kv0qNNz2J2X7x21WnbP5BHvixIlTxr77WTjs6jfCc2M/zvq0RivZ/x37aTjgyvExrZ1TsDusOaHihU9tjj9j+MdfDhnCilRX1Hjj+fGfhv2vwP5ha/dKe5hv9pnCFXvPx6Sd8v4mEuzf3vt+jG9e+/nmV/qHg+JgzwCCPewMELaafhjkd40E24bNTh8TL70elRQDZsV+4SkZKbJyavl9Pl4zBa8rx83iiyU/+mZtEgCCvfVZFVLZDPm1eqTL76n4bVYdGPZev1ZPaoNV2fuf+zg8HgkYtoQMnX/mbGvEN5frl016Nju9Yi+sYF93QH5yd/yf3gn/fPajKAr2ZMjj21P1Y+m18rweTNEn1q9PGBj9iGTY6/+XQwYzcXas1BMTCYS8XPgj5I1/v7KP28fXriu5a9d5OYyd2vivfrlvOHXH/Daww+IkAgS7Eny5feJ7DV9sGf2ntv7K39m+3r/z/RrT0wN57Y+KL2o/zcIXBHvfDfNjIVHv7rH6HeyJ8Y30/S4fkyPZP9ps7rDFCrVHzCzk6bH/CyNiWsxoOVPQ0bydK1xmWL+49dAawf73q59kj6rxFQOG7174Rhws/9ep/tusMiA23grZBZF6Lj4yXiE+Mmf49X0fhN/c+0GnfFPTrjb96sBw0GZ5gr3paa/n2u2ua88Wvrtmvo1vctqYpupRr5/YfMW4yv7Nmo4g2FvGVfapqXe98qaH3AXmnDlcaSZXtPmf4v7182+3vyeQ709XWLhvePzljzJ79Ysr2H/6cZ54HvvHd8J9z0zO2XN61K/ZeN922IKEKDseEl+oJw6tXr+vxDZ9xs75l1HbSf9Wx1f6iWc0u7+hvO1WnyuO0U3eIoIVbPZmnmTj/hHfmrcTyX4qrmDvf1lcWcHWbXALHYVDL/GD249YCM0iC+PiV0Owar3WErWX1c67493wx7jn0reLi34wf1gsPj5G+EdcdRzY9ws5gn3lPz4IV456r5pv4xUGhPnniKvFHQsZb8Sybns8bj8xOM8fV8Xx7U6zIB1+jVXwjva4SSTYh8RtCQxYwd741NeyeMpfIa64fe3LNaKPtFdARke5t/17cnjjnc/C/INiWXFFnPdxvO3xGIeVw0T7h/wVvjxrrMNMVfnA6rn4//oHJkbc4tcvYr5hqwwMP964piMI9uZxEmDriXQrLDpr+OqXZo0yK1hD58de+SRb9f3H0x+G27A1J6EH5GD1d5OI05pL9A8LREzxH5OcZ9/4LLPf9Q980KkeqO+3Vp0ts9kS88+UrSrD3vj/+CsfhesfnBgmfRgLNPYoKv/QLeYKG8cXTm34w8OTwgW3vlMqP+T2j/7y54NqkzvI+ukf3gr3PRtXdk29ofdaS/bNcMLED7pD5w/iAsp9z04Otz4xua7eKbtVcX4m4vzvPM4D4ns7ay3RL6y1VL8MV/g4sX0jfh0Ednn8xbh624HT99abI5s80Y/oX7Qv7u+2Vn6l/9bog+PgZ9F/n3g1rvZ3yPveOlEWbmIEa+CPkA/7Qz/aE/4IbG59YlInHIcOniVr12X9jfovEF/Q3MTZuqz+T7xceZJh7UncrF2gPwL9F5MsPAEpypfqR+An34tPr9A+0R6Aw20Rh3rtKCWf+ukYDWLaofBoPzy2W21Q/OpWfrEJ7WxqQnUFm0JAsk/40/hwz3+wl7QSPMl+Kq5g73vpqzPcyglnMjrmV9KERw2PO474EptFdjzlprfDTyKBYnjujU/CXr+s7SHGigtWMK8aUVt9PO2md8J2qw2sEm7kvTyS2ivvji9odexFPXPX+SIxnZViI5n8OBz0m/gFkI54HL+2SL9w1nfmrabByQYnvVJtl5t9bbZw6LDaS2Ug2Bv9rBLv5eeEmIsfxzIxeIMsnxV1suHHv50QHnvJfFc7PrkZOv8s4cTt58kGbZvWnoPAXz7q3UzPLVeMBHuzmo4fR4K92Wmv5ep59f4LViYbVog7B+4HRX0++NB8Vznqs+aS/aJ95q77CVJfj+HrzRl2i+SjXiAu1h5F7eSmkQtlBJ3yMFna6dz8k4RGcvCZvpujHBuOvu6tcO/T8cszHU/Mtv/67HE/e22yYtPyHP37j383PntSyXw4YpvFWbvOHxaPW1SKAoj2j389rlre4oOjT+wyX11svV/79kMcv7fuHPFniutjDr2svDuP/HJOVW9H1uuEb8+ba0u5TPHC1+vEHebNJg0+nb32/tYM/W27oD+gPaF/qWcX6PVobKPHXPdmJ//37Rz9yG1xkrXfN+ZM2u2vkWSfFvs0lq+jVoxtPzEj+8P2a84dfrhRkwk2fmjGdhw8f2XCp+HtD+KqQUc4YPN5wo6rVzrAp+LH/ve+JD5Gjt/G6lP7Rl9cRcBqApa+KkHxwmdG849RRy1M986Ow854rRPx2fEXY8K4+F4D/X//TQeFb8fVUIZUnsvuAemMK5od7efs3ebPkQIMjAdcOa4aD1lfjatyv4ikyIb1TnilerlpJK+HG/IPgr3BiZV4L7+ayZ0cEAk2SDTI/NmOYP/wN2+EJ7Ca2NHusYp3jtPHicsus7rehc+RTQlbrTx7GOkI9sYnv5qlI36jjs4TqZRM3PtrXF09+U9vVaPnj6uJ1/4wv+pbjTQnP4r1yFZFoz5rLzUgnPTt/CN+k7R6ClwexeSiQf+3YgI3W/9G+RnfP+41/ssheYJ9xO8nhH/8N65gd+A/fP05wu5xVbdRAMneMRL8Dz6Kk5EO/X+515CweMcTlqL8mQ9e8UYWjfd0rokTn0Y/UJbz66inbz/0r+HrDYq6NybYxA5KeL+o2rGjAgP6xRdD91+ooY7VenXgWLZtPBpXnPGyP/Bvhv6X3fNeuPyu+BSrIywWV9HRnhphzPQg/Qf8enyYiCcrMaD9nPO9BXL9yNg4uZstPtmoJzOzyYv4/KPGL7aPrF/u8A/iLXxmLP/YYY25wg83btwH0f5ljn3wQzNlEiLN5ivOHo7aZp4Agv2Di16LT0H6xKciU3QUDr3GD+4+Jk+w1z7upXDyzvOHdeJKKcM5t74brr3/3Wq7uO7AL1VXdO+Jj9mPvGp88HJ+FVevL7/z3SqOZw9fIKy4cG0FG4P5AZePq8aj3a20aL84gOYJNvRhu9wittefbFVbXQfBXvf4Svz+m80VH5XPEgbPiS0T+S+boCyGs299O7ww9pNsi8Yv4mBtw48iuXjkxQ+z8vpHMnPdj77UaeAeF7cJ3B23cEyMnwLFtph141aCW+JWl1/d+U6Wb6tVZgsjh9V0xAr2N+MqvO1X7j1mkfgI+7OI6cT4WPyjOHlBbNxKEl+Q3COufNqw/Tmvh/FxSwtS4B2SHVarTWzwGPykGyZkZBr6rhTJ7zpL9wt/eWxStR4ed5CWn/3p7fDMuI/C4EGzhMUXmDnsEBcafhUnQyDlVk/ibo+bxEnOkVvlPydncWuUn/F94y/t3nZYfqJx+LUTsieNLG+PDQbFRZDZws0R33v+OyluBQlhzHufZPU8cuu5c6vo50S7Xnf/B5n+Sy7QN/xq77xtEf+XKAdPBOBnWEFdIq5Yn/THCZndtv/6bOFHm9Tshm0hJ93wVrg7PvmEPphsrbP0gDDunU+r5eC+9/sfxkkjcAROm69QecHI+j1sCxtgiwvCLY9PDLc+OjHT+97oFzZ4XH8xfHCuDSEt9IQ/4sV+BPgjZP/w8rGZ3sAb+dAuvL/BX44w7Qn5h188Nmsf1H+2aKfFY7uyoaz+6APYLqCH7TcoD23z0bgvHwF4YXuHDY36Eab9yxMTIwb/y+rv9UUfdfhVb1TxoH/pKL7D/mhGPO601jzxs7bTkWCjce6x4aCwxhIDwg8ueDleodOrrVjrWnjM6P7wj+MWRzOohrWOeT5slk08a1s1nh33cRh+QdzrHNvH4kP6xa+H1F7aOvFPE8JfHn0//OO4xaoycPLLSNh+dQdWXyvt6dw9FozkoLa3G4Pq/r+CzFp7+9rQvuG83WqyIQf6MGwev+t55NY1vUCw1zm2Fg/99thw7vD9uHpoQ01GzZ+/NrRfLCu/GjziitfDYy9UBvs9Npyrk5xbHv8gnHQ9vjbEUJG3+JC+4bkxyBe/GBLf3D50i9qKMQj2hie80JGhkh74ArNU/3LFfl/OERriCwFpDOP+7mqo1a9ya0q47YihOSJ6zb/eD+fcMqEjR+f01h4p/VL4jrhiTMQtst8u9J/Yg/23wxetao6Tn1w9LreVDz8WNi5OLiZGUlwJNX29j97z9OTwk9+Nicn6hJRtNznlpSgHJLTmb1Zfb++Kz+PJQzo99fHtp4YFUlT09W2jlqZWH6TunK7mj6k6PffGx2FEfHfI44O01h7+GmVVwpRw5Lfmj8S2NmmrtNu3q/FfG9o/0U7S9vb6W1nrLDswnLJjfvJ8zb/ei76IrxRVAp4inLIzJhG1fmJifCqxyckvdaSYEtvAQrl4RNTaSAXPPxy0cG6SXelr0E7yeOtaeDRq3+0cv+Pa84YDN6m/xa6jYZU+dGkFG3vuzo8/RPPy25+E75/3QqUrjT5X9JQ062oVL3yiI/DpGo/w0Hb0j/tOWCrXuNY66unMvn87cvEwoG9tNel7F7wUnh8TV52HzR92+Hqt0W580nNxNfd/wcv55V1vhV/+Hb9gWGlP537/y9nKIwt7JG5H2P+Xr1TjcX+lOJifu3t+Tzj1QfywlecMR25TG6Q/j/u21zn6mUxfxAP/PTaaJ3x//fwK65pHP43onH1WXDQShz3yZe1/2avh0ecrv+R4xYhF4g9U1VbcM8J13kvZmFyvf9h61UHh0K1qe7s//vTzsMHxz3Yqn7gMnnOWsMBc8cWsuK99yKCZw7pLD8yVSxwh4KjtBsdV7vyKxKj/Tgy3RLJ+z1Px1wORyPVPHnckuTo+jfjLI++GZ+NnTKkH7qfy+/gUviOA2wvx6x8Q4sr3+RmPFxb/dtQSyFENP7lqTLhndPzyTBTEfDwiEfwDn7FbIhLvwfEpxbAVa6v98KcfRn+C/K8tFm3r/Gjsu5+GX94ZV6Sfmpj5K+Xy+P2E30DmNfe9E0n/B5mOqfp5vx/xq4jFixELg8M/Tsy3MeD1WIefMR0K6CTLpDvQtbtJsc199/xXwhtx7AJeCCn9rHzEo00DvwXmmDkMnmvm2Cb759ol/Y35Uu2krP6Q9avbK7/c6X0X+m975oth0uT4hSKj/zrLzBZO2cVNfI1//cL1I2iXu6FdxsD67/GNdB/A+CL/UnwE0fitxcnia+/TTxRf879WwWendRcIB8TtnM0MpQk2yTX2bj31+sdh9/MwCEbvqgY0N11X4ci6L+Exo+HxzxOXq1Upnq1x1Ojs+ujtF4xkrkakr/3nO+Gsm8eE60cuGR83z5ylufu/H4TDflPZA+3lXHrnm5FgV/a3IvF5P1g0G8yzjPHPIy9NDiMufZGX8Ri3OGC1bI9FzL2aPogftsqccUtXbfD9PC5hr/XTpzrSV9rr9zeaL/xgg9oqNyIrdcq355WGDoxlLZwra8SvXgqPvDApu9eoPpXutHN72CZ+GumwLQdX5YJgr3/cf+J1rfyBkeQcOGyBuFVhQBXLagZ3cumdEyKOWDVP48PkIJA3P/puuOYfb2UEkuWl8LB5Lvk7nkC8k8kv09/tuNbc4cDN8tsvTrh+TLglEvYy+Vn2gFlnCrcfnSeeh/3u1UiAsbKPUMFr85UGxa18c4aV44SoXvD+dPtRS+cmiDYvcLr2vrhNZixW3SthgTlnDX8cmX+awzhMIEdFkn1pxGrcu5903K7o5/2k5kM1e6fTTI5y8v7zzxOXZZHZsSarXPvx8ngNf4Pd1o3kFdtn6gXrb8i/0tABndpkRa/G+tdkdU1/j1dFDp66TKnTj9Twhs8c/a1aP4H6rnEU+ok83roWHvCNSqj5z4xyvcs6Q8IBcetkM0Onr4hA+E3x0e7Prudj0RAsuUb8U69/FIafEwfB+PZ5fN24+la5roXHjO4P9//sq2gC1fD1wx/P/H+dZWcPp+26SPU+yMiJf3gtXLl/bdXx0N+8HO4eXfl6xj+dnEvveCNcensk2B3t6bw9F8sIJQU+8uKksN8lz1fj8Tb3SovNFs7/wVAmyY7UB/FbrDpPOOpbtS0kINhrHvlErr3+4Bvzhx9sWFvlhpA1jngi+4qAbc8rRnJ7ftTJhv0ufSE88nxcCY79gK/P8bHutzwcH5036B+2BcHeuvbyXkawj3myWv6Sg/tm5YL0lAnA8ZcRR9Qf+mNf3YHD8uTBygEZHBHr8fTrkQB19Gc/3eHLkaQWr2TAtvtd/HyYGPfzNqofJgUet1siQT/+2jjR6kL/iU8N3n7s8lb1YP0J9T1/r8VzPpNL7C68Py0Rcb4g2tc+hXFZwvHXR5s++FZmb5S3+apzh4M3H1yYB9ge+uuX4gr1pKo9fPuBT9t44Hn/ySvkioZ9Ho5+5vHy6aw/XvnDJeKe8dp7ETc/8k444bpXG9oLT0Yu3HPxsEB8OlImeH9L2bus/pfeMT5c+rexGb5l2j/x8DhQDuLhe9CJAXYfETFn+wDeK8d+5LxEP0L5/IqErjW+w19mVH/YZb2FwoGb5xeb2G66e+xEsD25xt69o+OLjTaAYO929pPVx3qMs48/cE/XsT/HRK8jCI/2x+P+U1eiObPj6j95pHr992NXiC/y1YjgJX8fF/bcqLJ6CbLxjWMer/rDv07Jy0HaX96GL/NUxF2w9xJxYKzt9XzkxQ/Cfhc+W41HqnWXnTOc/r08wbb6bAmCvV3txTgQ7DWOeLSqL/zx+xsPqerICMqw/goyf8GetckC0u578bNxBbuyHaBRfZDeyuP1tmvMEw7bpqYjCPa6Rz+G6Cz9rw9cJkeURj31Xrjp4bfido34beS34lY1pz9wvPRv2FtcK2/wXLOEPb8xJKy37Bw5+2SJ4p8x73wStj0135+hvjutNV+Wh+ns8eZH3g4nXPNSzh6p+sGeHhvI+W5coHj29Q9L5Ud6bPW447g88TzkyhfiCjZWwkP4wTfzdhwbV44vvX1ceGbMpPj/w2wyZu2X8qfZYhk7rjt/2GLlueKTgvyLelkh8Q/0hjwE1HeBuWcJO689fxi20lxJbOH3W5/yZHWbiW8/+14SfQjkuUNeCi+m8fh6XK0/lmk/Xh6uz9+nc7u76t64Ev/Ox5m9Gvnbiosn2klHHX15Xn/ru13R38s56+bXwtX3jM/sk6rPvhd1bMGK9QXeqbaNPsDrq+v2H784vtj2ljU+0/566/V31ls4LsZMQ4J91q1vhd/fV/tMUIpcA/yMYJ/5WGycfWIjjO/z6ygceokfPHDaqux/suPqhzxQ9f8Dt1kk7BxJGQNeEhvYb6bs8uaH3wzHXfNi1U+8nEtuHxMuuS1+G7kDxwv3WSo+bp6doiKRfT/sc+HT1Xik+8EmC2XEsZoonlh9hq0+b/jp9otWozOCfdiDVX3RbvfceMFOMlY79MFcOUi30tDZwoX7LF2VhZN9Lno6PPpc/HZ3jL/j+JVyBCulb6qf+NYa84afbLtIVS4I9jpHPpyVv8Dcs4YbflIjlSDVu54ZV7dNf+Nx8jgSTx6HxVXXXdZeIJL2/BaKXf/vyYw4Mh2P+PLJel+ZKxy85Zdz9YPCKZyYzx5/utPQuB8+v0gBUr/vRf+Jk4SPc/Wx+Qb0nynuucXnHuNPi8cV7DtOXLmKE04OueK5cPeT8ZvFMd7jsM3Jj8WJw6dVO6Lex3y7NhlrZJ8l48u5e268UJxgzJkrsx6+60acdl57vpzfIrP1E+/3+1z432ySZuv9r9NXy5Vp85dNd/rwJXK6oy1udcoTVTytHOtPVr+x0Ubb/Cw/zh2z46I5W3o8GrUTW66vJ2Rdemv8Bny0py8H/rLNyfFpmetn11tuUDh9t/xWne+cPTo891p8apDwi5TdV1x8jnDh3vntR7Yfsfj48nUt/jOj+Md3NhwaCXZt/M51Qt28qK5gn/CnNzve1K9IKiLXiH3qtQ/Dd8+szXA5s9WxMsMVDjMuDg/8fI1cU1t15D/joFep75JDBoTfHvTVXDwvRl7+dLj73yBDlZU/L+fiv70WLv3rq9X4g7ZdNOwUiaANW5/0SCRNlRftQL5Q1pBBtRcLkdbqs9Vq84Wjd6ht6wDBXv3Q+6v6Qu8tQMJ3yA/Qthzqu9IScRDeJ7/ndZ8LnoqP7t/L5J2+x9IBg70NF9/2Wpw0xB+k6sAHR+g9e1zlZz22XWv+cPi3asQvI9g/+VeGw8qLxTL3rZUJcrD3+aOr8lB3YMBJDMq2OCKe5bAeOA7sO1O488T8RGmfC58Kjzz7XlyRjV/iyEhvbQUceoM8njE8T0JWO/ifVXtZ+ba+uA89bjgy/8QCuoL0/e7eceHmB8ZX9QQ+S0byv0W03QYRzw2OxmQn/pJjvy+Gu07ME8+Rlz0dRo1+O4v3/oR8FXJeqceFI5bLEd+MaJ03OtMfeEz86LMqrlb/34xcIdMH+iIQ3xr5z+OE+j54Rr6NWD/xehJ3i1+9/GXT7bT+4HDQlotkOvPPXU++Eye5z+VwgTz4GfzY+xtW/79zZtxS1eG/wOnGI1cs9Dekg//8+Yi8rY+PZd744ISqHMrz9QS2l9xaaS+pdnnmn18KV90dt5AYfc7YfclMb9YRdtzwSEyiG9udcny9Icv2I0yn44w7rtFfevNx142Ghh8Pq70PxDY1NceMYHty/aP4ozI7dfyoTEr46Eiwdz39gWwvDmew2NOFvTm6rsxohceM6Q8Pnbl2rkms+uN74mBW+6XHG49eNX4vOU96Meitd3gk4qZ9eDkX3fZKuOSv+EpIxX+GRYJ13E5L5soCIfvt3XH7Qxw8119u7rDUgrW9lUxo9dlqjQXCMTvUtnVkBPvge3P6YnC9aMRXmD07gljc+e+3wuz9vxjOvP6FLP1KMd3FLt3e5/87klL8YEyfsOSC/cPvDs4TCwh75vUoa3T8/GAkBZC35Srzh9/dMzZc9JeXMjy2W3OBcPj2NYIPgr3WYfdle0SXWmhglLliTreLbn0lI0P4RNnIrRcPQ+bKY31xxPHiiCPa3yX7fzUWOyX8+cHx4eHn3g3j4hcksF1kl/UWDDuvU9ubjgJWjnaEfWCXG2P6O5+cEB557v2MjC0R9dh53SFhy1Vre9XHxonOlifUVvobtfc9N/1y2Hvj2laYXKXqXNCe/eNXRO4+ec1cyoMvfyrc/fhbGf7e7+588q1IxqKvRBywt3D95efK5X0kEsq9zvt35m8rLTFnOCb62lX3vJ7h9Ezcj46V+5UXHxSO23mJHKE87upnwo1xQoD6bLD83NEfX49bJyaFp1+bGGaLkwNgdPA2tQkTCt3l549m8cTXKrJ31AHk1uL34Fnr2CRxUvXv8FD0M9t+kP7hRDr6I/S/6pCVO7XFMW9/HCclb4b349c40I7gj+Pe+Sjsee4Tmf53nZSfHABDYAl/22eTRQIm0TZYf6N+Xn/bbse9/VGGX0r/i+Nk9JLYLtif3PTT1Trp//Dz74aHnn8/a0/Aup7/Qx+0WbRdBtrd4g37X7Rfvg9ge2B/ZNNTP9ZX17X+X3i1L//53jeWCD/esvh9Hbahrhz77P/L56fc/dSkah7/s+jVCHMy+rXJ4bun/jO7wxmPic6dKr6ykpADxVwIn/bC56GzNzDWi8TswDuzFSXePHi7JSIZq720h/s3PjguHPeb+FKwCV7ORbe+HC655YVqitni1pIbj10jW22t3nQnz4yZGAf8yo9zMGqVA+7kadhqjcHhpzvVtnWAYK/247uq8TgZGEnRTccUl8P6rbTEoHDx/l/L5d3r3MciIXunWv+dN/hSJFc1spxLbC5sXb8Vie4R29cmEhnBHjmqmvquU9epiwEmL1hZZLCyL/7RipFc5Lc4MJ09XnX3a+Hnf6jsS/V2sens+cG/fDKMemJCJEPl/feYXZeJBDT/VMLKTJ0T/wGxjqMiFjYcBB3+XVkZbSQbEwI78XskErW9znk00x9E2tvWlsNzyNjl1Afij898FvbcfGjYe5OFGVV4ZDlM4PHd67zHwiPPvMPo7NgpTfSzR57Np0HCRumWXGi28LtDVsnJTl1YHW88ds0cTj69x9H6G9PW8zub3utv4yAL+l/8w6/l/JtlpI53xYnAyEueyEV5XWxdkRD+m7I/+5Gu+Heu4I4L5S/fPwi/zgj0pP/suvFS8alXfuzurFHX7uQ+01eGXEM8CPauJ+dX7uzM7QsdOhTNbBVfAUj4pGf+rewfj5zzjQ7vrhxW/OHfcitrg+fqG24+Nr/KfdClT4RRj8cvW8QVJawEoX4P/+KbOTkX/fXFcNEtlbf7Wf8t11gwHLtLbYuEzXDjA2PDn/81Nlzyw/yqMfRh/q3WXCgcs/My1Wwg2CsfcHs1nvpsXacc1g+D8CU/yu8B3vMXj4SHnql8VYLtf4uvDw6HbrtUbtWzqkDHCep68c3PZXhst/ZC4cgdapOAjz/9X1jzoDuylNBvva/MG87as7YP28oCBkMi3tCN4cIo+5IoG+HiA1bJxTGNPULGMb/BtpPKyou3r03Lc6t/amWP+BNfu9KH+hy23VLxW971P//Gsoj/wPgC4t2n5yd3B136eLjz8fgyW8Rp9kjAbzp+7SQZe+b1iZlv/dzgiInRnuc8nPnjKkvO3cmPWD6PkHHMb58K/331vcx/9hq2eNh700UZnTyijIMufjz+UiK2n6TxhQ6VSVpt5etR1zbgZ49EPyOexNe3Ifoj45F+yQVnC5cesHISFyqN8vc6+6FMfsrPmQ6+8npcgd7H1Bv+dnFst6wf/GHBufuFqw/7erINWP/0+sOvLoy+6/X/v2i3Rv7yu1GvhDP+EL9x39G/QGfUH23Wtg/WlfFIn7L/yrEfId7Wf718XCs+/+Te2k/4tI9/fHfTZSPB/hKbe1OOGcEeEB+nHbVtfGN+6fyLP0UljH417k372agYHZ/7dnx/Vce4FCA8Znh/2GuL2mprrGy4+CasfObbwS4bLRIH19pnvi6+6ZlOuHg5Dz/zdnj4GfxKW96PVl5ynrDLBguHVZfAI/4p8XNyH4Q/3/96uPGfr4fBcSDfco38jNvqs9WaC4avLJJfwT3pd6M76Qu5LAfkAKTk4efeDmPe+jAcc+W/s/SD5x4Qy8pvq7jxn6+FsTGNrz9W37eM5H6Dr84Xt2T0z0jwmEhMxr49Odz5xPhw432vZaugyLfduguHpeNKnQ0n/e7JeFnDYeUl54oYLBIxAJHuEx589u0Mg1GPj4s6fSnDgfktjkt+abaw/goLZPlmi/ZAvRBQt6df+yCS0zci5m/HOzX7rYf0Sw4KSy00e2CeiR9+muH+9Gvvh9/e8UqsM5741fSz+cveh96rxPoMiXgvFfWiv0C39+OLjThanAb0nTkc+K38HvB7n5oQRj02vqo/cN97iyWi7rNFUjVXJuPBuPJ71d9fjE8qZsnZD3a78Z/xs3WxHrQX8swen2hQn2eir70e0931xBuZv9l60h8qNsEqKPwTW4I+iNi+H+0zJunP3u/hxx7PvWIdbCjys7LpbP3o39i28fTr72d+8Ns7Xsr5MdrVyO2XyYjykLn6ZelQH+CFdgJ/ZLD+5vHZZ9himX2BDfwfuPzuzperuHj9H46TCO+P9CfrL5SH9gQ/9vpbPXz7sHZnukb9CNPpWOsnaBcdp64fbCX8vrvZ8uHgrbq+jY99QerYZ40jn5xy/u5D4q+hzZKKT94Dwd7pRKyEAdxK4G/T61p4AAH5A1aq1T7UH6g/UH+g/lDjgcaDVh8Ph2++Qjh460U4ZDXl2Gfcmx9Mwa8zdiWMfnVi2Pn42zqycAZDCbquzPSFRwUB+YP8ASs/DPIH+YP8ga2htoLHO2ofah9qH2wNPdk+dhu2YnyHaNFa0U04q36mryuyQLB3Ovavce9VbArRF7gRXdfCQ/6g9qD+IA4LsV9Uf6j+UP2h+kP1h+3RHw7fcqVOX0DqCi9Ope0ewX5lYtjx+EiwDbmmE+lYcSbhIBzUPmqDq9qD2oPag9qD+gH1A63aDwwftlIYue3QFE/u9r3uEexsBfsv2VvGAKuyUlN5C1zXwkP+gJU7tQd8hUD9gfoD9QfqD9Qfajxo9fFg+JYrt8gKdiTYOx5XW8EmvS+amSi+goDwSa/gyD/kH0BA7UPtI7XCqf5B/YP6B/WP03p8aLEV7M57sCudo5+p+UFD8fmZnPDJD6ryD/mHXflW+1D7sORC/YP6B/UPIJuVJ0PqH5vVP2oPtvZ8V18ozTuVb2S6Fj6WlMgf5A/yh6IVMN1X/6D+Qf1Di61gaw92fiXBr6zoWvjYlRb5g/xB/lBbeVN7UHtQe1B74Er89O8PWnYPtl8B0J457ZnTnrniFQG1D7UPtQ+1Dz9u8lr9g/oH9Q893z+02Ap25z3Yfk+QruOjN7OtRXgID/lD7XG02oPag9qD2gO3Z6g/UH8wPfsD7cE2ZJWNUsfimR5XRHSsDWLyF/mL2oPag/oB9QPqB9QP2H6gxVawtQdbe+i0hw6ddGXlZfrvIZM/yh/lj2qP6o+wEqz+WONB18aD3eJ3sEdu0wI/NPPY8++Ga0a9wq1iOgoBISAEhIAQEAJCQAgIgbZEYLYBs4Qjdly6qbp365ccP/nkk/DMM880VREJEwJCQAgIASEgBISAEBACPY3AQgstFOacc86mFiuC3VQ4JUwICAEhIASEgBAQAkKgnRAQwW4na0lXISAEhIAQEAJCQAgIgZZHQAS75U0kBYWAEBACQkAICAEhIATaCQER7HaylnQVAkJACAgBISAEhIAQaHkERLBb3kRSUAgIASEgBISAEBACQqCdEBDBbidrSVchIASEgBAQAkJACAiBlkdABLvlTSQFhYAQEAJCQAgIASEgBNoJARHsdrKWdBUCQkAICAEhIASEgBBoeQREsFveRFJQCAgBISAEhIAQEAJCoJ0QEMFuJ2tJVyEgBISAEBACQkAICIGWR0AEu+VNJAWFgBAQAkJACAgBISAE2gkBEex2spZ0FQJCQAgIASEgBISAEGh5BESwW95EUlAICAEhIASEgBAQAkKgnRAQwW4na0lXISAEhIAQEAJCQAgIgZZHQAS75U0kBYWAEOhNCDzxxBPhsccey1V5yy23DIMGDcrd00V7I/DSSy+FBx98MCyyyCJh1VVXbe/KtIj2kyZNCg899FAYP358WH/99cO8887bbc3uuuuuMGHChGr+ZZddNiy33HLVa50IgUYIiGA3Qkjx0x2B3//+9510wICEgUlBCMxoCIBgn3jiiblqXXTRRSLYOUTa++Lmm28OV1xxRbUSIG7HHHNM9Von3UPgkEMOCS+//HI1M667O3k59thjw1NPPVWV9e1vfzvgv4IQKIuACHZZpJRuuiCAVYTzzz+/U9nrrbdeGDFiRKf7uiEE2h0BEex2t2Bj/T15Q47zzjtvqlZcG5c6Y6cYPXp0OO6443KVxKozsO5O8DYSwe4Oir07jwh277Z/y9f+9NNPzx6jekUHDBgQLrvsMn9b10Kg7REQwW5vE2JbAbZ+gPBhy0KK4Pl+rX///uHyyy9v74pPZ+2Bu190KVqIgW2wlQTH3XbbLbn1QwR7Oht0BiheBHsGMOKMWgUMTrvvvnth9abm8V+hUEUIgemMgAj2dDbAVBTviXPRCir6NqTFFgTsEx4+fHi3tzJMhbozXFZMbDBRAdkGuQauWIyxAfcmT55cvYWtOam91SLYVYh00k0ERLC7CZyyTXsE/D5FDFZ2Txz21oFkKwiBGQkBEez2taYnZUUEu31r2P6a77DDDrlKiGDn4NBFExEQwW4imBLVXAT8Cyv77bdfp/3Y2CbiVyiaq4WkCYGeRUAEu2fxbmZpItjNRHPayBLBnja4SmpnBESwO2OiOy2AgN9Ph8eoeAkIe+wQxwDSjc8xpQJekBw1alQ1auGFF84eGWLf3XXXXZftv0MkZOMRIfbiFZF1ykJeG5gXevCxb58+fapJtt9++06PH/Fo2D6ixKNMWwfUz77YOc8883TaWwg9UDc8EkW5CNAdK2abb755pzIRj0en9g17louvtKB+KBcYDBs2DMmzgPsoA/9tQL3xFZfuPkGgXDyRoP6QjzrgyQTw9AH63XLLLZnd8Ikzhkb1RrqiukMPyKQ81Au24NcCoBvikY5+h3oDY2sz6gKCZe2fwhhpWc+Uz3WFYEMn2M/6AeQDQ+8HSIs2ZPVDfT3WwMJ+4QLyIKsrX2MAbpBh9UKdIQfYepyAA3BFSJVv47NE8U+RTRnPI3RBW4ENIZsB5REn6JYKyIO8qXaP/PR/1gfyvT+zXux/UA7lskwbx3s8omz4oG8r7HuAJ859SOGI1dqUbdj/UVcvK9XfIA1wQx74UEoHKwflou+z/pfqH+Gjb775ZjUr2xBvwI+L+kfagWmZ1+LtbQn9aX+mR37Isk9M+ZKj7S+RDvnRF8C3FYSARUAE26Kh85ZBAIMnBhUGdHwg1/5+vc9boSPEfwaQT3SEtnNmHI4YINC524BBAW+m24HZxvP82muvzU79BICdMtNhcPAv4mCQ50CNdCAlGIgY0HFj3yCDrxfv2yMGPNTVhtSAge/F2kmI1Rc6eGJt5eGc9fb3i667iifl+O1CvO+PwBJ154DJ+DJ1Z1ocSQTr2d5ixbx+dQxpHnjggdzEhmlxhJ6nnXZajpyUJdggDUW+zDLgA5ZAw7+tvZHOPyL3WIH8WX+k7KIj2gpwg61TATpBdxusDiBA/msQNp75vJ4pe0AXYFSv/YIgQb71ma76qbc7dbRHu13Et2EbxzzQAbo3aoNIj/7BE7wUjrD/oYceWmibFM5eV+pnj6l8Np7n/qmk79u60z9ybEAZ3g70iTJ1QH6mx3nKv3x/iXQMvi68r2PvRUAEu/favqVr7okqOmaQp6JBI7V64jtVpEEHXi94YpoiJKn8JJp+AuAHzhQpwsBuv4ji9WbdUa6Pwz18gcCuiOMegs2Haz9gYHLiV3M4wKTKgQxfFuuNuEahLGmBHCu3LLlm+alJl687MC8igFaOx4dxOELGueeemyNmfoAvWw4ICkMZgp3yI+T39sE9+1QCdUbbsv5i8Uq1L0wAQELLBMgHgWvUzrwsS9BSOth45vU2pe8yHrrsv//+neycwgj1Qz0Zitq9z0s/9XanHHu0fYFvXzYOeaB7vcmdlctz33elcEQ96002rC9AbkoG7nscUvZBOh+mdf/o7UCf8Hh7vXjN9Lj2/lWmLcNvUmMR5evYuxAQwe5d9m6L2vpOHZ05OmYGrNZYgmAJBNPgmOpUIQtbIDDQgKT41SE/0BV12CwHgxUIIFel/eozOmVLnosGbtsx+46dg3hqdccOqr6+fkXey2UdcES9IR+rixhkPMZ+dcbX28oqOk/V3doD+WB72IX2TtUZeaAnJlzQA+nt1hfIsbjgOlV3rMxyOwxs6GUgX6M0vhzvL5ABfYEpyQ3sZP0XaSxBaUSwU8TREoOUD9qJgI9H+ZyMeZysXKRrFLwPIj0xRFuArXybQxpbf9/+fTyuERrp6p/AQA/kgR7wK5zjyEAMcO3t6HHw/o96I6B+VibaIHwVwZ57nHy/4+ORH/pDFvwIGKIs60eol7VzCkfIQbuBHKzGohwrA/F2QuUxhJ7ACWUhoK7QAzKhV6OQ0on9G/Km+gjctzrBbnbrhn0Pp8huKJd5aCvIRcAK+HzzzZedo36YZCD4cnAPNoQvoP7oMygTcQjeTyp39be3IiCC3Vst38L19p2sfQQItX08Ona7+sSqpQYpO5AjnR9A0HFaQuw77CIyzzJx9Hns4EDiisHSEjpL1JgGsjBwYUBD8PX2uCCNX/m3ZacGDMhH2RwwQd5w7uswtQMHBmJOQqAngiU8lTuVv9QBV77OuGfrhGsEX2/vE77uIL2QzXqn9INcpMGgipBK43HxuCGf1zdFMqycRgTbr+h7coYy4TPWvyx5RLzHA3jBt+3WDNinK1tDINf6Lq6BHepPnHEPkye7/Qv3bLtM4WPjkR7B18FimLKVtSXygxjaLTa2PXk7lmn3KZ1StkE63zf5dB5HHw8ZKZysnql4W0fI8BjgnsXaY2z7I6TtTvB1s+2D7dj3j7ZeNr/HxdvN+gR19WlsfZkGR19332cgjdUF1x5f3FPovQiIYPde27dszX2n5clBagXOD56onB/EMNgjnQ2erCDOrqj4zhjxWP3BQIP/qeBJO8kzVr3w+BwBeTEAcvWIHbMnBkUDC2SkBgY/KFjsfFxqwIBcBA50lavKX6zsoO7Qtauh0UBeJM/rQZx8+pR86xO+7ik53u9SZMITVy/H+4tf+afeXo4lCo0Itq8L/YuycfS+74kG/Aw60P+QB+0D9xHgG/BjTi6ymw3+WP9m0pRumED579tbX04RQxtP2R4HW0ffri2+zO/LsWm8LyBPo3aPNF4nKxPxDN4+Nl0KR9uOKQPHen7k64f0KRy9z1qboQ35PfvwCTz5ge/biRPklwm+f2QfZ/s+3z+yLdo0KIt5Wa6vi/WJojQpTJDW25I6UE4qjbWjTafz3omACHbvtHvL1jpFlNBJ+oAByoYUkak3iDFvahCyBDu12sa8GFxQLv7bgcYP7tTN3sfAgIkCHzGS/PvJgyWJfvCgHvWOdoDxA0a9wcBjZ8soqrdN489T8izOPj2vfZ1tfZgGx5Qd7cDp656S0500HsOy+vqyrJxGBNuXYXEoOk/VN2UT5vfEhffrHRvZwOb1dbC2KivHY2jrWK9uVg9/Tp/sTruHLK+Ttasty+tn06Xqb/sBK8cTYPYjSJOSY7dTUI63hcUxJYP5cMSkA+m7MhGz/SBkcJJq78P/0DeiP0RgvXz/aFe/ka5eXRCP4NNY36ukqPz1trS4MF09OzKNjr0XARHs3mv7lqy5X90oqyQ7YJu+TOeXGkA4yFKWH8R4n0c8XkcnTZLtV6C4XcHWDQMDBgvoyIByMMHgPVun1Kof89U72kHBDxgk/kX565EM5IF+/vF/WVnEpCg97/vB0NaHaXBM2dEOnL7uKTndSWOJEfQoq68vy8rpKYJdz6csdqhXmeDJD/L4tkQ5HidbXiNbUobH0Nq0UZulDH+0+jbyf9/uIcvrZO1qy6rXN1miyTxWL97D0cvBkwfojZDCMSXH28LiCDnokyDTPu3AfQb0eyNHjqzuXeb9oqNfhWZf4PtH6G8/F1mvf2RZjeqCdD6N9T3KwdHb0uOCNB7/InsjrULvQ0AEu/fZvGVrXG/AL6O0X80o0/mVHYRAmjHwgUSkBhrf+fqtDRjY2LGTOPuy8bgXZXBV2xNg5icWWPnhyzm854/o8Ite2vE6+7y4xmAInTDIpurN1adUXnvP2wJxqcHe5sG5r3ORzh5L5LUDZ5nBsjtp/IBaVt96j/a7SrDx6BokpV6wfsB0ltDwHo/0UV6XOTaygZXhcbK2KivHtzHrG97fUB+stjYKkGED/B9+WtTuvf97H/L+QdleP5suVX/ft1FOvfJSclJtztvC4shy0DejD8DeeWDiQ1f9xdsOenFbDicJfqECW1egA/tHjz10KlMXn8b6nq2XxzaFSz07Wlk6750IiGD3Tru3ZK1TKzddUdR3uGU6v7KDkNUDnbxf0bEDJNL6FTR0ztAHweppO3sQasgmkfX7LjkAZULiH7tXkvfqHcsMGPXyg2RgILQvz2H1yr4UWpQ/ZVtfv1ReT0T9Z8SYB/awL85xkGZ8mbp3J423u7Unyra2pi5+BQ/37eDdiGB7TPxEjOXUO8LP7At+IEggUfQ9r1M9WYxLtSVbL6bzZeO+JTkpOTYe6VOTcVuW97euEkCU4QP0btTuvQ95/6DMen1Tyj+K2rrvE6y/pXDsLsGm3jiC+AIHEl3GFU0CGG+PyG/bK7aEcLXa1sHWr1H/CPm+/VmfYPk+jfctpvO2TMmqZ0fK0bH3IiCC3Xtt33I198Sh0aDo03uyV6bzazQIYTBJrQ42ku0fl6MuXPmxg6XtxP2b834w9CuOkFm0RYNlIQ2DLQv3UgMG0wIXrnzzHo6eHDWyEfOmSEPRo2WUwdVGjzPkWfxwDRv5HzaxgzTSlKl7d9J4AuUHb5RtB3AQw9T3jW2aRgTbkxPgiPwpPwXumGwgDQPu+R8bAaa4D7xt6AppQj5ff68bbAU/Rlk22PqnyLN/wcxjAFnWn1EOXyhmOfX2lXt/99eU4f3R2z/VRjHZ9qGRHL/Ci3aG/s7aOIWBxbFR30advM0sjkU4wH7Q0QbU0/Y3Ns6fl+0fLZ7A2pJ63z+ijHp1oQ5l0iBtmf6gkR1ZpsSOjPMAAEAASURBVI69EwER7N5p95ardarDbrQy51epUCkMQhiMEcp0fo0GIXTGIJr2MTwGb8gGEWDwhC5FEpjWDkReR6bxhAL3U7qCwICMYmDDOb5tC/0wgNnBFvnLDBhIh4B6YzCHbA7qsBEwh3yGlJ6M88cUIUAa4Av9IR/648jBEzhiILcrq8gDnfAfaYGLDxZjxJWpe3fSeILlB2/qhTqC6AI76GyDl9GIYCO/JzfWD4Aly8FkpZEfID3wAtb+R2KAMUh22QA5/qsT0A1yEKCXbTfZzfjH6+gJJtJBBv5DBv77YIkh4rw9cQ92AN44wm/gV/ilTWCA9Aypdg/c4cNWf9/uU+0ZsldbbbUwceLEqt18Ou8DfiILvYgj7esx8DJS/QXbFeuJo/dZiyMwQb2hP/tV1B/6oY9h8E+MeL/e0ZfLtLbtpvp4pCvqd7xMWxfKt6vivId+DgGLHPw2vveflKxGdsyE6k+vRUAEu9eavrUqniJfjVbPUkTDDnhlOr9Gg5DvsFOoYXDBSgsGPht8B404/23hVPlI51dpcQ8hhVMlpvNfT1q8PqkBg1LK1BtpG9mI8nDEwAwd7BYTG2/PLREowsimt+cp7MrUvTtpPKkpixv1he+gXBBHhkYEG+lSBIz5/dH6QYqwWLxScuut/PqyYOPUhMim809qEGd1xLVvu7hnQ0qG92f0D5hw+8mZlcPz7tgx1e5Bev3KeaoMXz9fPvKkJiuU5Y/QBelBwhlS7ca2K6bzPmtx9G2CefzR+pCPK7q2q9NMgz4U9WAowrPIJ+vVhTLr9aH16m7jKKuMHZlWx96HgAh277N5S9bYr1j5jrZIab9NBOn4KaoynV+jQch32F4P6Okf3TJNisykVuVTZdhVHMrj0deL9/3Rk18/WKYGDMpI6cQ4HFFvDKpYCexKAAGD/nb/ZSq/JwIYaDEggzQVhXo6lal7d9J4YuRxg73hY6lJRYpco25lCDbSgQyDLDQikPQDYOe3hvgJH+T6tgjSBhnAt0yoZyuUBx1Qhg2eYCMuRcBwv0hGyp+hC9pSCn/IYrATc9zzdmQ6Huu1+6L2aX3Fp7FxLANHpEM/Us/GyIs+yJJr5G3UtyENgq+rxdG3iUqO2l/4MNJz1bcW0/isbP+YWnEu6h/r1YUa1Zvo16u7jaOssnZkeh17FwIi2L3L3i1ZWwz8IAs2YEWPjyTtfX+OQcTuy0M8HvdhAPRxuMdHgZSTKhsdKQMGaMjB0ZI7EMtGOqZko07IZwPqbmVj0Go0YGGQQD7UHecMkI3/KMcPuL4cDMxFBLmo3pCNPGVsQ51SR9QXj+at/rAP5ONxNM5TAY+laQ/Gl9GpTN27k8b7VNEAD9nQHbaCXYBfykaoEwj2mDFjWL3suPrqq4dBgwbl7uEC8oiJ9aEUlkgHu9oAHYCfDb7dIA5pumJz6lVUZ49TimCjXIsbrtF+iRvIjQ31/Bl18vWHHVgvHG0o8v8y7R5yiCGOCCiLuuOa8ThH8H5UuVv5SyyRx9qY7dDrzrxIC/xssH0b79fDETJQLvDAfwboy/J9P8M0jY4p/VL+iDrYekNuqh64X68uiLcBBB91omzgiH4XdUPw5ab8qyt2tGXrvHcgIILdO+ysWgoBIdADCHjimFr16gE1Wr4Ij1MRwW75ikhBISAEhEABAiLYBcDothAQAkKgqwh44iiCnUbQ4ySCncZJd4WAEGhfBESw29d20lwICIEWQ8ATRxHstIE8TiLYaZx0VwgIgfZFQAS7fW0nzYWAEGgxBDxxFMFOG8jjJIKdxkl3hYAQaF8ERLDb13bSXAgIgRZDwBNHEey0gTxOIthpnHRXCAiB9kVABLt9bSfNhYAQaDEEPHEUwU4byOMkgp3GSXeFgBBoXwREsNvXdtJcCAgBISAEhIAQEAJCoAUREMFuQaNIJSEgBISAEBACQkAICIH2RUAEu31tJ82FgBAQAkJACAgBISAEWhABEewWNIpUEgLTEoEDDzwwJ37TTTcN+N+dgF9Bw6+n4YhfQtt+++07/YJkd+TaPGPHjg2nnnqqvRX23XffsNRSS+Xu6aJ5CMCm+JVN/Koofu4dv9rHgF+4GzVqFC+zI/ZQdzUgz3vvvVfNtswyy4S99967eq0TISAEhEA7IyCC3c7Wk+5CoBsI7LbbbuHDDz+s5txjjz26TbBHjBhR/alhCMRPHR9yyCFV2c04AcE+4IADcqJOOOEEEewcIs27ALm2PzmNn8I+7bTTqj8h7eNR8rXXXttlBUCw//Of/1TzbbTRRiLYVTR0IgSEQLsjIILd7haU/kKgiwh0lWBPmjQpTJ48uUqwbHH+axDzzjtvOO+882ySqT4XwZ5qCLsk4Nhjj81Wr20m+5UPEWyLjM6FgBAQAmkERLDTuOiuEJhhEShDsF966aXw4IMPZv9xXvS5OaxWv/zyy1WssJ1g+PDh1etmnIhgNwPF8jKwBeT888+vZsA2EUyasJKNIIJdhUYnQkAICIFCBESwC6FRhBCYMREoQ7D9ynQRwcbqNggXSDi2h6y//vpVItYs9ESwm4VkeTkg2Zhg4YnEsGHDck8vRLDL46iUQkAI9F4ERLB7r+1V816KQDMJdk9AKILdEyiXL0MEuzxWSikEhEDvRUAEu/faXjXvIQTwhQ38R8CKIP77gBVg7HNGWHjhhZu+CmzLaxWCTVxwBCbLLrusVbN63h2CbfGEoGmFKcqhbbGFoqgO1cqYE9bf3OpSftZx9OjR2RdcUH6jeiJtnz59Ao78MkhXdIaujQg2y0Daevp09SVHPC2x25GK2hLK7WqwOiNvPb0bybayymLLPDh21y6N9OpKvPfN7uJBOdhmhK8M+UAfxhH2LOPDXoauhUCrIiCC3aqWkV5thUBqSwW2S2AvKwZNGzCAjhw5MhtMEIc0JGlMh8fy/pN3SHvccccxSXa0L58xIqULtngwFBHslHzmsUe7XcS/EGfjkCcVj33aZ5xxRidcMLhCN+BmQ1cINrY1XH755Z3whDzIhXyU05WQwhOyUrYFScAedGyXKQrYfgGS6m3O9I30RN5bbrklgHCmgvcJpLviiiuyLR9FefCVEBKgRgS6KP7mm28O1113XSe9iupTlmADJ5QJ3Hwog7fPg2vggE8NQibIXSqgne63336dJsSpdoKvqBT5HvwBPu8DdIAd69mSdvHtKPW1HuyTt59PhD2R3wb/1R/kAYYMRXVAPOyI9m3T436RP6Adom4ImGigDgy0Z1EbQDtFH6ggBNoZARHsdraedG8ZBDwJW2+99cJDDz3UiWxQYQyQGHQ9YWY8jpCBAZEhNbB7MoW0XhdPeqcnwUad8X3lIlID/UFqMJgzlCXYnmAwvz2CHGCg9yTBpvHnHk/UAUSmiKwiP17+9CQb6VMTC18erjEJgG1JepkGugO/esH6BMqEj9XDG7JsniLCxDJT8fAx3C8KKbKHMht9pg96Q/96WKNM7zNFevB+qi0xzh5hB/uJQsSl8sLep59+us2aO0/p518QzmXouKBdMHnBJIkB/gt/t8GTZ8Rddtll1QklMNx9992rWbAqbXW2hLiayJ2k/LKMP1iCXaad+j7LqaFLIdAWCIhgt4WZpGSrI+BJWBl9MVg1Ig52hSk1sHMAtuV5XfxgNT0JttWz6NyTmjIEOzXIg+Di0TQIqV0pw8okcCsbPJ5l8qEOIDc2gMxghbBsgIxzzz23SpCw2mq/7gE5IEmrrbZaGD9+fEaisYXC+kSKNAEXEF6QV/wHNjZPCkv7netUfJk6+VVJlFmPYEOvQw89NNdGQCyhP9qC3S6C8m1baaSPbUvcvkBM/ASmzES3UVv2hNgTZuhbz5awE7CwwdYXWNnJONPZiZ6tM+IxUcTqOkLKt0CKgQl81rYf3LMr4yl/8HiQYKfKSdXb91mZkvojBNoMARHsNjOY1G1NBFIkjIQCJBrkyg/cqAkGOQwmCCBD9hEv7lni4wdIH49rBK+LH6yKCDb0xECO4FfWQTK4qjzffPNVV4D9iqovy8dDNgZUkAWQDpQJ0uiJJ7FD+kYEO0Uu7IohykB5liTw0TvkNwoeT6QHYQChQR0gF2TH29fqkLIdcIAMEBboCKLCR+rUyeKJMqx/IL9dgUQe1hF6IXj8LanKEsQ/0M3aNEWYGhFskFToisf6qAuIlF1xRVmeZDYi2L6+INYgjAxeT0+EmS51RJ2RHzpzzzPTpWxl65+KR/1BVtFGYAP4l58AWJ/zdknp7m3pV6gteU4RdtTH+o/Hi30L7LX//vvnJjK2/aXaD/OiDC8X9xBgb9iMZBv4+AlfkQ9jwujtUpGqv0KgfRAQwW4fW0nTFkbAkzCu2FBlDJZ+halMGjtApgZ2O9CxLK+LlYE0RQSb+XFsJINpPVHwZfl4EBGQQhJAyMEADmz4kifuWWwaEWxPLmxeyELwJMDrWUmV/uuxQB1A/kAcGFL2taTJr16nZECWxwtlcCXcxwFD3LNYUh8ePSkDmYfPWN2ZlkePFe5bgpmKt4SMcjxBxn3co77Qo2gFGz5htzMgr93ugGuPOeRCfjOCx822s1Q79P6USmNleFvCLiDMxCZVB4+nLdP6F0gryb1tCzYN/A9kF8GvKqdIr7e5naj5OMiEDNTR+5mvA+KBC+qvIARmNAREsGc0i6o+0wUBT8Ls4EeFpjZNo0G7bDnTk2D7VUjq7AmHJUuNCLbPi5UvEAsbsLoM/BiK9GC8PZaxG9J7PSy5weqdnUBYgmLLwko+iJANXPn0EwmmQV2wTWSVVVbpRGgsqWJ6HJEHq604+pAiTI0Ito2nvEb+Wo9gexxAxICZD9DVhpQeNt6fg6TTN3COlVMcfbDkOFUv2sjm835jJyF+JZf5aEtMznzwmKT8C8QZdrVPQoiJnTTYyZ8nvan2A0xAxBls2Sl/sXgxD45FPgyCjacfKR+2+XUuBNoJARHsdrKWdG1ZBPxg2l2C7UmalZMa2FMDWSNdpifBtvWxxkwN0iQGXSXYVm7RuSUIRWl4vxGeTOdtZ8soK6OejVMr/SwbR5JQYMyALT/Qy5J7xuGIiQzIvyXa9WyBPI3ikQahXl0QX49gp8pAnkaBPtMoHbDEF09A+MoE285S9UqVW8/mIKxYsS6yC2yJdgqyzJBa1Ue5sDH3Z3PiZCdp0B0E1j4RsNuXvN+yvHpH69spW6XwgDzUAeVxhd2XkfJhn0bXQqBdEBDBbhdLSc+WRqDeYErFy6Txg50lpKmB3Q78ZctpRYLtV9HsI2wR7GOq+1FBzICV3+9N2+PoV8dBwLBiWi+P3c/biDA1iqcujfx1ehJs386gM0gjJhzYj47VWruSbdtZql4pQtmovZexpV31ho5eb6ycY2UbNkEAcQbJtmQafQjqZt+rsNttvMxMUIM/3SXYEAuSDX+07xP44rwP+3hdC4F2QEAEux2sJB1bHoFGgykqUCaNH+xmNIJtH01bo4Lg2VUtO4B3lWCjDLvyZ8vhOVbKsKpXJpSxG+TYR/C4BtFBvRC8jCIC4bcBIC8Itd+bC3KG1Vekt0QQ6REsgarcqawogzgij185tVtyGhHoRvEsD2X5r57YrRRdIdiwF74d3yiUeTHOrvhCHuqOdmcx9racFgSbdYE+tIu3JerNPfhI77EHoUZeTp7oK7Y9oS0BF5Jwv8fa9znw29R2HOqLo20/XifEpyYcuG9DIx9mXWwenQuBdkJABLudrCVdWxYBT6AsMabSZdL4wc7KSa2c2ZVHlINBC+TABisD96fnCrYlctQxpbMloI0INgZiuxoG4gwi16zg7QayAsJlQ8o2Fndv1xQOkOf3TBels2Wn9rVaQmjT8txjhvskRY0IUyo+VZ6vsy0D58hT9JJjCs9mES6vv7UT9EptxbD1S+lG7JCfwfuNL4fp7NHrhjg7KfGTA7QT7re2voIVYt4HGQbJxsQKwevhy+xq+/H5UUYKD9wvCt3x4SJZui8EWgUBEexWsYT0aGsEygymZdJ4UuIHQy8DgyFINgZXEIPUj4p4GWUItl/BA6nkr0+iHAzaCI309fHIg9Vl6AAZICv4nBuIgw2W0DQi2KlVX19nyibJwCpd2eAxRz7UAauHCKgDfkAGuNhgCSFWGf1qLnSADGKZIiq2HohHHtjchtQEhaQMOkB/+IcNKcxIilJ6MA4yUvHQCXXBETjApqizDf7pRT2CjXzeByGbvm7lov4PPPBA6V/+8/oDU8hFgO6pz0Zaf2wGwQahxIup3i5lZNsXZuE79DuLr7cvsGMbo28Qw1SZto0yHY5Ii6cftv14PJHO+guuGZCWK+q8hyPqYLe14J7FHNcKQqDdEBDBbjeLSd+WRMCTMEuMqHCZNJ6QejkgAnYrBWXXO3oZZQi214PyMaBj1QwyEXw6X5aPp5x6R7s9BOkaEWykSeEC8oL9tNjPDVLAL0R4HZG/XvB2q5eWcZbs8F5KR8aljtAfZIgEnFiCLGHCw3qBXIJkMiAfiDUCdbd5kBYEjMQM6ay+jQhTKh4yGgVP7BoR7NSqJspA3RlQBxBH7zOMTx09+UQa4AOc6SM+nyV7KUKaIpTEnrKs36VsmbKL384BWbCtfWJD+fAvEt8UYUU6+24D8+FIfew94AFc6GfExtYD6VP+kMIDaVkO5OI/fBVts54PI5+CEGhHBESw29Fq0rnlEKg3mFLZMmk4ADGPH8xS5IBpcQRR8oOvl1GGYKdIBMux8hrp6+NBGOpNEDCYY5sEBl6GMgQb5ARlWaLJ/P5o9fdxqWtvN+hWr5yibwAjD0hmvfqzfJQBwgQSwuCx5H1/tGTQ6+7T4tpj3ogwpeIbYYLVbayK2tCIYCNtEZm0cnDeFYIN8ul/eMjKgywE7mvGucU01TZShNJjb/2ujC1hF6SzPgBdUk9DcN/vu09N6OxECnkYgAnKKuObth7In/KHFB5IizIsrriXCnaykIrXPSHQDgiIYLeDlaRjyyNQbzCl8mXS+AHID2aQVTTAct9yo3LKEGyUg4ETq4j+hTirUyN9U/EgY9gj6uWC2GBbAOJtKEOwkR4kAXL9BMPKgmyQPbsKauNT5yk8kQ74+IAVRLvtw8dDR+SDDX39mRYkCNsAuHLN+43IJvBDPkvI/DYLyuIxVVYjwpSKx8QB+nmCBpIInTy5RvllCDbSwQexn7hoUoMy8A1l+GXZgFVvTOS8TGAIcoc4SwSbTbC7Y0vWDT7kt1OkVrrRFrgPm3lTEx3Gdbf9pPyhiGBPTb2pp45CoF0QEMFuF0tJz5ZGAKtaNtifnub9Mmkw8GOgY0jJQRyIAeThCNIIwkhi2qgcTyogj3lxbgPS4tEwyQYIiCWnjfRNEWwQIdQRq/Ep/W35OE/pi/tFOkM29IVuDCCd+F+Uh+lSxxTBRh2gV5ENUnL8PeSFjiTaHlufHteoG/KgbOLSqG5IBxv6spDPk3iUwfQ4Z7A2rxePOtFXIJ9bFijHHqm/vVfPPlY28iAt/lvdrKwy5/BB4AKSDjnQGQH3bDu0WNEGWcKOPykdoK8Nvi1TDsqiD6Ac+EHKLlZWmfOUnWw9imRAr1T7AdbI70OqnBQezMd6Ix99AHLxv579mV9HIdAuCIhgt4ulpKcQaEMEigh2O1WliGC3Ux2kqxAQAkJACPQsAiLYPYu3ShMCvQoBEexeZW5VVggIASEgBDoQEMGWKwgBITDNEBDBnmbQSrAQEAJCQAi0MAIi2C1sHKkmBNodARHsdreg9BcCQkAICIHuICCC3R3UlEcICIFSCIhgl4JJiYSAEBACQmAGQ0AEewYzqKojBFoJAXwqDF9JYMDn2lKfbGN8Kx4xSbChHetg9de5EBACQkAITHsERLCnPcYqQQgIASEgBISAEBACQqAXISCC3YuMraoKASEgBISAEBACQkAITHsERLCnPcYqQQgIASEgBISAEBACQqAXISCC3YuMraoKASEgBISAEBACQkAITHsERLCnPcYqQQgIASEgBISAEBACQqAXISCC3YuMraoKASEgBISAEBACQkAITHsERLCnPcYqQQgIASEgBISAEBACQqAXISCC3YuMraoKgVZD4IYbbgrvvPNO2GqrLcJccw1qNfVK63P11b8PDz30SNhxx+3DqquuXM334IMPZ/dXWWWl3H0kuOyyX4fRo58Kw4d/Nyy//LLVPNPr5MMPP8yK7tPnC6Fv31l7VI0zzzwnjBkzNhx00I/CkCGDS5WN9PCfFLalBCjRDIHAhx9+lPmOrQx8qF+/vvbWDHuOdoA+9PXXx4YFFxwchg4d2mvq3upGFcFudQu1mH5ozBgMEbbeeouwzjprtZiGUqddEMDAePTRx2XqbrzxRmHjjb/RLqp30vOkk06Ng9y7nQjiBRdcHJ5//sWw2GKLhn333SuXj3mOOOLQlphcjBx5eKbfPvv8ICy++GI5Xaf1Bcs+44yTSxfFSU3fvn3DiSceUzpfdxM++eRT4fLLf51l33ffPaNNh3ZXVI/ne/75F8IFF1ySLBdkFH15O9UHFcHk9Z57/tGJXLOSqBf6lFaYvFInf2T/4O/jGvaA7qussnIhYWYfYvNjYoEFCzvRt/E67zkERLB7DusZoiTbIYBco2NWEAJFCGBChsF90KBByYEOK5Bvv/1O2GmnbxcOIkWyW+U+Jwopooe633bb7dlAbwlMvTzTo15YvT766OOzok866dgw66zNXcGGHzz00MMZWfAr1CR/qUlIPSwg84YbbsxkTmsyAXudddY5ma9Cp3ZbXIAP3nbb32M7nDM3mcPkj6FdJg2wOyY66DcQ4E9oW1yxxn34FCa8rT5GcWIJ37cBq9EfffRRdgv12nHHb3fqP+GTZ555dkak2bdgEohJB0KrTNwzZXrpHxHsXmr47lQbKwbXXHNd9kgWj8O7OiB2p0zlaW8EOCFrl8G7O2h3hyByNXS55ZYNu+/+3e4U29Q8zzzzbLj44l9lMruyilxWCRKJlGwQAky0WpkMkaCizwMpbWVdUzbhdiQ/MQBJQxsFaW2H/hzt5pprfh+gN/TdeustC7cUYbzCtjOSzxQu0/Me+w1Meo488rBOqsAmt956e7aNDJG+DwUGnFTYzPTVdvNRW4cZ5VwEe0ax5DSuBxrzz352ataxYZ8ktomgcZ9wwrR/NDuNqybx0xCBesRqGhbbo6I5oHVlm0t38kzLSv3973eGv/zltrDaaquEHXbYrqlFkUhgpRF9hw8kf62yF93rhxXRn/3stLgvvW82GcJWi3Ygo7YeRx11XLYimtrjzsleq/fnIJyYDGAswr57PPVq58CJZaNJNrdCYbKAVelGge2t3Xy0Ub3aMV4Eux2tNh10xgoTOgR2bNz7BYKdmkVDRTT0EPpkgxGu0THipS4MWHgZAx1LKth0iG/mixvopCEfg73XG/risSIC4vG/UYA85MMqiU2POqKuKAsB8ejwuhMga+zYsdlLLNB5yJAhnR71dkduKg/qgjr169cvviyzaO5xsk2PNBiYEaCTrz/uQxbICAYGPOJEqOjfGNcscfyDcvACD21mMWYaHBFf0Rv4VORbG2AvI+/bfDi36XDdVVt1hyB2d2WfvoAj/CDlU4h74YUXs7aK+pQJl1/+m2jP0WG77bYJa6yxei7L1OJDIgFcuc/e+gH7Ej7Shi1feOGFhjbPKekuplZnK4624uovJo3Qf2oXF8r6ttUF58AHfQsC+lDoUi8AC0wQEFJPENhO0T5SE6B6shGHeuA/AmQUtbMsgfnT1XrQDo0IqSmiy6fsR8aMGZO1L9SlEb4sBBigT6zX1zAtjiTOjSbm0AlbQTA2lZmEcsKkFWyL9vQ5F8GePri3VansoLGCg0dZ6HDY2fnHVrZidvUSjZ6P9pgm1aHjsd6f/3xTNogwHY52Nm7l2jQ458qgTc80rId/JAfdUCbibUBHiQ7NBtYbAxF0BXlAsOVxMmLz4bxRR+rTY+CDLA5ePp6THd5vRPSoF4kC8+FYhAH2tuLLGAzo7GFHpPeBvkBC5eNxbQdHT6xs+iJ9QNKgj//iCEkC5bOuVmaq3ql0yJNKa2XZc64O+skmfTFl96I88Clsw0Jb22+/vapkBb6J+6inDcABPmpJDWV3pQ4nnHBKeO+998KBB+4fFlpowWoRJAHVGx0n/mspPh7X9MdUHDGBP+FFV+5fh+8AN9xnSJVFH0uRiKnRmWXyCD/Efl/bZ9TzW+ard4QNYUvf3xT5NnVAe1933bUzfZiXba5eecxv+yibnm0ghaVN58/rtVH4pCemtBlsv/zyy1VXoyGXkytfBq/Zvu0YxLhmHOFvf/vb7eHuuyv9OWWiDpgUAhsf6Ado93gXANsmEdgH+fT+ml/OKWPDsjZCPTBGYcwoI9frpOvmIiCC3Vw8Z0hpJJV2wK5HHgACBgCsmqBTx1vQ3LsNwobGjw4DwQ6ezIP7GEzQCSNgZQ0rqigfgR2TX41B58JtLKnBhAOvLdMSGugGUg09oB9eMrF1Rtkk95CBOqEzRScMooOOuEgeXvBCPAbRMoFykJZYoBzohoEKGHrd2OEXDVa0o+94iQsGL2IA+bfG/X8eA3b0IBwY7EHskPbuu++Nq18HZFhg4MUKEOqAVRfoT0KM+uM/bU1iZTFJ1R3x99xzbySYL2Zl+pU264+QjQGZ+mHQ5IqfJcEc8KHDuuuulemFusDfytqK9bAEjHUpwpt5/ASTdsB9EBRiBp0gC/4Nf4ONLB5IZx8dd5Vgw0bwnZlmmimccsoJVL86WQU+m2zyjaqtgc/WWxfvfaWAiv3fyUgL/AjEigF9AvQmcUK9cI08tAXi8GIegvdpYmXbMtLRD7qrM2TYwDZl20yRXW2+ovOp8W2QPPjxlClTMh9AO/PtIFUuMUkRaNsGuHiSkuHvpeqBF2XZN3nfRn72HfADpMOLz+gL4E+pPci2TOZFX9LsrSGWlNp+Df6HvgO+i/aIscEGjgXoh6EfdENYcMEhSUJu86JMfkHJj2M2Hc9pw9S4xjSQickg9J4WOLEcHcsjIIJdHqtemZKrH55AsGMuasjMh0Fz8uQPc6txAJIDJFeycI+dSJFMpEHgAOc7JuaHrijXfhYNnQ/INwZedubUEffsaiHKYJzt0EiMEI8ydt/9e7mVQ9ynbn7gR1zZwLKhF16A86ScZdhBnx028hR9sowDgsWNZXlSB105iFoMSN4wsCNPvcCJkCdHyMNy/WoPCVdR3Ul4PL70J+oE29sVNNbdYkb9UoNnvXrZuKJ6IA3LtKQe94kr/Ry241MB3Ns6klfqTr/F0deZNodMa1PcxwTH+w3SpcITTzwZrrzyt2HRRRcJI0bsXU1CrKcGH+pY5Jdss7SbnVhAEdrI2s3e937YDJ0JAHWz/o843rd9F/PUO2KihPogeFviHtu1j+OTAPRp2LbVVYJJucgPUsuArVfo01A/bOFCfJlg26jvN2Hvou0MVo+u1qOZdvV1JL7oi/zXjNhWgY2dxNKWaKeVvqrzWODLsdfEEH7flUmS90XKhD78sgr7FcbpOP0QEMGefti3fMnoLPlpKj/AsYMoavAchFBJPwjiHuPtIJW6h7Q+sKO2OkFXEGh8ZP+jjz7MVlVBbBgo2w5eJItWDtOzY7UEkHVGmiLSkdKNMsseqVdRGSRullSR6BXZgwOCjSdmOKZslMKAZXvS6OsGmfbRv4+nPaz9kYaEyq/OMz/z+dU45sPEhyvpzGPlWltz0Lb3bJ4y51xZ8/oS79QAavPAvzAwIr3HAuUzra8v4ogxzq0v4Lor4ZZb/hruuGNUtuK/1VbDqllp69QEqZqowQnbjG1HNgvbC0gKfNCTPMZ7G1E3X2+2HZ/ellnmHMQTfR8w9vVvtLhQJJ91SdkZeSjXY0Xftm23qIzUfWKVisOq7FZbde1HnthuvM9TPtuorydt0516sA7eFiyzu0f6Z1G/Abks2/aR7BsR3x1fI0ZlyXC99NQFbQg24RMu6KYwfREQwZ6++Ld06WzUqQ7Rrub6QQ6V4mBS1IFw1cCSSHYUWBXACiRXtTxI1Mt2bCQi6ICxGoitBNQLg6RfvWZZfjBDWeh0QXqQz+rHclN4UEe7kupXUZmm3pF6FZVRRNyomx/UWBYHb0vUWFbKRiDswNFjQLuiE8dkpShw4ErhizyUY23IumGw41MGL5/1sPhARz5utfJsXg6S9AnETa2tICNVD9yvhy3zYDCE3eIT/8KBkaTET2jQ/vDeAOxUhDH0KBMuuujS8Oyzz4fvfGensOKKK1SzsI3C1iBhXFWvJihx0sgvWb8iskYyZ8kNfcv6AFWhziCOWJXtjs6QRTmp9lSvfOrhj/RtkKCirRgpuWV825dlr2259nOQkMstV43625S8em2UNrf2seNFd0hyqv1avbp7TjsX+R/ksr3avoXjja1jV3RguXbBp15+pvd6ciKIPsQ/TagnT3E9g4AIds/g3Hal2A4RA6x9tMjKoCNFSHWYHDhTccjDgdPHk/QgDfbBYoDzgR04OzzqSqLIzo9kiultZ8Zy8Bidj9IxwPGxaaX8/K8LsqO1crxuGLiQDgMbBq7UDwT4PPaaHWlRGSRuligjP3UjJlYmzllfO2FgWSAj+CIFQiMMUK/zz78425eICRAeqaYmQsQ8RVBQDv3DEkeSZ183pGegXNoa90lMUqvF9eJhKz7OxsopMKcvsLxGx6KBnz7oB0TIYx6QLezvLLIZ6wXdsGcZofKTyGMy/8I16tydiRzyMtAWP/nJyDDPPHPzdlYGbd1dfOhjqTqy3QKH1LYmkksfTx9I+RZkYsUXuHZXZ+IOILAfvG/fflVMcAIboB0isI/JLur8oW9bv/XJ2bYtaaMu9Qitl2OvKbNoEsZ+AX4PGzUKxL5MG7X24RO2ojbaqFy2mbJ4N5LHePq+H4cYjyPLtpM89re2P7V5Gp2zXCuzKA/aAd8t8unZzxSNF0Uydb9nEBDB7hmc264UdiBlFPedDFdNijrTRgMrO3GUnSIQjOegzUGCnSTjuUXEr15DblH9UB4GGwwgGKBtYKdoSaGN5zk6ROjEF+vsQMM0RUeWwbr4dCQsvkNlviLdUhOaIgwwwAMDvojmdYD9+JQAcV4X3KOetBHuMdD+njTQjil5zMs0FlPa295jehzrkRtvq3plW5k4r+fnxNbXn4SJGOMlviK9WS9fLnCDfZaPLwH7F6982kbX48a9EUni/4WBAweGY489slNy4HP55VdmT4QQ2RV8kL6eXzYifxYrTCIY6Fu+32H81OrMtkJ59Y6e8BSlpS2LbI18qTS8V4/QFpWJ+yRgReUSY6QtQ17p10XYQw7bqJ1cUg97D2nLhhTJLZu3KB38pN42NuRjGpxbfKiPvYc0ZQL7P6Qtk58+4PtL5IevQsfUBBXxCtMXARHs6Yt/S5bO1QasHGEFpyigc8ZWDN95c+AsWq2hfLtS48sAeUFHjaOXz0EB90EC8bUSWxY7JJAbpAWJ8cSAAwU6fJAVEOt6gZ1iqpMryod6og71Vil93nodt+3s7cDeSDfaw+veVQy8rsQZK/VHHFH5fCPTlCFW1mbIR308KaVMHEl+bP0bEa7UgG9l4pwY4bxocoM4G0jcU+SnyI42D57QcAXd1odlEF/ghKdI8NNmB9Z7mWWWDt///m6F4qk3EpTFh35ZNNFuRLpYf9/+6QON9OiOziwTOtebvAC3st8lBmb0bd8PIY6Be62t/zfybeYtOqZk2rTsS3GvDNljPayOVh7O2fatT5fJ5+XYa+b3vmDTdPWcda83DrF92DRl8tXTpcz4x/wY/1B39P0p34G/4uky+geF1kNABLv1bDJdNUJD5ouNqQZtleMA5h8/dnfgtLJxXtQRsYNDZ4tBHOnsvkZ2ihgEQK4Q/J5edtj1BoosY8cfyvSk0KZJnROLsgNDETGDbMryj8yJhx0EqIu1p7dTVzGgTHtMDeAkVp7QMx/r4VezSCaK/I7+4OVyQC9avaeOdsCnLvZIvcraisTd61vPHj4P65QioWxfXfU5W6dG59dff0O477774yQ2vR3L5qfu3m42jT1n3Yr0b+R/jLerpZxk+jZgy7XnXdEZfssXGxv1C131lUbp6TNd9W1b19Q5+5OitsEJhe8bUrJwL2UTm5b9pPdn6lGGxFt5PKfc1GSeabp6ZD8FmcAnFdh32DbOdpmaWKdk+HvEvFE/Y8l1Wfv4snQ9fREQwZ6++Ldc6Wz8voNMKcpBwadlp1REaBp10iyLA7Qnjuh4UAYGbnzc33dU1IvxtnOkbA54RYM/BnJ0vAxMX5Zc+HxeR8b7I1fnLKlAGtYJ5x4Pdvj+PuoArPF9Wqy2eR1Yp7IYoGwfaGtLSKir14d5aX+bB3G1enTeD2oHG4sNB0lPTFgWjmUH966QMchtVI/UAEz72tVXTiy8fejn9UgF6u+3MkG3suHss88Lr776Wly9Hh6WWWaputmoZ6o9pTKyL/H1YtpGduHEyWLVyLcom8eu6My0ZcgMfbVMWuhCgoinEPB7G+wkuKu+beX4c2JV1DYQz5e5fVv0snjNemN1H7raYNuolUc/9uOEzVvmnH0N5NR77wBtAqFMu6CPWX2pC/tHrzf7ibLtgPJ4ZL9hbc04HKE/xjW0H4Si+sJv0LeXqWcmSH96HAER7B6HvHULJFmBhqkOx2tu09uVibIDp11VQee5+OJD4xcRKj8ugxeJbrjhxsJHYywDK1l29Ro6cmABMWHnlNIdZWL7Bh6vYasJBiIQ0dGjR4d//zv/4wfsFItwARYXXnhJtnfbvjDITtKSBK+LvSYpgd5bd3wPGRMN3Mdgjn3dnrBwAIMcvHSIdMAAeSAHHTHyed2t/bAVCNhbDJ577oXs02mQC4KATh9Y8YVXluFXEy3+/GYv0nCLA7HEqim358BW0BMElDb55jc3ys5RP/qCnwxwElZEdKiLJfuQh0ESdZkaWxXVgyTE2pB2SO35hB3oi35SyjKQHz/2Mnjw4OwlO9QB5WCQB4bAji9CeTvDfqnwv/99Hg47rLLv+thjj4r7sAdkySp2/XuhrX17S8nGPe/L+HwmfAd1oV1wjjr7QN/05JD5QCr4lRDKw1awIv9spDPlwk+hTyPSwvRF+vv6pHwbafCT9t31bV+Gv6Yfoi58SRZp/Iuyvk15Ofba1gPtF3nRXoEH7I14P7Hk5MLft3LLnMPn+dIt+ovll6+8g8CXUPHtd+iBPsGOR/Vk00chz/adwA5y4A/+6xwk+mX7dF8+xy7rq0gD3UGYUU8GYAacoZ8PnBwUEXWfXtc9j4AIds9j3rIlcjAvIispxdlZkBhw4LGExuZLDZzolPmZNZsW50WdMsv1ZBN5WAbO65ENdPxYpcAA4YMfdFheUcdNoufloIPGqjc60zIBWMAOtpNFPhAplAGinOpQaTtbBrFjXGpAKIsBByIrH+cgQP4Hd+wgzPR2tYcDP+PoO7i2gyjjcQSOIJiokw1cZSp6skC9iQXycsC3cnCOMjDI1tt7a/NQNu8RX0uYEcey67UNYuIJG7C0LxmyLByBPb/5Ddww8CNYrLMbBX9efvmV8ItfXJDFWr+mLj4b8En98JFPx2vbDnmPGLEMYsN4HtmefF/kfYvxlMf8PJbVmU8WUv0JZdmjrZvFzqbx5131bfpXkW97+f6aK/L+Pq/RR3OSy3tljl2tx9Su+FqdgDvaPF8gt3E8p0/wutGR+vl0wAeTODvZgv+lJsk+b9G1badFadCuMXlA27Bl2/TUA/e66x9Wns6nDQIi2NMG17aUCgKAgFW91Iw5VSl0GJh1Mw86QKyQ4KfNQRZ8QMeAlQYfj3woH3kRsNKFlbmiDqaRroznqqnXg9fQBz/Vy3KhFzpoXy7keZ0pg0dggXTAAwGYoPyyWFIOjiCB1Ilf8yAJIEmx6X0edNDEvxEWZTFA/YAVA+pXREYhEz8PT9/w6Sgr9YKO1wdpUJ8Ujt7/qBuP9EfIsDZtlq2ALf77eqBckADUn/Yr0oW60k5sS7yPIwgn2g0D/Mr7NtLAZzAwlwnPPvtc+PTTz7Kkyy67dC5LCh9vw1yGggvUGX6AYHVuhEW9eOtbxBbyUzqXaX+QR2xT2EN2KtSzVyo97jXTt4vK4H22DV7bo/cdG1fmvCfrkdIH/sH2xXi0QdTLtnPGNToCK9u32f7T5qWvNBoLbB57zvz2nj3vqv9B77Lt3Zaj855BQAS7Z3BWKUJgqhDAgIKvpfhH5lMlVJmFgBAQAkJACAiBaYKACPY0gVVChUBzESh6ZN7cUiRNCAgBISAEhIAQaAYCItjNQFEyhMA0RoD7McvuEZ3G6ki8EBACQkAICAEhUAcBEew64ChKCLQKAnxRsd5Lm62iq/QQAkJACAgBIdDbERDB7u0eoPq3BQKNvmLSFpWQkkJACAgBISAEegkCIti9xNCqphAQAkJACAgBISAEhEDPICCC3TM4qxQhIASEgBAQAkJACAiBXoKACHYvMbSqKQSEgBAQAkJACAgBIdAzCIhg9wzOKkUICAEhIASEgBAQAkKglyAggt1LDK1qCgEhIASEgBAQAkJACPQMAiLYPYOzShECQkAICAEhIASEgBDoJQiIYPcSQ6uaQkAICAEhIASEgBAQAj2DgAh2z+CsUoSAEBACQkAICAEhIAR6CQIi2L3E0KqmEBACQkAICAEhIASEQM8gIILdMzirFCEgBISAEBACQkAICIFegoAIdi8xtKopBISAEBACQkAICAEh0DMIiGD3DM4qRQgIASEgBISAEBACQqCXINDWBPuFyfP2EjOpmkJACDQTgaH9JzRTnGQJASEgBISAEMgh0NYE+7ZXh4QJH3yeq5AuhIAQEAL1EFh03i+ENecdUy+J4oSAEBACQkAITBUCIthTBZ8yCwEh0G4IiGC3m8WkrxAQAkKg/RAQwW4/m0ljISAEpgIBEeypAE9ZhYAQEAJCoBQCItilYFIiISAEZhQERLBnFEuqHkJACAiB1kVABLt1bSPNhIAQmAYIiGBPA1AlUggIASEgBHIIiGDn4NCFEBACMzoCItgzuoVVPyEgBITA9EdABHv620AaCAEh0IMIiGD3INgqSggIASHQSxEQwe6lhle1hUBvRUAEu7daXvUWAkJACPQcAiLYPYe1ShICQqAFEBDBbgEjSAUhIASEwAyOgAj2DG5gVU8ICIE8AiLYeTx0JQSEgBAQAs1HQAS7+ZhOF4kfvD22U7mzzTW4072evOF1GjDnfOELX/hiT6qgsoRAJwREsDtBohtCQAgIASHQZAREsJsM6PQSN/bZh8J1J29fLX659b8TNvzeSdXrMicfT34/3HzuXuH1/94fZu0/exi2/8VhwaW/XiZrMs3Vxw4LE14ZXY0bcenzIthVNKbdyd1XHR8e/9uvMhuuvdNPw7Jr1/xiWpT61L3XhXuvPj7Af7628ffDOjsdPS2KaZpMEeymQTnDC7r88svDLbf8P3vnH2tXdd35bRqgGNlJi810sM3PUfEPJZgITAIOoBaT1m5DRGyC84+NaawROIKq2JE6SmzDjFKM21LxYzRUYDxqaoMdJJgxo2BS8cMQfikmoRiTafgRY6IBoya4gZAmYc7nPK/j713vnHPPfe++H/e9taT3zjl7r7322t+937vfs+46+zyYjj322LRs2bJ04YUXdmXMjzzySNq8eXP6+c9/nhYtWpTb7orhETbCeDZu3JhefPHFNHXq1LR69ep08sknj7BX0X0gMDIIBMHuEPf7bvxi2v/y0/1aLV33YJpy4ux+5VawZe3CdGDfHrssjl+567XifDAn3SDYEKXv3HVd4QbjYVwDlbFCsJ++/+b0TPZjMu+Sa9M52c9oFG6O7ttweYtr3VpjLUbl4pYVJ8tVSpeu2TqoG7MWY0NwMdYI9rp169KePf3/t3jo1q5dm+bMmeOL47oCAUji+vXrW2rvvffeluuBXlx22WUtTcfK3Gzbti3xYzJ79uzE+gwJBMYjAkGwO5z1KoJ9xoIV6fylXy+1dvDAG+nuNfNL67pFfpoS7Fe+91Da/8On03HTZ/WLbFJHBNtkyoyMYK8Pgt1LBPvAj/ekLesW2hSmo4+ZlFbe9kJxPZATbO7/4VPpR9n6gDx7uePqj6cP3j9YFLe72SwUR+gkCPbwAQ9Jfe6559KHH36Yli9f3nHHb7/9dn7z8Mwzz6QlS5YMazT0tddeS2vWrCl8njhxYiKi3Q0Bi/fee68wtWHDhmEdW9FxhydEqLmZe/bZZxPk2Uf0d+zYkUfmzezZZ5+dR7HtOo6BwHhCIAh2h7NdRbAnTZmRlm94vNTa8zvvTI9vuaG0brgItideVVFYyOT+vd/N0wvOueTPa6PypQOSwohgCxjDeGpzSJfnL107qDncccuX0yu7dxbel61X1tZjW9ZnKSIHEzeaQ52SUjgzwJMg2AMErsNmSiIHEsn0ZG0korz4AJlEGE+30h0g75B1SxHxRLVDqIdF3d9wcMPDjxfGxY3V8ccfn2NGqkhIIDAeEQiC3eGsK8EmOtgkcqfpIZOPm57efeeNotcywlJUdnDSLoLtUweqCHYHXbZVDYLdFqJRr6DrHWe7tV5HcuBjnWCPBBEtm09NgxgIwfbpBqNlXGVjHQ9lPmWmimCPByxijIFAEwSCYDdBSXSUcEw7/ZyWfOyyNBFND4GQk9esOdxKWNB99539RW9TZszKI8lWwENkB/a9ZJdp8nHT0qQp0/PrKoJtbQ7sezE99o/XF21nzV+SZp3X9/Cb9WO6pmT+2rUeTRe7nE87/dMt/qDblGBD/g+8saewU9ev+UDU9O0Mi4Pv7MuLpsyYk6ZmeBkepscR/8AcX5FJx83Idety5nPFQ78GmyLi++cB0mm//6lGkWVLz8AGfp965oLcK10HHi/wTBMm5Hq+zsbFWnsjw0TxYz0ZJvTLzSORac5NLv3qPf3smq7p8HdRJejavDGeydn6rdOvsjOY8iDYfegRPX399dcTkUmE6CzRxnYRR/RJE6A9Ql63toOITcjWn+beYpsIMKK6eYH7RVoIPzwIyI+JRpAh7Ijpmo7ZttSSk046KR+X1XPEP8aN//hO6gf+lYmNhTqvp3XUm0+Ugw8R3LL+0QVDTRGxtmV12OAhS8ZENN381ja086L6+HLWWWfldtRvPyZvg2tdJ5oiQ9TdIu/mo+maHZsPu646ggdzgr7ZMl2zyREc2o2bdmbvrbfeyueWNrStEusDbBB0WRNN+qqyGeWBQBDsDteAJ9iQJfsKnfOVt/6gxaKSs5kZoT14YF8lwVZdjPiHxeqi0FUE27dpce7QhfXjdSE+RqisHUTv8a03pJd2HX6Qxeo4mi3O6wg2BO+xbOcJ8r7L5NRPXpzOz3bA8IQZgrYzexBTSZ+21xuWdr42jeL7eWnarl3/7NDCGI3U6jho+3A2To8PqUhzFyxvSTny86QPHfo6cAe/nIRrh4fObf50nZeo5aTY1obX1TmwtjxA+8wDf5evfyuzo/fRyofqON4JNsQLomSpDx5nSBM7ZnhCAmm56aabcqLn23BtDwBq5LpMr13k00euy2xYX14X25BGduhAIEgQ/XZjhmhfddVV/W4udCxmy/zROso2bdqUPxQJTiplu4Tgkz6YauOhna8jcs+NxqOPPqpmcwJInZ8nlJhfdj9RQQ99zSv3Y1J9O4d0+oc9rc6O9u2C19W59nX0zU3T7bffnhNis4WftmML47bdVqwe4stccfQC9t6e6TAPixcv7odXGVbWRv23sjgGAk0RCILdFKlDekomIAYzz1vSsvMGW9tBDk02r55fpIRQRz52VQTbEzkjO2bLE2AlesNBsCF9W7Kt97hJqBL1uY5g+7GU2YN46i4mkEMe4MOPKlFyp3NVpq/4ldVbmZ+XJu2aYGX2FTMr07QiK6s6eoJaRbCb+GS+tMNO+/S6Ogf4vPPO69LeJ7ZXud9C1iuVulgx3gm2Jzpl0EI4IU0mEFSIGZG+KjGS6Imn129HWjxp9u25tr68LqRNiasRyCZjJnp62223tXSnYzFbpqB1lEH4PLk2XQihRXsp8yTaxlNWV2f3ggsuSFdffbV1kx+5Caq6eYK86hz6MbUYOnTRBLuBEGzwJoqv/lj/+Llw4cKWHUmsjmPZXDFmxl4n/qHLOnKNnXZrta6vqAsEgmB3uAaUTEAyFn3l79Mdqz5RWCFKveDKjfk1UVbb0YGv6tnNQdujpGTEEzkjO2bck1IlelUEGx/sATTdJpBc8ElTpuWm7UE4b19JFIred8pIWZhy4pz0bkZ+X/3et3M8bO/sJgQbvKad/qn00UOpLtyA2DcC2P/DFRuLh+Y8PqTknHZm380M5PFHux8qxR477Iiiu7ygOzkb/9wFV1JdK75fxb2qoceK+Z+ZpeXwLcfeLJqrefiULduwq0gHKnsoFpxIpyCirfNI/36eqgi2t4vN2dkPYvhxzfyxl/aBbA9z1o8+Z0BfCHNuePqx6pr2Wz/mbbO5oI/8259sPEdPnNTvm5K8kyH6NdYJNoQMgqLC1+5ECxEjTJAzS++gnIinEjMjTdRBPDWCSoSRfhDIN5FGSCNiRyW6mooA0VSymTeSX9jiB7v8mGjqgPXhCbbp2tEIZNWY6cePi0iniZJos1VWZ2WQQrDn4UifAmI+o8u54lNHsNGnb+aK3VRIpVAhcm7zzXiI4KpAKpkrMNA+0fFj0nZ2zk0DRNTSKKwckssPwtqyPjTarQTV5sDac2RdsBawrfNgOtQzH/iga5N63XmF9qtWrSrIOn7hE0fmQm3rzY7OL+uLa/pE6I/2uh7yivgVCDREIAh2Q6BMTcmEERuN0GmaiL3wg7ZGvLU95UpGPJHrBsGmD8ST5zKS6HVsfLT3W/hBGEkR0PQGSBoCBkgdwSYajfgUEMqUIGpeu8eubjs4Pxa1Qx+diJ+XMuzUnt5YUQ5WS9f/n5ax+rF8ZunXCrKv33rQXuu49m11nqhX/LTOj8N/20JbL74vXa+mW6fjt/Arw461ULYOzH63j2OdYJfhpUTKSKuRI9OHpFxxxRV22RK984QQwu3bFw0PnSh50f69XtW1J89K+K2N16FcSR1jxc+qMaMPETMy7CPCdWPQOuxw02FkDEKoqRgQYIiwicezjmD7qCsRaxsP9hQXH71Wn9D19Z3MiyfIijO2kTodX4e+kuSyaLKuM3+Tp0S53Y4zipnh6f2xcvwKCQS6gUAQ7A5RVDJh5MWTTyMuSpSsTNvTtRIWT4BGE8HWmwX81sgy12VSR7BNH1JuDyCyzRtvfsyjpofIumGMvvcBIp8/rHnuF1qIPrqe5FJG6s6sLKXHHhSkrIn4eSkjiWrH65eRe79m8IlvQyCbumc63zQsu2mXmu93s6MYoVhFsH0EG/xO/eRn06wMP/vWoaWj7KJuvZpulY6fg8Huq279DfY43gm24QehJqIJGYRgcoR0mCjZ9OQGwkjkER0il2WiBLQTIme2PHlWIlml064fxkwEmHHamBk35YhvXzcGraOtkmSuldT5+k4Ith93HS7qE5FY5s2i2/jAWJX4+/GiUyWekA6WYPu+vX1f71NAtH/Fk/HyBkkVyDtjR7jhAhd/Q0kd3xKwru2BUMpCAoGBIhAEu0PklEwosdFIHdHqM7P0BZ8eQlfanuteIdje77roMeNC6gg2xLruYck+C63pD3kO9to/bklZKPSylINzPndtC1H0PpsuxPKcz1+bzrhohRXVHj1h7pRg282VduLJp60lH3m3cm3bTqeKYIP55tXnleLHw5PnZ6841+cH6NNjqOvVfKrS8TcRZTcaZmM4j2OdYEN62TVCBVIBcUCIfkI4/FfueaX8UoIDOYHEWKRX1HJSoikjVqdkT21ZfbtjHZG0tl5HSZfpcIRM8bAcKRR14v3lMw82AAA7oElEQVSsG4PWYdMTbCV9vr6TuqYE2xNGPxYbt/pdpWO6evQEuAzrOh1f5/v29aTb8O2Cia/X/vVbCNOvO9pclUXNaQdJp3/6CAkEBopAEOwOkVMyoeTHp4lAJuz12kostD1dK2HxRG40RbC938uznOF2X+vXEWxvDyzJ64X8ao6yYgxeEFNyyvVBUcpNFDMIJZh+f+ddVt1y1HlpqXAXfl46Jdjqk5muIsm+3FKLrB1Hr+MxqiLYtOUm5akMk6qHDv3NgJ8nXa/YQ6p0OsWtz9rQ/x7rBNsTMo8o0T3N44XoEIWGVEBYTTwBgmRDSHweL/q0pV+NZg+UyFn/njyXjcvrKOkyOxw9odUxQ7ot5cKPuW4MWkcfRto4R3yfWt9JnR+3H7PVewLqx4JPTUk4ul68/TKs63R8nffP13v7dfV+Lrzv/lrnAjx9zrzp+4d9rTyOgUATBIJgN0FJdJRMKLHxpIeIoO22odFebY9ZJSyekPg0DP8VvxK9qocczXXvn7at0tHxeb89ETMbeqwi2N4XP079NkB9UNsQxd0ZcX41e0hOHxi0VAvVhWjzmu+9T2zrR8wVf22j535eyrDrVN9Hd41Ie2z8Tir049t6jOoItvkJfj/a/e30/Yc2teDn+/PzXoZXlU6TsZg/w3kczwTbf81u5MzwV6LiCZDpQLQhpdgyYkqdppRw3cQWelVSRSRV3+t4UoauJ2Y+L1lTOfyY68agdfSjpI3rTki0tvXt/Bz5MWu9+mSpEPhi4rHw4zW9sqNv2wRr1fHtfd++XtviT129ziG6YFInEGcVbjxYz/ajdZonruVxHgi0QyAIdjuEXL2SCU9sNOfamvkcWm2PjhIWv+OCkS6z5dsq0euUYHvf6cMTItXx5J4biKXrdhQPNJqPkFmi0EgVwfa22Dvc2lSlTmBPbXONULZ17cKCJKrPfRqtv30et978tGoevuqUYPsxlGHl51JvMpQg44VGwBnvfRsuzyP55qEfs7b3ddZGj35LQF2T3k+dK7Phdaw9vuoOO+iX3ZiVzavZHorjeCbYnpwpsfMERgkQBIQotUq7aGg7sqe2ys69r/pQm+l7HU/K0PMPwOmuG9ws1OUk6xgUD+xqHdeKJdeeKGt9J3VKoLHrx6z1ZURTyaTv148J+1Xi14e/oaKd19H58HW+b1+vbdvZ9s8IKCZV4ylb0+j69dLEVlUfUT6+EQiC3eH8K5nw5MWTN0z7NARtT72REc49MaOMh/jYTu6N7O18EGCVTgg27ZR4cT334iszYjspnTr34vwhwTqCDQny+buQ4rkXH85j3rPrW2nBipuKPOgqgu0jsPZWSfoAQ4v846NiDHaTp56Yv4HS3j7pbVkEm7E8/cDN+UON07P0E9JZwFdfUmNbJ9JPnXiCXadr8+nnmcjwaYf2R/dz6W/CNN2IvsAZjJgrIvGMQ0UxolznWesYx8HsTaGnZlsbUo5d8OOFNmCPqD7Xfhw8DDk9+zkq2xnFtjj0OoYB7f1YKGMsU0+clff57oH96d23fzzqtunb97Mj0oGfH5HOPOFXuDyqxZOmOkLgyQN52fxANtjejaOJEiB2n4Bgo0s5es8991zLvtGecPlUFNsujjQSztsJUXLdco7+yYslD5woNOLJpidl6PioPX2z2wdjIOVFo/A6Ztoqia6rQ1cJNNd+XrS+kzo/n37MWs94/AtmGCsPPJLaA4lV8WPSOn/ub0aoN9s2t3Uk2df5vn29n8u6el9na8VuLphr1gFHewCSNqwv5hhfbLcZMETXBPJOHW2vv/56K86/sWE9hgQCVQgEwa5CpqJcyYQnI2UE2UdItT1dKBnh2tdTZgJ51D2iOyXYVbYtQlpHsPHBk1nzS49mi7Iqgl33sCKk9+iJHy2NSFf5r/0b3n4sqmPn/ubHyv1xIASbtYC/uoe0t8t12XaHdfjQht04dC9su6mgDqkj2PZcQJ9m/98+wuy/VbEWuvb9vOiahrjfd2MWcd/XelNgdjiqLS0fqvN2EewnX/9IuvuZo9IxR6V03YW/SDM++puhcqUrdj1ZU8LlO4BMEuUsE0iYPsSoBMj34dvTFhIOETEpI3vUeeJk+v6Ir5Ah9cl0jKx6sllmu84OfkPGjGTrmOmr1wg2JJD5LcOM8bDXs8+/Z26bio+QWztbc57o6nz4Oo+1r9e29NOu3kexzTc9ap/enurZOTcORshfeOGFdMMNN1hVuvnmm9MJJ5xQXMdJIOARCILtEWlzrWSijBhomoiPTGJa23OtZIRriNmOW1YWBJMyBDLIS1VIDzDplGBX2TZS7Elp2fjQeTh7M5/mPZs/HI3gcl5FsKmDuO3KHlZUAgrZZKu6p+//2yJXWn3w2GHHBKwvyl7wY9vN+bGYHkf6OSOL3p9zybVaXHk+EIKNMYjyzjv/ohiL74Cx2Ut+fF0VQYdMs9XgjltXFk10HVA4EIINJvOXri1e6lMYz07KcK+bF7+m7ZuJqgcr/Q2C9j0U53UE28g1/Z4x7ddpxdkfpGOOHAovumfTk18jO1U9+MgwehBN7FSlS/g+1DakGhKiDzhSD9mjnRI6yj1xoqxKynxFtxOCjT52IPxKPBkzfkPS7cFNJWC06zWCjc9EmrnZsZsGyhDINQS5ao77tOp/Q0qxrTjSwtacJ606177OY+3rtS19tKtHhzlmrr1/1CG6M4m316dx+LftIsINGPLAAw+kf/iHf8jPIU5/8zd/k5/Hr0CgCoEg2FXIVJRDfOzrdL5i1xet0ARSxVsNkbJ6bY+OEULOTbAPQXz7UNSPt+uR4kA57U14s5/t5EEO9ocffmhV+fGE3+//Nay3PSkjpqQu4Ku3X+a/dYB/b7x8OGUFOzOy1AHzB703f3j4azauf+8/fTIdccRvcZoLWPGgHftfk/4AcaRPxcj7QN0bL383b4MR2k2dMacUR+zvy/w8+E7ffKA/Vd4iyHUT0Tltp182n2U+Tz/90/3Wjrft54obLNabz2H3BJu5MWmHH3rTszdplvltNjgyBt5+iYC5+q/zRX2VLZ3vMjuUDYdUEWwl158+5dfpirM+GA53Bt0HhAoyawLRNVJgZf4I+eKtgBARCDKROtpAOky4VtJMnRFRdCCofAWvOtZWj9qONvPmzWuJdKtu2bn6Sj394S9C3VtvvZWf84vtCTWKXlRkJ2oHP0h3YYyKX9mYzUZdHTqWjmD6atfXd1LHeOnbxI/Z15seaQ70gxhmnlQq4cwVG/xirWEbPxBdP9RZn9TpfPg6j6ev17bYalePDoIe61T9YPwQesWxU10i1k8++WTeB6kxlqaUF8SvQKAEgSDYJaBEUSAwmhHwDyXqtwaj2e/R4lsZwd76g6PSP738kdzFXiLXowXT8KM3EPDfRBDBt5uV3hjByHl5zTXXpJ/85Ce5A3/5l3+Z5s6dO3LORM89gUAQ7J6YpnByPCFARPj5hzdlL865puUbASLAj229Ps+FNzw0VcPK4liPgCfYm547On331b5vVv7g9F+lyz/xy3oDURsIjGIEyEUmwkrU1oSo7vbt2/MdMqyMyDO6Ie0R+Ld/+7e0YsWKQvGee+5JEyZMKK7jJBAoQyAIdhkqURYIjCACPn+clAt229DdVXCv7AHJEXS7Z7pWgq3kevm8X6ZzTxr9u4b0DNDh6IggoHnjpK2QQmPpHOpQRK8Vjfpz0k0OHDhQKPEq9ZBAoB0CQbDbIRT1gcAwI+AJdln39kBoVb5zWZso60MAgn3mx95MkOvn3+iLXC+c/av08d8bGXL9sWNSOm7i6N6pJNZO7yCgBLvMa/LPedgxUkPK0ImyQKB7CATB7h6WYSkQ6AoCPNzIziUQbd3eDlLNQ46nZg+l2gOhXelwnBmZ9jsT0mP//K/pjZ8eMSpG/o1FvwiCPSpmYmw4wa4oPMDqd2/hIT9ItT3cOTZGG6MIBEYvAkGwR+/chGeBQCAwBAgQwX759XeKvOsh6KIjk0GwO4IrlAOBQCAQ6AkEgmD3xDSFk4FAINAtBCwHW/OvF8351YjlXx+RPSsVKSLdmt2wEwgEAoHA6EAgCPbomIfwIhAIBIYJASPYdKckO7bnG6YJiG4CgUAgEBgHCATBHgeTHEMMBAKBwwgowab04X85Mt27u+91jUGyD+MUZ4FAIBAIBAIDRyAI9sCxi5aBQCDQgwh4gs0QevUtjj0If7gcCAQCgcC4QCAI9riY5hjkcCDA7h97n9yev3J+1rl9r7cfjn6jj84QKCPYWFCSPf1jv0mrL/xFOqYvsN1ZB8Oo/f7776dvfvObLT3+2Z/9WXHNC0YefPDB/JrXYvtXRReKcTIqEWD/6kceeaTwjZfDsAvIYOSFF15ITz/9dGHiyCOPjNd+F2jESSDQPQSCYHcPy1FniTf/vZQRvjqZMmNO4m2AR0+cXKcWdQ0Q0FeYgyevMJ80ZXqDlqEynAhUEWx8gGTf8/xR6f3sZY69QLIh2MuWLSvgO/3009MNN9xQXLPfsb1kJN7cV8DSMycvvvhiWr9+feEvW+3xuvPBCARb1whvfdQ1NBjb0TYQCAQOIxAE+zAWY+6syQtLbNDnfP7aNO9z19plHAeAwC0rTm5pdemarSleBNMCyai4qCPYOLjvZ0ekjY/8du7rdVkUe8ZHR+9LYOoINm+fW7NmTQvmGzZsaHmFdktlXIw6BIJgj7opCYcCgcYIBMFuDFXvKXZCsBndGQtWpPOXfr33BjpKPN5xy5fTK7t35t5MPm56unz9g/HNwCiZG3WjHcFGF5KNjGZyjX91BJt6XodtLxw56aST0k033URxSI8gEAS7RyYq3AwEShAIgl0Cylgp8gR7yozZOYEmV/jtfXvS3l3b07vvvNEy3JW3/iBIYQsinV2AOfgSuY60m86wGy7tJgR7uHwZbD/tCDb2n3322bybeDX2YNEe/vZBsIcf8+gxEOgWAkGwu4XkKLTjCTa51pd+9Z7CU4jgHas+UVxzEmkNLXDExRhEYLwR7DE4heNmSEGwx81Ux0DHIAJBsMfgpNqQ2hFs9O678Ytp/8uHnyivIth7smg3O2S8e2B/bn7ylGnptE9enGZmu2VURWrp//mH70oa1Z170Yo0OXvw77Gt15ubuY3Z8xfn1wd+vKdfHf08vvWGrO99yaLwRePsBPsvPfmt9PaPX0y0J3o8ecqMdM7nril9yNB2+/jR9x7K9bmelOkzJsZjvlgfjP3V5x/K7L+UDmY+IPQx9cQ56TOXf83U0mNbrk8Hsm8GTMDSS1nf4Ie9U8/8bJp13hd8k9xHjxeYfD/D1sZA++nZT+TR94OvX8F4Itg8EDdhQvaqyEOydu1aO0133313kT5CIXXkbW/bti3t2bMnTZw4MR1//PH5A3Ann3xy0U5P2KXk0UcfzaPkb731Vl5FG3a6uOCCC1S1OCeijn364sf64QG+sp1O2EWDPkywO2fOnNxPHuAk9WX58uVWXXrEhj3sicKSJUsSvm/fvj33gTIeAqV/Gyvklh1Y3nvvParz8sWLF1fuxIJ99BkTbRFsYg+7+Fwlvi26fOPA2Jo85Ojb0yc/hpX2Gw85KhpxHggMHQJBsIcO2xG33I5gQ/Y2r5mfpzSYs8s37GohpexE8r9vXZmTPNPR45QTZ6c/WXVHSxvqd955Xdr7RPkOJrPmL0kv7dpWmJl3ybXpnOwH8T5T9wpE+BBx9VH4un6w94crNrYQZsbzrQ1LC6KMjkqn9r9y12tFc3+zonUoMbYdGZbgXiXgCTHXmxaPCbny+19+qnROIo++CtnD5eOJYF922WWHB56d3XvvvcU15Buia0K+dlWONuTbE0SIJPpKXM0WRwge7XRrQMg7P1WCLm1oa+LbQFYh6dZvk501/Fh52BPiCslWsf4hyJs3b9aq/BzCTFsdExUQePS9PTUAYb7qqqv6tQXHMl9oCw7Um5SNdceOHaW+Whu/S0gQbEMmjoHA0CIQBHto8R1R656YKXmE5BEVVqKr9TiOzn0bLm8hcuggGvWGFLIlnQmEGCLZVOoINpFlixpjT30kYvz9nXcV3fBg4aQsCn0wi7Jrbjm+4SPiSTAR8aMnTsrriH7nBPdQGg2R6+/cdV1ex6+jj5lU2LE+lER721qHbbCsI9fWkcfTzyPku86Ov0kyu3HsQyAIdh8OnnRCGqsIIuQa4muC3qpVq1r0IX+IknbfTsky0Vnr0x7EpH1dG+qtDedIGensqzn8u5OxQqKNvB+2cPiM6Dc/JpD9qhsT07EjJJsbGRP6YaeXKtxNz45+rBD722+/3arziLn5r2OgT8vBD4JdwBUngcCQIhAEe0jhHVnjZcQM8gY5g/CplO168fT9N6dnsh8TjQY/v/PO9PiWw/vtamrJ5tXzWwjuzPMWp09lkeijMmL40hPbWtphu45gUw+xnZlFvS2qS7SbSPTdWfTd5NQzF6RFX/n7/JLxbV59Xvrg/YP5Nf0vuHJjfq5b6UGul64/fGOAApiRboF4wuwfAFXdMn0l2N4W/p6/dG0e+cfOw1nEX28KFGs/j/T1maVfS3MXXJnjcN+Nl7e0VTzRDWlFIAh2Hx6edFqqBQS3LCqq0e/bbrutJW1Dt/9TEk1PWoddUi58KkhdG1+HTdJKiMyaKOG1Mj36sdIe0kmEGPv2Mh5rAxbsIQ5ZJZVGU1Q8ydW9xmlPWgbfHNAWAkx7SzOhXr8N8Dhav/gFcade22rf/iaHtozTbkAg7kayscc8IEGwcxjiVyAw5AgEwR5yiEeugzJi5r2BvJ7yyc/mu4sYgTUdJcplZFTJqpE6iPuWdQvNRIK4L7tpV3HNiU/rsLbUlfmsEWh0EE/+fdRW+2BckGNEfaZ8UZbeYoQ6V5BfnhTPvfjKLMf5moLoi2p+6vWNYPubATBfdtMTLXb8uDVS7+v0ZoKOfaRd8fQ+xnVKQbD7VoEnnRrlRINrjSxv2rSpSG/Q1BMIJSTTBOJ3xRVX2GWew61kuKg4dAIJhExCRE2UhHqCDTmGeEIkm4ofK6ka+kZEcriVyCoWfjwQZ/pHfPRaCbD55m9WFC/fL3axb+Ij1Grf16nPtPe42fwFwTZ04xgIDC0CQbCHFt8Rte6JWZkzkMyLsjzlU7OH5lSIAusOI/YQoOpg38QIYROyV6fjfTa71o8dPZn1JJmHMTW1xMiub4c9xsZDg3MvuqIll9xH6a1vcshnnfuFfsTc27Y+/Ziq8qSV/NNXVXtPoL19X29+x7EPgSDYfTh40qkRajR8vZFev7MFhJAHG1XsIT/KylIqIKbkFmt+sbbXNp4oEv2GmHYifiwaVceOrzcyan34KLVh5X1Tv62tx8tIMjcWemNi5daOI/joy4JUx/fNNw8qPHRqEWzKbf6CYCtKcR4IDB0CQbCHDtsRt+yJl+3A8bMsvWLXlvVFCgWOaooH174tZXViRNhHlsvInretOr6uiox6MlvnG3VGVomw77hlZUtKhbXlZmP+5V9veShSXx5jenaEaF+04ia77JdSYn16oq7jLRpnJ0GwFY2hOw+C3YetJ5VGGg15X28EzRNG0686KukkQqvpFk3aeCKp9qra+3I/lnZjbVrvfTOMfP8a8TeS7HG08iZt0fFj8u38tfkWBNsjE9eBwNAgEAR7aHAdFVY9WTUSjHP+QUTI5bJsBxGOiE/10Af8cgX3a0q2ZR1vgWxCsOt0vM9VZNQTbMZWJ37/b3LBX9r1rWJ3Em3r003wCZJsb2lUXcuFpsz7ZAS76ZiCYCuyQ3ceBLsPW0/Q2pFKI2ieGBLB1rQGP3OkYvDjUxogypBKIq/eppJoT2K1zvdVdd3pWNthYfXet2XLlrXkhuOPH5sR6apyHUOdjr9ZIQe7Lm2GqD+52EGwFeE4DwSGDoEg2EOH7Yhb9sROCTbO+egsaSLkJJso4fM7W5iOP3riTuqG3w/aE1El0d5nrdO+NMeack+KVbfunPzox7JovpJnJc3alrSZ57NdS/TBT8XUj8sItr9ZISVl+YbH1XS/bwzUbjtM2tW3dBQXkYN9aA10SjqNYPvUBr8zRtUSY6cNUkMQI5mm63OZlUR7Eqt11r7dsdOxGoE2u1Xtvd+aX21tfQ62pbj43G7I8a233tpCkv3YFTdf53OwrX9/DILtEYnrQGBoEAiCPTS4jgqrnngpacNByOWWtX9cmSriCfj5X/p6OiN7UYwKNt7NfiwH2uduo6vtfPSaeiXR3metQ9fE53HTPzcHFoFHD18gzvryFtr5F8n4mwLbwYNy7KpN7OqNh2JaRbBpow+Mcs0Dk/aSGjD0e42bD+i2w6RdPTZCDiMQEew+LKpIoyHl641gU+9zkv1Dg+iQPwyJtNxgtUfE2x4URId9oNE3URLtiaTWmX67o/aNblMCbXar2uM7WOgDkooFY/J7XNfhSKSfKDhkm+j1xo0bW7bwU4KNbc3PJjoNyfbfJvDNAXZNgmAbEnEMBIYWgSDYQ4vviFr3xEvJoDnmCS/R1aXrduSk0renjT7sCIElOutJsI8uW19VR23v+9Q6337L2oUtKR4QYSLtJthCLJLMOeQYPYjz1EyXhyFf+d63W/aVtmi4EWZ0eUsisidLK9GHJzXabfq5YvZL+/Uk3nTwBRxV/I4t7TBpV6+24zx2EbE1UEUaq+qVGPrILW0geJaiAPGEACoZ9ikNEEEejvQP42FL241mgo2vPkJNGQIW4KDiI9w+bUZ1y86VYFPv55Ayu6HhnDnAB72hCIINMiGBwNAjEAR76DEesR488Soj2Djno6v6YKGPFJcNxpNgCCN7M9vbF7UNudwQVk3J0PbeZ61TO5zXPbCoukp0NfqsOnaukWNPmE3Hjh5Pr6/90qYJlpDrS79a/yZHj0knmJnv4/kYEey+2ffkTEkYGr5eCTb1VcSSOhMlyj61xHQ4Qjz14UdtN9oJNv6zxaDfS5tylao3OfobD23jcfEEG/LMPOl2itreznVug2AbKnEMBIYWgSDYQ4vviFqHgJJfbGIPItq1HSFoT9//t3aZHxdc+dfFlnXU85AfR3t5C0qQQcjymdmruydNmd7SHpJNdPzVLM3CXqDC/s3nXPLn6Ue7H2rJYya1w7YJ9D7PPC97EGr+4hbbekE/5EUTIVZCz/7bJ2S+nXbmxYVt2vH2R/WJMtOdnb2QhvGYQIj3Zg9D6lsrqYNYl/mF7QM/ftGaZ0T5nuLcTkgHeSrD5c0MS8OFuiqb1LXDpF09NkIOIzDWCPZf/dVfHR5cdkZKggnkS0WvIYVEOE20jjJfbw/JmT5H0hgglhw1TQIiCKGcN29eS8oCehBme9sjD+bZS1noz4SUBktrIMrLj4nWWVm7ox9Lu7F2Wk//ZViwZzcRZbCw8ZT5CiaM0bbVq8KFbwn8FoWQbNryo0S7qm8I9vbt2ws3Tj311DwtpSiIk0AgEOgKAkGwuwJjGOkEAR/pLXuRTCf2QjcQ6ASBsUSwOxl36AYCgUAgEAgMHwJBsIcP63HTk+UU+4cDAcDnIpe96XHcABUDHREEgmCPCOzRaSAQCAQC4wqBINjjarqHZ7Ckkuy4dWXiRSykkUzO0kfYaWT/y09le09va3FC00NaKuIiEBgiBIJgDxGwYTYQCAQCgUCgQCAIdgFFnHQLAQj2fRsub2tuZpbzvODKjW31QiEQ6CYCQbC7iWbYCgQCgUAgEChDIAh2GSpRNigE2hFs0kI+k7310R5sHFRn0TgQ6BCBINgdAhbqgUAgEAgEAh0jEAS7Y8iiQRME2C3jR7vZX/pgoX70xElp6ow5LTt1FJVxEggMEwJBsIcJ6OgmEAgEAoFxjEAQ7HE8+TH0QGA8IhAEezzOeow5EAgEAoHhRSAI9vDiHb0FAoHACCMQBHuEJyC6DwQCgUBgHCAQBHscTHIMMRAIBA4jEAT7MBZxFggEAoFAIDA0CATBHhpcw2ogEAiMUgSCYI/SiQm3AoFAIBAYQwgEwR5DkxlDCQQCgfYIBMFuj1FoBAKBQCAQCAwOgZ4m2G/+8rjBjT5aBwKBwLhE4ISj3hmX445BBwKBQCAQCAwPAj1NsIcHouglEAgEAoFAIBAIBAKBQCAQaI5AEOzmWIVmIBAIBAKBQCAQCAQCgUAg0BaBINhtIQqFQCAQCAQCgUAgEAgEAoFAoDkCQbCbYxWagUAgEAgEAoFAIBAIBAKBQFsEgmC3hSgUAoFAIBAIBAKBQCAQCAQCgeYIBMFujlVoBgKBQCAQCAQCgUAgEAgEAm0RCILdFqJQCAQCgUAgEAgEAoFAIBAIBJojEAS7OVahGQgEAoFAIBAIBAKBQCAQCLRFIAh2W4hCIRAIBAKBQCAQCAQCgUAgEGiOQE8T7JcO/ofmIw3NQCAQCAQOITBr0v8LLAKBQCAQCAQCgSFDoKcJ9kP7TkhvH/zNkIEThgOBQGDsIXDK1CPSuVPfHHsDixEFAoFAIBAIjBoEgmCPmqkIRwKBQGA4EAiCPRwoRx+BQCAQCIxvBIJgj+/5j9EHAuMOgSDY427KY8CBQCAQCAw7AkGwhx3y6DAQCARGEoEg2COJfvQdCAQCgcD4QCAI9viY5xhlIBAIHEIgCHYshUAgEAgEAoGhRiAI9lAjHPYDgUBgVCEQBHtUTUc4EwgEAoHAmEQgCPaYnNYYVCAQCFQhEAS7CpkoDwQCgUAgEOgWAkGwu4Vk2AkEAoGeQCAIdk9MUzgZCAQCgUBPIxAEu6enL5wPBAKBThEIgt0pYqEfCAQCgUAg0CkCQbA7RSz0A4FAoKcRCILd09MXzgcCgUAg0BMIBMHuiWkKJwOBQKBbCATB7haSYScQCAQCgUCgCoEg2FXIRHkgEAiMSQSCYI/JaY1BBQKBQCAwqhAIgj2qpqP7zhzYtzd9Z9PqwvDH/uNp6bNfvrm4bnryyvceSs8/fFeaPGVGOudz16RJU6Y3bdpPb/s3lqRf//svivLPX/fNdPTEycV1nIwNBO65/k9bBvLFr/+vluuRugiCPVLIR7+BQCAQCIwfBIJgd2muH9tyfTqwb08/awtWbKwlo0/ff3Pa//JT/dpdumZrv7KBFECwt6z9o6Lp73/68x0T7AM/3pO2rFtY2Jhy4uy0dN2DxXWnJxDsn/zfZ4tmK2/9QU8SbHB5bOv1xTg4qZvvPbu2p71Pbi/0z7/86wksx6r8zzWfST87sC8f3pFHT0z/+b/3//sYibEHwR4J1KPPQCAQCATGFwJBsLs03/fd+MWMKD/dz9pnln4tzV1wZb9yK7jj6o+nD94/aJfF8St3vVacD+akE4K9f+9TGSF6I82ev7ilS6LXO25dWZQdfcyktPK2F4rrTk/GCsEGr/s2XN4y/GkzP5Wqbo64mXom+zFBD/2xKkGwx+rMxrgCgUAgEAgE2iEQBLsdQg3rqwh2XbTXE1ftargI9sGMUBOFhSx+8N67adrp56RLv3qPupKX77jly8UNRLubhpbGJRdjmWAz3Cp8gmBHBLvkzyGKAoFAIBAIBMYgAkGwuzSpVQQb88s37CpNE9l553Vp7xOHUwbUleEi2D4KW0awzS/I+FFZrvRg86XHOsEGH1JofJ56EOwg2Pa3FMdAIBAIBAKBsY1AEOwuza8SbEiqpouURTSJFm9eMz+PDuOCbzMaCXanUEHIPcnERqcEG6w6JfXkRyNNcpzx89139udzkDdq+MvfnGizUz95cVq06g4tSgMh2PQxGtNImsxJpIi0TH9cBAKBQCAQCIwjBIJgd2myPcH+4L2DxUOPk7KdN5ZveLylJx54+85d1+VlkGtESbkS7HYPx/mH7Waeu7jIo67KwbY2ECUjo/gAkTVSanZMl3pkyozZ6fylX++7kN+QQXYa4YhdE8Z//uVfS5BOpB3BZryvPv9QPzv4RX74GRetMNMtRwjsS098Kx089GCdVdKOhw9tXJSbr6TpqDD++dnDhz4PXXXsHBuag+1vkiDYNmbaNCXYjP+ZB/6uZRxgOHv+F9K8z11r3efHQa0N95Am8318hpWlDDEeSxcCp1ee35neyMas+JpfzIm/CQqC3TJVcREIBAKBQCAwjhAIgt2lyfYEG2L1+JYbCuukDCjBI6f5ld0783oi3BCYKoLdjph5ojfvkmvTOdkPUkWwfZtc2f0yO15XiZc1YReV7++8yy77Hc0WFXUE2/fVz1BWUBYdVvzL2ugDhZBSu7kp01Vfy+qtzPsKoX48w+Hdd97IVSCcy7L0ICOe7eaRm5KHs5suT/qtP46z5i9JF624qShqZ9P7qGMrq6Nv2w1H5/mWFScXfZadsLb9zjJBsMuQirJAIBAIBAKB8YBAEOwuzbISPIjJgiv/Ot2dpYCYnLFgRRH1hUjdseoTVpXnaO+88y96lmBXkWsi3UQ72SVFiV0Tgs1OJZC2aTM/neO0NyPFRlwpqCPM9GuR43ez9I9Xv/fttOgrf5+nWoC9pubQz8yMtEKCqXs1I5gzsyi53aAUk1Ry4gmq7R6iUW29GWhHhj2Op565IMNgTvYNw4vFzRhu/GEWjbcIezub3kedB19HNFqj055g25zgE3jt3/vdljXrU6GCYJcsmigKBAKBQCAQGBcIBMHu0jR7gs1X61vWLiyigZAXSxPRCCokCvKn7XFJU0QGQ6KqItg2bE+ylFQ10SF/WW8kaKMkjmv6gLwa6a0j2KSj/Gj3Q9nWhq0pB/6mRG9Y2uFDW6SPFLamdXhf0UPfos5cV4nHzki/J8qWKlLnp8fR1oX1retDo8V1NmnrfdTx+jr0Jx83PZ2dffvx0exFQjpn9OPnBP3Nq+cXNz5+7QTBBqGQQCAQCAQCgfGIQBDsLs26EiAjGkqk6cbSRDQ9xKKR2h7dXiHYz++8syUVxhNDxuKljmB7XQg3EfAD+15Mux/aVERYDWP0PcmEgJ5zyZ8nfPHiSSVE+jNL1+a6TUi12vO2jGBDTLdmN1cWcefmaum6Hen5LIWmah9sPwZbK9afr7f14cvNB2vnfWxHsKt2vDF7HO2hUL4doH+LeutNJHpBsEEhJBAIBAKBQGA8IhAEu0uzrgTZyB9ES1NBiLqSeqBl9hZDbY9LRqA4HwyJGuoItt4s4KsneJR5aUewIYVPP5C94TI7VolhTL0nkdYGwjz34hX5Q5FKnjXqarocyW+ee9EVLbnyWu/Pfb86dl/H3ONDFcH28+/78tdGwAezNryPiqnvjxsdmxPWdZXoug2CXYVSlAcC7RG49dZb04svvlgorlmzJp1yyinFdZx0H4Gf//znafv27em1115LU6dOTcuWLUvHHntsVzrasWNHeu6559LEiRPTkiVL0sknn9wVuyNtBMxuv/329N5776XZs2fnYxtpn0ZL/0GwuzQTSpCUqCgBJcI373PXFA/YabRX2+OSEpXBkKihJtjebyWZVdDWEWw/VmyQtjBpyrR8txN766VijA7fFuzasr70rZjg/ier/kdBnCGL7EFuD/PRXsW+VdCysnNPUP3YfaoI0eNuEWzry+Nl5eav97Eugq1pN9aeIw8+6ps8KbN8bPuGgTJE120Q7D5Mhvv3I488kn/gab8LFy5My5cv16I4H+UIQLAfe+yxwssbb7wxCHaBxtCcQII3b95cGL/gggvS1VdfXVwP9ATCzg2SCeT9tttus8uePt50003p2WefLcbATcmiRYuK6/F8EgS7S7OvRFPJn08TgVTb7iFK5LQ9LilR8STKcnrNdd+HkqheItg+D3nmeYvzB0Mt+qwYKcaGA5HVl57Yll7a9a1+5FlvZkwf4oi+zYeV05/u/mHl/ujJqye3+LN59XkF6ceuRn9VX8dGPxbx9n3a9ewMG/YYH8za8P7rurF+ON5x9ceLMYA7D/Da/ubeb123QbAVxeE79x949DyWPtCHD8mR7Wk8E2wi90R7IWvDKXfffXd68MEHiy7PPvvstHr16uJ6oCcQUP4uTYhi01evCDcIe/bsySPUPvK+bt26vM7GQnSen5CUgmB3aRUo0fDkTwmKdmfpIZRpe66VqHgC7SON/o2QSpSGmmD7HGxeigJxrJOqCLaPlFoaBLYgpppa4zH2/RFZBVOLePv8YNXH9sPZLi5KtJX8qq6ee4Ja1saPSdurfjuirO30fDBrw/uv68b6AMct6xbaZb8UIH2QFyVdt0GwC9iG7eTtt9+ujLhBFCAMIb2BwHgk2JDORx99NJF2gNx7773DOln8/UAYOXJTyt+MJ5QDdchufCHXfJt04YUXDtTUsLXj27Bt27bleNDp2rVr05w5c1r6h3yDmaWIgFm30mpaOurBiyDYXZo0Jcie/HkCTJc+oqrtqVei4okQkdCLsiji5OOmpf0/fCo99o/X06QQJUrtCDaN/B7H7GoydcasnNTywKDvX8fno87Yg2TzkOGUzAYEDR+PyrbDm7vgSqor98H2ZNS2fYMAP771hiwyvS1vzy/1AZL528d+tOWhRtpsWbeoeADP9PHnlewlNv7FKD6dQ8l90ak78bgoYVZVTRPSctX3ONocs05M8P35hze17IPtfbB2TdaGb6vrxvr0OroPt8eMNrpue5lgv//v2R7y7x2RZnz0NwZFTxz1K24IAkTBpFtfd5u9OA4tAuORYEPUiJSaDDfBtn7j2IcA5JofkzKCbXVx7I9AEOz+mAyoRAmykTkz5Ikj5ZoewrW251qJCtdVD+ZRRz6sRWq5VqLUhGBX2TY7nmT58fnoKz54MVuUV0WwfbTU29Bxqg/av71WHFuQbBPrX8fCzQOE9N0D+wsijr7atvZlR7VFvRJm1ccPTRWxOq9fRlhNV4/dWhvef8NI++K86hsYr8e1+tarBHvfz45IGx/57XTcxN+k1Rf+Ih1zZNlIR2cZ0aPXX389d45oNZFAIyxElTZt2tTYcSJTRKWIuDWJ4nWq39iRGkX65EaiLGJGmsGECRPSSSedVFpfZRbMDENs8zMS0inBtvHia6djtvFxQ2bR2ybjVqwG2qf1zXGwBBsMfIQVu+onD+J1WzrFrWn/5nfTv0HTx3435mOoCTZ/v1X/WxgL/3+arEPGO5Cx27zRfij+1oNgg2wXRAlyGUFTkgJRXHnbCy29ansqlKhwXUbSKeelKuxMog+hKVFqQrCrbJsdT8TKxqckF7+8mC3Kqwg2dVUkk7QYXrhib7tUH9r1TS73gis3Yr5fND4vlF/giS7ku514XDxh1vZlGJfpl33boXY479ba8P7rHGmfPg3I6oiuc/Ngc+J960WCbeT6/V+m9LvHfpiuPu+Dnoli82GlD1JdddVV+YeOPrTVLk0EgkIOqj60ZPONPf+1dp2+PljZjjjVfZCX1fHByLj4ULWcT87Nd7BQgYDjT11uKF+H09635YPXvvb2D7xxw+LJvd7kDCbXtgnBxld8xncvEM3Fixf3I5zM2fr16wt1CCc+b9y4sWXXEht3GQECf+ZF+zWMMUydSZOop18f1taO+IgO4nWxz3oFB8TWg/nIeDlXYWykafiUKb/WzBZtPW7U8ffADhrUmVThdtlll5lKnsts46FQ67hmXbGbCd9ImYAvOen+b5B61r7Xp5yHDc8666yW+dYxoVMmHocyHft2wevqfJfVMQ7+dhUzxmQ7trCmaKdzxlzx91sm2GHsag891j82y9Yv9ulH+6ANvuF/WRvqO5Ug2J0iVqFPmsLBQ6/InpTtemFv2jN1CNbb+/q++iqr1/a0KXuTIFHZ3dleyrbvMG855OUfv8xIzp4ntltXafrpn8rTNCiAYO998ltFHSfzv/hfWq65gGxBpCzqS2SXlA4iwqQvqP0y/7GBHv5BhFVmnrcknZa9Oh6byK57/ptWp7P/9CtFHRVgsTd7+BChzaysPS+pUYzUB/OPNwuq8MbB0868uMCCOsbHg42M18ZKOTna0zLc/LxRVyXWr9Xbg4d27Y99+B4siqv08Y2X7SiO4MBcnHbmZ4sHDAtD2clA1ob3X9eN2ubcrw/mg/Whc4Kerls/z2XrjjbDLadMPSKdO/XNft0quZ72sd+kNT0WvfYPaPEhTQRISWHdQ1uaXtIPnKzAfzDzAQW5qJI6UmQfzta27IPYIpG+Dj8oMzG/+IBV4mj1eiwbP+TEk0ttw7mRBk/u/E0HH9iK92DSctoR7Hb42xi8jx4nIp2sE082aA/hwA+OJuhxIwduZQI50ZsUw65M18o8rlZux7q1BMbkbZvYevDrxur1CAHTHS98G7NFG48bhA9i3xQ3JdE6HmxrHdes07KbXOrK8NSbOnRU/HzomFRPzz0OWmfn9jfsddU/XwfekOGytcM8Hn/88S1/29YXR7+OKWvyN6D+0KbdWvP6tBmoBMEeKHLRLhAIBHoSgTKCvfvNj6S7nz0qEbnuRXLNREDs7MNeiaT/8C2LuvJhrrsc2MRCBBDSTPSDuYm+kgj/oWYfztaP/yDWDzlfB9nTD2jzywgQEUTGb6TwmWeeKVI+6E9tc20Pn3Fugg1+6Id0EWvjP9A9gfb1GzZsGHA0rI5gQ2D12wr89j7bWDia/5wbTpyrMF82Xi33JNTPJbq6TrQt59q3r7PrMptWx7FuLake57YebN1wAzFv3rxCjRtJbigQfwNhbUzZbHE9WNyUROt4sK11XCOW4mEpXn2lfeSbv2kT7zPl1pZ1YmM1fR2TlfljmU2vY3/DXlfn29dhw9Jdynyrq2d969aGfj6otyg3/dq4mWP+5yH+/xY4ET1Hh/+d/P1C5O3mPm80iF9BsAcBXjQNBAKB3kPAE+wnX8/I9TNH5QPpVXLtPzg02uMj01pns6fknDKII1/LGkmFePEBRDQMaaIPMYDoIp5A2YdzXpn98h/E7T6k8Y8PRgQf8YsPbH6sPK/MfuH7FVdcYZcFAaPAf0hT5vHBJn3wAY6AS9mHN3VK1j0hoL4TqSPY2g82PdH3eEIYwBQpGzOEzebKt9WbNdaARughS8ytrQu/DulP55LrOmm3TmjrdSiDNBrJ55rx4gv+ebLkfVT//NiVjLbDDfKnkXTIHmvFREl0O4Ktbb2/fl3peqQvXb+sXfCy9Uq9jonrOvF4KFbWrk7H1zEfrF3GwN8la0l9o5ybUiO8rEut1/8b+jegdvHLz5Wtb++PldtYOOKX/d/T8oGcB8EeCGrRJhAIBHoWASXYSq4/fcqv0xVnfdCT4/If7lzzYYXwIavRTiVb1PsPI6I6fHhVibfnP/DL2nlSpB+U6PsPPv0g93VN+uNDksgzhPCtt97K83MpQ5Tc+LSaJuTD+2Mf0thXIu8jv3nnHfyqI9hK1iAXzLcnBf4myL65aDfffn4VL3+zpkTQhubnWufSdKqOvq1fJ7TzOv7mosw264AfxsaRcZjonPu51TqPm+KCrXb1Ome+rdZhy+aKc8STaMOlXZ+0Hcgapx3i8SibyzodX+fXi/+/pXjTv59r7V8xK1sDWm92/frlfwl65Kn7vx/6H6wEwR4sgtE+EAgEegoBI9gP/8uR6d7dfVuE9DK5htitWrUqj7wwEXxQ8EGmwgedihJw/yFoH0aqr+f+Q6qdPm39B6URBLPrfdAPUl9X1x++EUWESFWJkhvvl/Zb1R6CplFcIw0+0qgYV9mqK68i2L5/HY/aqxpbE1Km5ETt+7kow6uJjvqp595nv07Q9TplPqDH3wX5vqQIgVmV6Hryvmudx83mXe1W4YZO0zp0/bj9mK2+yd+iX5c6JvqqE49HGdZ1Or5Oo+v06+u9/ap6Pxd1Y6DOCDjrwEfFrS3ffIGNBSasfDDHINiDQS/aBgKBQM8hAMF++fV30ndf/a3c9wnZ7yP4NULyXxf+It8ScKDd+7zfJnY0uuo/xLSuzJbXb/KBXUUQzL63qR+0vs4ixtbWjvqVMWVEdkld4IYDkmGihNH75SOH1sYftS+LqGs0TtMqfNum11UE25MLHY/a9mMzTJu0ryKDOm76Mpvar5+vMh3V13PvsxHJOp0y+5BrHnjVGy3myX40lUPXr/dd6zxuWmf+VeFGfdM6dP24q3Cp8xc7yGgi2H6uvP9N6/1c9I20+rf+jbAmWMdlN138r7juuuv6pRVVW66vCYJdj0/UBgKBwBhDYNrvTEiP/vO/pv0/PWJUjOwbiwZHsD3paTIoiCe5joj/kCsjDmrTR83KInmqz3kVQTA9X68ftN4/rbP2fGhqGowfQxW5qevXbJcdPWkBS2xZvqiP1JXZaFdWRbB9KorOpdrkRsT286bcIuqenCj5sPZVePm5KLvZGSim9O3beqJZplO2HvwaVR3/DYCuFT8+rfO4aV073KivwtTXce3HXYVLnb/YQZro9Gn2/+3bKo6mXadTV0f7gdb7ueCG1n9rZ/5xhDjzd6LC3zDfbujNFvVVf0/atul5EOymSIVeIBAIjAkEiGCf+bE304bsZTKQ7KM+ktIfzfxV+t3sxTIjIadP/c2AI9ieLBC11W3HdDyQDiOAlNsOF/7Dig8jyB3HMvFkFj1s1X216gmCEjMII+RYI0r6Qd7uQxgfPaFSguIxUkLp81N9fnrZ+K1Mc5z5gLcoOXOA3cFKFcHGrvbNtc0l54ifI4uyU+fnW/GgHqkigx5nj5fHGls6l1zXiV8nZd8oeJ0y+/ptgh+f/8ZHibJfa1rncdM6G1MVbtQ3rUNX1y/XfsxW733SeaYd4tdKmd99mv1/ezzKsK7Tqaujt8HUK55l4+4/mvKSsv8/hm95i+alQbCbYxWagUAgMAYQsBxsXoduJJthLZ/3y3TuSb/qqRF6wmO5hmWDUNJBvUae/YcwH1h8gHFE+CBX8u4/8CHZPIil+pA8iDTiP0iJEvFBTzvqsK+iH+S+rdZZG48DvjA+CB8RfnwxUcJVRgghjfhmAiEDV8pV1C/GwQc14ueAPu6///6iKWO/6KKLiuuqkzqCrX3Tnv6Jmpsf7E9u/lCvpMqTMsUDXUTJi9aX4cXNBTd11EFMOKqUzZfW67lfV+TF8vPhhx8W+HudMvv6rQ6Y2Nf+jJ09z6uw8bjW4aZ1NoYq3KhvWoeuJ3h+zFrvH4C0vy3sMB5d+5SV+U15mXg87G8D/Jh3xOvofNTVtWvbrt7/P8Mfxsb4EeaanYz0/xb/Jwgy8L+BdYEwFr3B79YNMraDYINCSCAQCIwbBIxgM2BI9l3PHp2+v78vH7vXSLYnxhoZ9hPq0xo06sOHsKZY+LZc6wcz+nzoa0Tct1Fi5kmd12XnEk1n6ORDGltN/Lc+1S/KfBTb9PSo/lh5GdmkzkeT+RoaUmdiqRp2XXWsI9iQAvBXzKrs+F1h/Fx4PLBTRwaVvJb1CUHRdVGGXVk7yvyNkumpj4xb94Yus19lB3veP13XnhBqncdN68zPOtya1mFLCTTXfsxaXzdW2nop89vr2LUfs5VzNB88ZjofdXXYGEw9f3/8v9O1hk0vOl7tD4LN/0D+d6iovpYP5DwI9kBQizaBQCDQswgowbZBbHru6OKhx14h2Z5UNom8+GiXkkE+TCFPVR9Y/oOH/tH3EUvDVEkRZT7iZHoWTeLDz6STD2lrU2WfyK6+cdL7RXv94DV7elR/tNyTTb1pMb2tW7em++67L7885ZRT0o033mhVtcc6gk1DSDb9K9n0BonqWWTb6jxpKsOjjgzW9Uv03r+Nrwo780ePVTcO6qMnm1X2IV/+BoS/EdrrzaSua78OtM7jpnU2hjrcmtZhy8ir2fVj9vVVa5+bK6LO9hp57LV7iNn6tKNf41ZuPnjMdD7q6rAz2Pp2/4PoQ8fr+6NexX/7pHUDOQ+CPRDUok0gMAgEeEX5u+/sLyxMmTGr5VXxRUWcDAkCZQSbjpRk/8Hpv0qXfyJ7reMoFj7wlVxB7vg6vU5Id1BCDHHR1AcIDpFubJsedtGBrNnXqtoHNr0+X9Pii9c323wwUge5xrYfC23pF6mrUz84V18MD+zzwWpi5XZtR8Zr7a2McdjYrUyP6Ct51w9z0/vGN76Rdu/enV9+/vOfT1/60pesqvb4wAMPtNR//OMfTxB0L2CJHxxNzGf7utzKOdo4rawMjyZ40SfzyZrBBm9LZI14Mtg0Ym/+YA/brG3OWSesB2wj1Nna5FrXCtcqRHfNDliQyoKvOj79G/BrTes8blpnfapdj2vTOmxB3lX8mH09uvhuemAGXmBDv9q3EmDto+4cu9g33LFtz3p4zHQ+6urMZ/0fpm2b1JvP3j/m2jDQvwH8xyf+VuzvBT10GJPqmu3BHINgDwa9Mdr2wI/3pA/eP1iMbvJx09KkKdOLa3+yf+9TKU2YkBcffcykNOXEvtcre7247kPg6ftvTs9kPyaXrtmaps38lF3GcYgRqCLYdLv1B0elf3r5I7kHvbw39hBDGOYPIdCETK5cuTL99Kc/zVsMhNz0EtgQYt2THZIJwQ4ZOQR8GlnZg6Mj593Y7nkoCPb/BwAA//8V5uUPAABAAElEQVTsvQuwHeV15/vh2MiIkewEialEEmDIGD3GRnCNwCBAiRGeSC7bhSUs5YUQZZzLY2LHknzrZgDJnsSFJN+Q4pEYl4SUVF1hwCT2RNwY2R4eAoOgeDiDAKcsA5I8GSQyARkRbMfc/rVYrbXX6e69zzn7nLMf/6/qnO7+nuv7f33O/vXq1b2P+MlPfvJmGmT66U9/mn7wgx8MqtU9u38t7Tvwi0G1UeXmCtx13SfT3uceKSpOmDQtLV29NY0bP7HI8zs3LD+hOJxy8hnpws9/rTjWzkAFHvnG9WlH9mPpwlW3pSnTz7RDbUdYgfdMfls6a/KPK0d56IW3p007jszLPzfvjXTy5H+rrKuC/lXg6aefTmvWrCkEOO+889IVV1xRHLOzb9++Iu/oo49Ot956a0N5Nx4wb+ZywgknNJj/2muvpfXr1yfKLS1evDjxozRyCnCOPf/88+n0009vGIT12Lx5c7r33nuL/JkzZ6bVq1cXx9oZWQWmTp2a3v3ud7d1kCME2G3Vc9Q7i4CNAafMX57OXXpNqS0jBdgH9u9JezLQ3/vcw2nX499Kl934/dLxuy1TgD22K9YMsLEOyCaddfzP861+SQEUWLduXTp48GAuhgfJ8ePH52WTJ09uEOp73/te+tGPflTk/fZv/3ax3607d9xxR+IHwGa+bHfu3JlDHlBnibK1a9fmMG552rZfAbvQQ+9jjz02AdEANz/AtyXOUeCa9VIaHQUE2KOjc1eNUgbYTKDK0zpSgO37ZfyrNj7PpuuTAHtsl7AVwB5bCzV6pyoAoACTMa1cuXKABzHW6ZVjA+y6+QB7aCKYq1OpPWUG2HW9AdfcXYle7ro2Khu+AgLs4WvYcz1UAfak42ZmoSJ3D5ivB+F2hoj4fhlUgD1AemUMQQEB9hBEU5NcgQjYgOTll1+eZs2a1TcKPfroo2nTpk0N3lGbPHrMmzcvLViwQJ5rE2WEt3ipuVPwwgsvDBgJsAaqL7roovxuw4AKyhhRBQTYIypvd3buAXvcURPSG68fKCYy52OfSWdkPz55EK4DbEI+dj15T3p1/968+bjxE9KMsxalCZOm+u7S3mcfTumIIxJ2+GSx3dgE7Fs9q8PYPu1/cWeD7b78jYOvpv27nymqT5o2Y0CMOe2x942DBzKbd6fJ2ZiTps1K9FMWj878Xn350Nzo2Pp8ctuG9NNMwynvPTOPta7zYEe7fD+FsdoZlgIC7GHJ1/eNLTSE2/EAZb8mwkEIQ7CEt5rYbKWxUSCuR7+fn2OzCo2jCrAb9dBRpoAHbINS/9AjXmwA11IrgH3/li+kp7ZttCYN29kXXJrOWXJ1kef7KzLdDjYB21tv+FTa9cS2omTZ2u0NsH7LFe9rAGxiuA2MgfO71i4p2vo5Adb33/aFQwBf1Di8Qx9zl1yTZs5ddDgz24vgvPDKW9Ij37w+0R/JLk5iPR96E3WaNG1mNtfbCrvzjvRrWAoIsIclnxpLASkgBaRACwoIsFsQqd+qRMCef+mX05Zrf6uA1Rgq4oHY4NdrFkHYl9m+f4jS92flfmtj4Bl+YMsXiyKA9sTTLsiPgdotqxcUZez4cg+5E4+Zmi5et71oB3jjSW6WPrR8fQNk+z5pi53+wqQZYEeb8dRzIeEvZprZpPLmCgiwm2ukGlJACkgBKTA8BQTYw9OvJ1t7wOYVfcvWPpAizBosIoAHYoNfE2bn9jvTdzausMM0/exFaf6l6/PjbRtWpGcfvLMoMw80oEryr7LjmDFJEzIgxntMSMamVXPzPH55SI+wG8s99HubNq+cm4V57Cn6xIM8Y+4n0pFHvSv96IlvNXjM8WRfnHnNzSteNiYd0QcJ+Ce8JtYzD/aWaxdkYSuHH6CKAJ93ol/DVkCAPWwJ1YEUkAJSQAo0UUCA3USgfiz2gM387eHCmG9AXAfYHhq9p5h+8RLfcuX72c3TOUuvTrPnX2qHDeBOptlRVMh2PBB7uI+20sZ73n0782zHsBHsXbLm7gKg6SNeFHibIzhXeaBjPQB7T/YqQn9B4aGfcZXap4AAu31aqicpIAWkgBQoV0CAXa5LX+dGODWwxWPsQ0X4chTgsAqwI0ADuCe9FcJhAj/yt4e81RxHqPT9Um52sG8pxixbHWuL19uDq71L24O9xWZH8PXwbONFCPdQH9t7L7+1ZxvrMY4PdcHjvTQDe6WRUUCAPTK6qlcpIAWkgBQ4rIAA+7AW2ntLgSrApjiGigCHux6/p4g19sAZYbSZwL4tdQ2SrZ3Bsx2zZeytN15WZPGw4qvZhYDl5Q8aZiEnFnrBBQHJHnD0Y0bwtdCNovNsJ85psO3pK45DiImP+S4b19ug/eEpIMAenn5qLQWkgBSQAs0VEGA316jvatQBNmL4sA/gkNfs2ZsyPHBGGG0mpG9L3VYAO9YjbhmYtjeWAOXey21x3ObV9l7qCL5loBvn5G1upT32xnrEjpu9lNudAfaV2q+AALv9mqpHKSAFpIAUaFRAgN2oh44yBZoBdnzbhRfNA2d8CPHEU+enhVd91Vev3W8VsOMDi9gHZJst3stNHhcF9no/iyPHkOid9w9NmqHxoU1fJ4JzGaDTT1m9R77xZ8VdAOpUhZdQpjQ8BQTYw9NPraWAFJACUqC5AgLs5hr1XY1mgI0gERJNJINaO/YPEwK2QGfZa+eA8fiFM74t/XkYtv7ZejBmfHs1nkGqjwXHhnFHTczfFBIfuowXDtQl5MTsop8tqxdmby/ZXQxvD0iSETUZDGBPzO4C+Ph2+qtqT5nS0BUQYA9dO7WUAlJACkiB1hQQYLemU1/VagWwEcSHiphAEbCjxxdonTF3cZo4aUoRd7xz+9fTjLM/MeAbIqMdvDJwZvbKvFf27Sle9ce4EYzNFv/lMWW2eu+ztYljYi8hG+PGvyvtyb6cxsN1fBhxOIDNGP5CAXuY79LVWxveYmJ2ajt0BQTYQ9dOLaWAFJACUqA1BQTYrenUV7UiZJY9XIggZWAbAZt6PoSD47Jk3mZfFuHcysrGiN5uXpF32U3/YE0a4rAt03ufLQ8v9V3XLSkeirT8uMX7zTcsmneb8uECNn1E7Xl3NnYqtU8BAXb7tFRPUkAKSAEpUK6AALtcl77OjZBXBdiIFKGyDH6t3lP3bCi+DdILjCf43KXX5J5in8++f0DRysrGiO+njvHePg7b+qmaF5DNuP5LcKwNW14niL14t32KWlSFeNTVi3Hr9O8fxPTjaX9oCgiwh6abWkkBKSAFpEDrCgiwW9dKNdugAF5vANYSscfeC2z5fgt08uo92lnIhi8fyX3eGuIToRxK3a2AALu710/WSwEpIAW6QQEBdjeskmyUAlKgbQoIsNsmpTqSAlJACkiBCgUE2BXCKFsKSIHeVECA3ZvrqllJASkgBTpJAQF2J62GbJECUmDEFRBgj7jEGkAKSAEp0PcKCLD7/hSQAFKgvxQQYPfXemu2UkAKSIGxUECAPRaqa0wpIAXGTAEB9phJr4GlgBSQAn2jgAC7b5ZaE5UCUgAFBNg6D6SAFJACUmCkFRBgj7TC6l8KSIGOUkCA3VHLIWOkgBSQAj2pgAC7J5dVk5ICUqBKAQF2lTLKlwJSQApIgXYpIMBul5LqRwpIga5QQIDdFcskI6WAFJACXa1AVwP2yz//5a4WX8ZLASkwNgoc8/b/PTYDa1QpIAWkgBToCwW6GrD7YoU0SSkgBaSAFJACUkAKSIGuUkCA3VXLJWOlgBSQAlJACkgBKSAFOl0BAXanr5DskwJSQApIASkgBaSAFOgqBQTYXbVcMlYKSAEpIAWkgBSQAlKg0xUQYHf6Csk+KSAFpIAUkAJSQApIga5SQIDdVcslY6WAFJACUkAKSAEpIAU6XQEBdqevkOyTAlJACkgBKSAFpIAU6CoFBNhdtVwyVgpIASkgBaSAFJACUqDTFRBgd/oKyT4pIAWkgBSQAlJACkiBrlKgqwH7Z+ntXSW2jJUCUqAzFHhH+nlnGCIrpIAUkAJSoCcV6GrAvmf3r6V9B37RkwujSUkBKTAyCrxn8tvSWZN/PDKdq1cpIAWkgBSQApkCAmydBlJACvSVAgLsvlpuTVYKSAEpMCYKCLDHRHYNKgWkwFgpIMAeK+U1rhSQAlKgfxQQYPfPWmumUkAKZAoIsHUaSAEpIAWkwEgrIMAeaYXVvxSQAh2lgAC7o5ZDxkgBKSAFelIBAXZPLqsmJQWkQJUCAuwqZZQvBaSAFJAC7VJAgN0uJdWPFJACXaGAALsrlklGSgEpIAW6WgEBdlcvn4yXAlJgsAoIsAermOpLASkgBaTAYBUQYA9WMdWXAlKgqxUQYHf18sl4KSAFpEBXKCDA7oplkpFSQAq0SwEBdruUVD9SQApIASlQpYAAu0oZ5UsBKdCTCgiwe3JZNSkpIAWkQEcpIMDuqOWQMVJACoy0AgLskVZY/UsBKSAFpIAAu0vPge9u/r8bLP/Ni/+04biVgzcOvpqefejOxHbGWYvShElTW2lWWucf/vtfp30vPlOUHXv8zPQf5/1ucaydkVPgwP496ZlsHceNn5imZ+vIdqhp5/Y700/+eU+a8t4z05TpZw61m45uJ8Du6OWRcVJACkiBnlBAgF2zjAYusUoz+BhquzhO3fFf/1/z0r+89Hxe5e3vGJf+z688V1e9tGzLtQvS/t078zKgbOnqu4cM2QD2vX99dTHOb/z+f+1awH7kG9enI44opjJgB60mTZuVppx8xoCy0c7gXNuyekF+kcTYk46bma/jUOy4f8sX0lPbNhZNF155SzrxtAuK417ZEWD3ykpqHlJACkiBzlVAgF2zNnuffTjdtXbJgBonnjo/LbzqqwPyLSOCiuXP+dhn0hnZTzvSYACbeZR5I29YfkKDKReuuq20XkOlioNeAuyoS8WUc0/x7AuWp1POXz4sr3FV/63kl52jV218vpWmA+rcdd0n097nHiny23m+Fp12wI4AuwMWQSZIASkgBXpcAQF2zQKXwYtVX7Z2e6W395Yr3pfeeP2AVS227QSWZoC96/F78rAB5kAISBl0bduwIj374J25fZOmzUwXfv62IYNiPwK2LeyESdPSR678Su49trzR2rK2d123pLgTMf3sRWn+peuHNDye+x3ZD2ncUROy8+FrYzKnIRk/iEYC7EGIpapSQApIASkwJAUE2DWyRcAGOgycz1l6dZo9/9IBrQHbrTdeluf7+mSMJmBHb2QZYGOTATge7uHE7vYyYLNupFezcIwD+3c3eHnzguwXoRncARiOhtbXYLdANuvI2GV3KgbT3/4Xd+bznJzNZzgx+YMZc7TrCrBHW3GNJwWkgBToPwUE2DVrHgGbmFu7hV4V6+q9wr4+w3QiYNdMf1BFvQzY8eKEuOdtGz5XnAsmVDvX1/rUtv0KCLDbr6l6lAJSQApIgUYFBNiNejQcRcAGoOwWOhXLwkR8eAix2rue2Fb0GQEML7M9TUeIxrlLrynqslNXXhUiQvw3Dy7iicSzack8m34cq2t18MCWJd4s8aMn78k9m/TLxQXeUt5YMXPuorxJM8DGFrTY9cS3BvQz4+zFacbZnygbOvMY70lPfvvW7A0lT+deWioRkjH5uBlpdhb7bPOyxk9u25CPY/PHTuxlLcruOFg7v40x2BGwrW68S8BYl934fSsutszhkW/+eTbvzPudeZqxZ/Jxs7I3t3xigP1Fo2zHz535kKztnI/+YeEtrztPaMOYzzz09VzD2M8ZWT/mqWadeauMpXOXXJOPZ8e2tfOBfu0cYx1Oyh6ILHuLCWPef9sXrHn2gOihc521eubBr+fnqs3rnCVXF/MqGrR5R4DdZkHVnRSQAlJACgxQQIA9QJLDGRGweauChX9QK4aJ+PCQicdMTdMz+PRAHgHbgxzebmJefaorrwLsCH2+P/b9OLFuBEnAaNvGFTkAxX5iX80AO44V+wOweIuJT+j57Wx8gzhfxr7Xkzo8kGoAGev6eceyeOx1pyzqYvUB4E2r5tphvmUOzMUSEPnAli/a4YDt7AsuTUBlTFUPylo9/0CqtzfO08dVW1u/9f3Eur6MNs3OB+pwkUE7r0H8O8JGLpIs/p92lsrOAytr11aA3S4l1Y8UkAJSQApUKSDArlImy49gADh8O3sw8NWX9+StIgz48BDgj9StgA08+te/5ZMJvzzMDQawaUc6sH9voSXHHpg59ncDOLZ27BOq4+t77SnnAmfCpCnsFh7SeAGTF5b88sBKcRVgU7Z55dyGOXgoxdP7newCwZLZFOcdX4cX52Lt/daP4+31awIQs4aWeCbAwPeNgwfyOx2+nzrA5gJmy+qFeQy69Ve1BbL9Kx/j31FVO8v/0PL1xZ0Ry2vnVoDdTjXVlxSQAlJACpQpIMAuU+WtvAgGwMi+3U83eCR9mIgHQvJ3Zm/oGG3ABqqAoarwDwuZYIrRq+xBMkIecHhOFsJiIRlosy8LRbHXDjYDbOwhNMBCSkx2/y5uvJrL1j6QF0XtT5m/vCGEhjlyEWDA6EGXcZauafSG05/ZbmNXbT2wUsfrEttEDQ0OsW9z5t1mS8Im/5YW365u3rRl7jOzt4MwV9b3h0/ck0469YJi7t5eD9gRmCPIox/JQkRifQ/f8XywMA80pZ8Yk+7fZhLXkjEpt5CorTd8qiGe3belbruTALvdiqo/KSAFpIAUiAoIsKMi7jiCAcAxMfu2Qx8WYGEi3ltpgBeBxXtcGaYKjMyEuvKqEBFr6wGOvDJIrKoDFN5y5futq/yVbUvX/H8FiBUFbqcZYLuqxS6wSHz1M9vvKPLMzqg9cElMcBUk+4sbOjv/0i9XxnUXg1XseN2pYjaVVY8aGsT684F2lm99xHOD2G0ufiLIxgsLa++33t46wOZLY9DQgNr3wX60yQN21NdfWNJ2wDnj4tHjWnKxdvG67TTLU/S0+zlYnXZuBdjtVFN9SQEpIAWkQJkCAuwyVd7Ki2BgwFHmdcULZw80GnRHYOkWwI7zbsWj2ApgE49MaAf9m2c3ym8xzJRvXnl28VpEq4e399QLLhnwMF0EU+oDrCee9uHkH+Szfuq2HlipVwfYETztHIlrzwUC9lhifoClJWvnzy3KIshafb/19no4jetobbhIKXuwNNpsNkV4tgtI68+28WLD1jLa4W20tlVzsPJ2bgXY7VRTfUkBKSAFpECZAgLsMlXeyotgYMARvZNA0JZrf6uAQYOiCCzdAthxftHuMsnqAJsQgq+vXdpS/K5pzBj5Q47Z6/Ds3eN+XGCVuhYiAgT6L1zxddm3i56YX3bsYY/yKsCO5wd1zRPtL7jIb5Zs3q2O7fvzbSK81j0siXaMa+Afz1ezKc4zjmG2RMAeTPu6OVj/7doKsNulpPoZLQVee+219Nhjj6Xnn38+/5k8eXI677zz0qxZs0bLhJ4YB/0OHjyYjj/++HT00Ue3bU779u1L/LAu/PRSMs1mzjz88H4vzW8k5yLArlE3goUBQ/To4eG1NyJ4714ElgiqzaCirnwkQ0T821CQp5UwhTrA9rDJg3Zzl16bjs3gDsCLGpnGtixo/eS2jenZ7IFBe7jUynzssuVxcfBUVp9XFcYU+47ldux1J68KsCNQ8irAhVd9Ne8mgi3nxbjxE2yIAdtzM03QI45twD6ggcvwbcrglwuch7NvaPzR498acLHi705UrQXtfVhU2RiYE/WwC834d1TWvtkc3HSHvdvrgM0H/b333tugEx+OgrEGSbrmAMBZs2ZNArJ9Wrx4ceJHqTUFtm7dmjZv3pxXBq7Xrl3bFhiO63P55ZenefPmtWZUh9e66aab0n333ZdbecIJJ6Rrr722rRcmHT79YZsnwK6RMIKBBzQPjb4L7ymNwFIH2MAVt9QtNYOakQTsGBMLyC5dvbXwdJqNflsH2B6eogYRRL3Gvn/2WQ/qe3iuqm9QaRc+tI9jk1eWvL2UR8AG+h+47YsNsePU87bEtY8x2NQvSxFS/flUVp88b28ZvPp2XIBs37KmAbRtftFmP58YCmPwbH3Hi04upC676R/y4vh3VGbjYOZgYw512+uAvWnTpnT33Yf/l6DT6aefnlauXDlUydRujBQAqq+88soBcI05YwnY2DUcD/Bw2w9lOTj/X3jhhaJpu/S74447Ej+WuJhdvXq1HXbFtmo9Lrroogb7AWxdqDdIUnsgwK6RJ4KBB44YRmHdePCIwBIBz0MF7f0bKHjfNuNbilAyWMA2wAOELCQgwpyBFmP6t3JwTNzu/Oz1afaAHAD7w+xLY+zLW1oFbO8xBeR5dzU2WTKN6Z+vJo8PNUYgtzhftJ6dvW3D5kZ/8SIl6m9jxm1clzM+fuiVi9R7NXu14C48wc5m8v28OC67SPnEqi2FftQhcR75N6vE84r5MP4p2ZfqWKKO3QEgz9vrzxPOHx7KtTWz9j7O24NwPF9tLWgXY9x5YPL87HzAPrTgfeXc+bDk9Yh/R95Gq181Bytv57bXAfuKK67Ib1dHzfBG9drt6zjHXjvmTsTNN99cTIvQBqBn/Pjxed5owQ53RXbu3Jl27NiRb1esWDFo0Hr66afzMBe22D/aEOq9sYjXLlh89NFH07p164o1WrBgQVq2bFlx3Kk7eN5ZU+xnTW6//fYBpvqLEtaMeep/yACZKjME2JXSlL8H24APqIgP4UVwiMASAS9CS40pDV8QQ71mgB1B1Pr2NtYBdgwTsfYGVBz7vuoAO8I63nr68RcQ1r9BnUEZ3vOpGdxPzN5pvS8Dcg9x/m0UABp9sj6Ts/55z/MPM+A7kH17oiWDcTuu2nrYq6rj8z1M+vyyuxx2/lAPCOc88hc25HsA5tiS1950oszb69fEzj/0njr9g5k+E3JNGNeSt93qW5kfg4sV/5yB1SnbAu0Xr3swXw/KbS2trrfR8qrmYOXt3LYC2K//7NCIR72jnSOPfF/xw96PePHFF6eFCxf6LO13uALxbgTAw92I0U7RSzsUOPXe0LHy8hImQgx2u0Om+LsDWIHPbgkP4QIHwLZUBth4tu1uGPMSXJtarW0F2DU6RTDwwEGzCMjmgbYuI7BEwK6DFkID/DcARihpBtjRe2s2+X7qAJv60ZtqfdjW91UH2FXfZgiIvSd7y4cP4zCNo/Y2pm1pyxfHAI8kD2hWx2+j9r4s7jfry+rbu8Hx5pYl4LnuwUtrEwGbtfu7Gy5rCIWxurY1nTj29vo1ieeftbUtceH+3dyxvh+DNvkdh+s+2RBeYn3ZNq4L+XEtvY3WrmoOVt7ObTPABq7X3fvOfMhr5v9rO4ce8b68lw6IsIevGJgYSuJOlbpHgQhBQwHbdsy2VwC7HVr0Sh/x3CoD7F6Z61jNQ4BdozxAcX8Wr2rJHkSzY8DhkW/8mR3mD7jhZbQEoD774OHYrOlnL24IB6AeY9AHffG2DA9tALClScfNKr6YgzxCK3wChmKiT+DWXh9IuX9gMY9nfvHpolnZNx0Ce2UPyPFAH696M7gEsP/x0a1FX++d85H0H+f9bnGMFv7hQ9qf8bHPppey+XuNTGMb98fZHPzDjUAhXuBTs3AQH/rAXJivj89Gy0nHzcjDWLznuDCqYsfrHquwvqzF1JPPHBC+EuvasZ0HvKLQktn2nlM/POCcoA5wzsOdeOzjnN6TAT1f8GPnmrfXnye2/nZu2dgALuvG+lkflJmdVs/Wwo7ZVtnFutBnDNOhTfw78jZSTqqaw6HS9v6uA2yD6z3/8rZ01JEprZj3r2nau37RXgNGqDe8TT5el1vV5NlDSgzbSpgInjg8cgbneK2AczynZR4s6pkHj33q8OM9XoQ6UGYpPpzHLWrvTfNty8q4Xc28sNW/TYP50g/51h+2Yw/1msUN2zxob57OY489Nn3gAx/Ij+PDo3EezK/ZXE2Duq31YVur63XxY9t82fo1YM2Yf1VivqYXddCV+jaOaU8d9i1ZOcfeDiv3W4tPti1ldn6w773Jvg5l9M18TAeOaUuyOVPOvtnu+8srvvXL5mJ5fg5VZYyLRnYulOlp9lm/fvxYZvNmPM5fyskj1IfzsyoxPzvfqYMd1CffznPy/Zw4LkumpW2tjq2j2Uh+VZ2yMhubNsS687fIueT/7piv/Y3xt8hY/J+qS9Tnh7Yk9LWxytrZGGjMupGww2wpazNSeQLskVJW/UoBKdCRClQBdjfDNULzwebjdfFW82Hj40P5ELUP0rg4fCCuX7++AaR8ndiW+ozHh19Z4oMQLxmpmbeszkNaVkbohD2wZnYBG2Vv28gNyH7xgV4VO8yHMXOxD3FrY1vzHBNbax/alKExH9w++TrETHv9fb26/ahXWV3zONJ/1RrQjrAgwoN8Qivmy7YsmaZR+7K6ZkdZGXk+NKSsjo1VVpeQGK+frYO/U1PWJwDK2zz8BVWci/VF+7Iy8jy82jjxLSGcO5x3lvx8Yhl/E9hmbzOxNmyBRvqOKYYJWTnnHf1ZCAf5fk5WL26bnVut/t3GflgrNIvnFGvA38lLL72U/3/h/4ZPXFxgd0z008o56tvF/4G+jP1m52qsP9xjAfZwFVR7KSAFukqBMsDe/crb0vosLOT1n6au81yb+B608AwBISQPfD7f2rHlQ2/VqlWVgEkdDw7UByrihyn1LLX6QU39MsCxB/hiGR4x75U3uyLMmB1+y4f9jTfe2ABewKmHOF/f9g1cIuzY2FYv2hBhzOo120Z4KatvsNBK3WiHf3itrG+bV9S+rK7ZUVZG3nAAm/PVX/TYOrQyZ5uD2RXnYn1RHsuA16pzO55Dcc39uLGMthEwzT623iaO/asFOW6WYvuy+s20a/XvNvZTNzf0ZB2r5h7PT+ry/8jX5w6Fv7hlbr5d1Lps7s3O1bI2w8kTYA9HPbWVAlKg6xSIgO3h+leOfjNdcfYbXRMWYuLzgcTbQyz5NxlEb1+Z1zXW4cMMjxreNj7U+PACduwhybr62GC3dAE5Uvwwjh90EXA8KMSyvMPsFyBAwkbswkY8XgAOeXzgk6JXyz/syQe4D6uhPvNEP4MC2tMnwB91pg56WooAfuuttxZ2WJ1WtozJWLa1NlxcELJCwiYSFwfYwZzZkmiL5mxJ/sIKcAReLNHO1ok884aTj6YWHuK9uWV2WH9xy/qRbMs+9nB+kVhHu5gqg3Hq8sNacbFIXeZGHutuc6acfLuzQd/+PIvnUbNzjHMADdArepx92wh2dYBt86UOf1f8HXlo9G3Lzk3swS7aMDc/V/r2dnFcluycsq3VsfPJrw16+nX3esYy+uFvi/XAq27nkfXP/xTWj/5ZC9+vh3rqx745P5l7XGP6QkOSdzBw7P/P2d+SzZHy0UgC7NFQWWNIASnQMQp4wPZwPeXdv0irspjrbntzCMJGT5f/oOWDFPC05OGbPD58PJyT5z+cOPaJD7lLLrnEZ9XWp2L8wPQf1JQPFn7K7MMug2r69IkPdgMZ4NDmG3UjpANbq/qhzzgXPuD5oCf5VyT6cfLCIfyKY/l1bdZd1NQ0j0AIuOAJrJtz7GswdpidHp4jUJXVIa/Ktqq1juvpz5O6OcQyxvUXHfGC0l+kRT09JMcyIJO+TOv4t+l1qStDGzTgfLPzmrzBrEs8t+z8oB9LdXVimZ93mW3e20y5/x+CHlyMkuL/o/j/Kt5xsjWO9hiU21zGYivAHgvVNaYUkAJjpoAB9hM/fnva9OiReVhIN8M1QvJhYt4sPsTxpFqq+zCjTvwgjx9o1o9t4wdcKyAZP/zih3kEHA8KsaxuPOaKZ4wfPI/EfZoX1+z3EBO9Xq18KMf5G2wxnvcMt9KX2VS1jbp5XXwbxsYu8woCdjEZiESAoR6AA1SirXmTffu4BlV2+DZxfyiA7S9eYn/MmfkyVyCzbM7ezro5xDJbUxsz/o14mIwQXVfmzz36jn+bvjzaVHY+xfPXz9dsr9rGcyv+TdKurk4si2PHcjv/zB7/P4s8Gz9qzZ0Ou4ClHucvdSyZLvHuEeV402nPeW0XNdZuNLYC7NFQWWNIASnQMQoA2OngS2nTjiNzm47M3nU9NXtTyNuOGBsTL53z03TM+KG/qSSCXSuzsA8l6sYPcg8IZX0Ntj59xA9b+zC1/mOf/sM6lkX4sT4AHTz1Eait3LYeYqJddUBn7dl6T7V5O7331N+69u0Gux/t87rQF4DGnIHrZsm39bbGdgA2D4N6IIlr4PuK7auOhwLY8Tyxvstgysr81ttZN4e6MvobDET7v5/Yzp97ZmeVLs3WnvbN7LYxyrax/zKt6+rEMq8148Xy2H9VeZxTme0+z/Tmb4E+zdHg63AuUw/nwWgmAfZoqq2xpIAUGHMFpvzyEelvdrySe67H3JjMgC8t/NdhAXarsOHn6r3A8QPNPrB8fb8/2Pq0rfowtX5jn/7Duq7M2vPh6mOpAVw8VwANyb/lwUNOtCt62az/uI02cXubvuzDvdldgNhf1XG0z+tCm7j2rOucOXPyV9YB3XVvmAD+mId5vb0NQDZjWYrzjXZYvbptFUj6Nr4O+RHKyIu2mNedNefiyodDeTtju1bLGDOCsv8bGUyZP/fol+Tn7MubrT1tu8mDHdcyzs/K4zoRtuUv9pi3T/yd80Pi/wDnPBeQPnQmL8x+eceC5Y3kVoA9kuqqbykgBTpOATzY0478p+KtIUePS2nWr/5beucYfWvjb733Z8MCbO9NbVVsPrDsbRrxlmyEq9hnDJFoVp/28cM0gmwsHwz80H+0KXqiqyAmAoqPE6XfqhTDLPCq+wfh4vhV/TTLr9OFtj62PEJ9BBWvqR/XbrlHKPFr1Gpfvt+4X7UGvp6vQ75Bl6/jQwv8hSJ16uwcahn9Dgai6+DbAzT9kvycfXm8eCo7N+PfftUaHxqp8Xc8t8q0rqsTy+LYsTz2X1Ue/x9V3bFqnM3AI/4n0BdbS3a3yY5HeivAHmmF1b8UkAIdpYDFYPsHHLvtS2VM0AiWde9d9mBCe/vAjrBIGeDmb6fiGaIeMY3s+weUqurjGeUDjRThxr+buSzExX9Yx7a+LO+8pH//9o4YDuEhJn6Yc+ER35XNvEk+DpRjD+fowjxIZWvwyiuv5GX8evPNN9O73/3u4rhuJ0JInLuHMw/Y0aPPGNaWMlL0CtbpHMs8ROadtfDL2+ov8HxTX4f8CGXk+ToesJlXfNWkzZl2cQ6tltF2LAA7npucf5wPdh7G+WCnnxPHdSmeW/6CytrFOn49YlkcO5b7tvRfVR7/HzFfbIvnK/V43sTyWSMu9mPy54v/24/1RuJYgD0SqqpPKSAFOlYBA2wM3H/wbemmB49Me7vwmxuxH0+pfyd0nbcngqb35kRvGX0DjfbhBTwCcIAVqezDnQ9C2nBrlvr2Rg7qR1ggz/rngzEm/2Edx/Jl1i5eaNA38+NDmLF9ih+y0QtIXf9BjX2tjGlj2IWLHf/gBz9I/+W//Bc7TFdffXV63/veVxzX7UQIiXZ4Dzb9cOECdDBnuzCw/q0t8+ELhdCHebJurBc6G3zHB2Xpz4de0CdtqQ/8tJKizoxr35KJ3SQPQxxHKCMvXigSHkBfnAPMwyebM3l151FdGW3RzIcZ+QuMwZTFc4++/Zx9OdqimQ914O+Rc7vs4V368vPluC75C0TqWd+TJk3KxyUvnn9+PWJZHDuW+7bN+o7/17CNdea8JNnDrX5MxuOcpx4akTgnOHct+YtQyxvJrQB7JNVV31JACnScAh6wMY5vcFybfclMt0E2H8DRk1wXmsCHDx/YPnlPb/xA9PXY91DBcfyAJs8nDwvkR8Cyunxo8qFYFS/cDH7opwxGrH9A32KjyYt2AWXM3YOMtbWt/yC3PLZlc/KaUudb3/pW2rBhA7vpqKOOagglyTNrfsU1iXZEbXxXzNPHV1vbCIS+je3Hi4Q6fSM4WR9xW2WrP688bNK+rO94oejHiWttc6ZOHL/VMtpGzbzNgymL5x59+znH8njhSH1L/N0Anv5Cys/J6lVto91Wz9sQzz+/HrEsjh3LfVvGqivnfKPc/92afX7rx4z9+XrscxFW5gmP9dp5LMBup5rqSwpIgY5XIAI2BnvI5nhZ9maPs47/Obsdm6JXsSw0IRofvX8RpICQGItLH3yYA5MW8mH9Uh87/Ie81ccrCYhYog4fcP5Dkw9zvLCABH1Z8h+cdWBk9dkCykC/t4X+mbO/EPEAYe35QMeL7+8GWBm6EnrAB3RM0TYfsmB1//Iv/zJ997vfzQ8/+MEPps9+9rNW1HQbocHrQmPsxgZ/cYKdaIoeZZqW6WSGMFeAL64z5WXtOC/QrdVEXW8r7fxdFw+blEUoI48UdccO5szal825rI3XMvbny2gbYXS0AJux+dtAt3heM1/y/UVU3QU2fcXE3zpz9xeXrD1/M6R4/vn1iGVRs1ju2zbrm3LO7aoHFllv7OR8sb9LtOB/kZ8L/Vhd9OKCZDSTAHs01e6DsW5YfkJLs7xq4/Mt1VMlKdBuBcoAmzGA7I2PjktP7f2lfMhugOx2a2P9AVN8wJHs1rGVlW0HU59+qd9Kv2VjNcsDRLiFTviBffg2a+PLgSlL3Gqu+1COYAaYRDgFzpkv6Q/+4A/Sb/7mb+b77f5ldvvwlroxbB2sTrO5Wj3Tl+NWx7K2tjVbh7pGsZ+h2mH9dMOWc4g183838Q5KhNhW51XWd6ttR6Oe2cdYzc5TX9drNRp2xjEE2FERHQ9LAQF2Srsevyft/cEj6ZipM9LMuYuGpacat1+BKsC2kW59bFz63o8OQfbn5r2RTp78b1akrRQoFAB24qsB8SD69LOf/Sz9zu/8TpH1F3/xF+mYY44pjrUjBYaqQAyV8Z7nofapdu1VQIDdXj37vrd+Buz9L+5MW1YffpH9nI99Jp2R/Sh1lgLNABtrgWzSJR94I9/qlxRAAW7X4xXjTSDcvubYUgy3IT8+4DhUD6ONoW1/KcAdEu4YAM94brkjg/f/sccey0O5vBoxRMOXaX9sFBBgj43uPTtqBOx+CgXZ++zD6a61S4q1FWAXUnTUTiuA3VEGy5iOUSDGlZphdd5DvNiW3vGOd9iutlKgqQIxBKmqwWi/HaPKDuU3KiDAbtRDR8NUYLiAjRd43+5n0oGXd6dx4yemSdNmpSknn1Fq1RsHX037s7qWJh4zJU2YNDUd2L8nvfry3jTuqAlp0nGHvsnN6rC18r3PfS/r+4PJ2vk6cR943r9nZ2LMCcdMSxOzcaJdAuyoWmceC7A7c126waoywLYHKetitbthbrKx8xRoBtg8wMcDl/aaw86bQX9bJMDu7/Vv++yHCtg7t9+ZdnzzzzP43T3ApgmTpqVTL7gknXL+8oayMqCdcMzUtP22L+QgDABf+PmvFW0A620bVyTaxTRj7uJ0zpKrc6j3Zdhl/fl89q3/aEesx/GFq25LU6afWVakvFFWQIA9yoL30HC8pYAfErfrebiOVwwqSYGRUIDwEEJCCEUi5t+SnXvcOdGFnanSeVsBduetSVdbNBTA3rZhRXr2wTubzhsIPn/5uqJeBNvpZy9q6McAmAZ4xgnfwANdlfB2L119d1H85LYN6YEtXyyO4471H+2I9TgWYJepMjZ5Auyx0V2jSgEpIAX6SQEBdj+t9ijMNQJ2mdd2+lmLirdr4CH+TuZV9unEU+dnoR2zMih+Ou16YpsvSguvvCWdeNoFeV4zsDUApvKWaxdk4SQ783aEjsxdem16VxbmsW/30w0Q7eOmN6+cm4Wa7CnanH/plwsPN+32vfhMmn/p+hze79+yJoP3A8UYNJqYedMnTJqStz83G68sXCUv1K9RVUCAPapyazApIAWkQF8qIMDuy2UfuUlHwC4bqQpiqXvO0qvT7PmXFs2iF9lDcxlg48U+d+k1OQgTEkJMdoT4OMbWGz5VgLz3Yvu5+PzCuLAT7fHzDFV1OIYKCLDHUHwNLQWkgBToEwUE2H2y0KM1TQ+lVWMaeALAm1bNLarh8b143fbi2HZin/Zmkgi0eKYvXvdg4WW29jEEZeFVX22o80wWnvLM9sPfImf933LF+9Ibrx+wbvIY6hlnL0542HkAM6Zoj80z1tPx2CogwB5b/TW6FJACUqAfFBBg98Mqj+IcIwwDmTFNPfnMHFYjkHrvtG9z13WfTHufe6TIWrZ2e+6Zju0BX+A5ptg+lsdjA+xHvnF92pH9lKXZF1ya5nz0DxtAO9ojwC5TbuzzBNhjvwayQApIASnQ6woIsHt9hUd5fhGwDVbLzIhA2iogX3bj93Owje2rgHaogI3NhKjs+NvrGzzZNpcYNtKqPdZe27FRQIA9NrprVCkgBaRAPykgwO6n1R6FuQ4GsGOISARWM7eqz1aBNgI2D0qWhXjYeGUPZvL15z984p6Gt5RQv+6hyyrgt3G0HRsFBNhjo7tGlQJSQAr0kwIC7H5a7VGYaxUMVw3t39RBnfg6u/iQo/dytwrYMdQjPuRYZVtZPqC99cbLiiIP0dEeHrjkLSNKnaWAALuz1kPWSAEpIAV6UQEBdi+u6hjOabCAHeEXzzLvu56Yvd7u1f1705P3bGiYjQfwCLQedn2j6ClnjNkXLM+/xdHq8a2OO7d/PS1b+4BlJeB/9oeXpxNnX5DHfPMObb4Mx9tU58H2c5ny3jP1mr5C2bHdEWCPrf4aXQpIASnQDwoIsPthlUdxjoMFbEyLIRxV5p4yf3n+Cj4rbxWwqR9B3vqIWx8zHucS606aln0xzZrDX0xDefTIWxt/YWB52o6NAgLssdFdo0oBKSAF+kkBAXY/rfYozDVCqQfWuuHv3/KF9NS2jaVVeP3enI9/puH92FQcDGBTv+6BRcojMMe5UMcSbzyJr/ujjBCSb2/43ICHIgXYptzYbwXYY78GskAKSAEp0OsKCLB7fYVHeX5Ar09lDwz6cr9PKMfurP2Bt749kbLJmZeYPsoeSiRkg69AtzQx+1IZvlimLtEGG/e99a2O1B03fkI66dQPD2hL3R9mwOztoe7Ukz9YG+5Bu2cevCP/Zkf6n5C93/uk7Nsny+ZAudLoKiDAHl29NZoUkAJSoB8VEGD346przlKgjxUQYPfx4mvqUkAKSIFRUkCAPUpCaxgpIAU6QwEBdmesg6yQAlJACvSyAgLsXl5dzU0KSIEBCgiwB0iiDCkgBaSAFGizAgLsNguq7qSAFOhsBQTYnb0+sk4KSAEp0AsKCLB7YRU1BykgBVpWQIDdslSqKAWkgBSQAkNUQIA9ROHUTApIge5UQIDdnesmq6WAFJAC3aSAALubVku2SgEpMGwFBNjDllAdSAEpIAWkQBMFBNhNBFKxFJACvaWAALu31lOzkQJSQAp0ogIC7E5cFdkkBaTAiCkgwB4xadWxFJACUkAKvKVAVwP2d//nr2ohpYAUkAKDUuCdbz8inTX5x4Nqo8pSQApIASkgBQajQFcD9mAmqrpSQApIASkgBaSAFJACUmA0FBBgj4bKGkMKSAEpIAWkgBSQAlKgbxQQYPfNUmuiUkAKSAEpIAWkgBSQAqOhgAB7NFTWGFJACkgBKSAFpIAUkAJ9o4AAu2+WWhOVAlJACkgBKSAFpIAUGA0FBNijobLGkAJSQApIASkgBaSAFOgbBQTYfbPUmqgUkAJSQApIASkgBaTAaCggwB4NlTWGFJACUkAKSAEpIAWkQN8oIMDum6XWRKWAFJACUkAKSAEpIAVGQwEB9miorDGkgBSQAlJACkgBKSAF+kYBAXbfLLUmKgWkgBSQAlJACkgBKTAaCnQ1YO95Y9JoaKQxpIAU6DEFpo7b32Mz0nSkgBSQAlKgkxToasC+Z/evpX0HftFJesoWKSAFOlyB90x+Wzpr8o873EqZJwWkgBSQAt2sgAC7m1dPtksBKTBoBQTYg5ZMDaSAFJACUmCQCgiwBymYqksBKdDdCgiwu3v9ZL0UkAJSoBsUEGB3wyrJRikgBdqmgAC7bVKqIykgBaSAFKhQQIBdIYyypYAU6E0FBNi9ua6alRSQAlKgkxQQYHfSasgWKSAFRlwBAfaIS6wBpIAUkAJ9r4AAu+9PAQkgBfpLAQF2f623ZisFpIAUGAsFBNhjobrGlAJSYMwUEGCPmfQaWApIASnQNwoIsPtmqTVRKSAFUECArfNACkgBKSAFRloBAfZIK6z+pYAU6CgFBNgdtRwyRgpIASnQkwoIsHtyWTUpKSAFqhQQYFcpo3wpIAWkgBRolwIC7HYpqX6kgBToCgUE2F2xTGNq5OrVq9MRRxyR23D88cenZcuWjak9nTb4pk2b0gsvvFCYde211xb72pECUuCQAgJsnQnpwP496dWX96aJx0xJEyZNlSIdosCjf3dD+h///f8trJn7yT9O/2HOR/Lj5x7+Znroji8VZe//0O+l/2PB5cWxdqoV6DXAjrBTNvOZM2emWbNmJbZKzRW46KKLikpoBnArHVYAPXbu3Flk3H777cW+dqSAFDikgAC7T8+EndvvTM8+dGfa++zDDQqMGz8xTZl+Zpp9/vJ821CogyErgN7f2biiof0p85enc5de05DnDwDsh+/6cpH1n/7gxgbAvueW/1yUnbVolQC7UKN+p9cAO8JO3ewnT56ce2NPP/30ump9XybATun5559Pjz76aFq8ePGA8yGecwLsARIpQwokAXafnQRvHHw13bV2Sdr/4mHvQ5UEM+YuTucvX1dVrPxBKLD1hk+lXU9sa2jBxcxlN36/Ic8fCLC9Gu3b72fANhXnzZuXLr9cdzxMj7jtZ8C+6aab0mOPPZZee+21XJYyeBZgxzNGx1JgoAIC7IGa9GwOcL151dzEttUkyG5Vqep66H3Lle8vrbDwylvSiaddUFomwC6VZdiZvQ7YF198cTrhhBPSvn378p+nn3664Xa+Cbhy5cokT7ap0bjtZ8D2c0cVAXbjuaEjKdCqAgLsVpXqgXpbrl2Q9u9u9FwTpnDSqYcA75UsFvupbRsH1PnQ8vVp5txFPaDA2EzhyW0b0gNbvpgPPu6oCfn2jdcP5NvpZy9K8y9dX2qYALtUlmFn9jpg88AZ8dY+Adnr1q1LBw8eLLIJF8FbWZeAdBJ1h5IIMyAB/K0mPKeMO5g2vu+htGc8P0cPmc1isNH26KOPHrK93nb64uFKHqykz6Ek5s86+/kMph8/d9oNFrDRcvz48YOyn/OE+Q7V5sHMT3WlwGgpIMAeLaXHeJyyGOClq+9Ok44b+NDTXdd9Mu197pHC4gmTpqVlax8ojut28pju7ANi0rQZiRCIVpK16YSHLPE279/9TG42MFymTytz8nX8hc2Jp87P7yCYvmh08drtpVoJsL2K7dvvR8BGPeBtzZo1DUJWwfjdd9+dx9/6yoSVEI8bIagMyO64445EHxZmADwtWLCgNJ6XMah35513pnvvvbdoQ/4JGZjjkY8XDPTPjyV7i8XmzZvz2GHLX7hwYd7ejv0WPRiTrSU8+oTOXHLJJZaVPxhKSIRP2Mn87ALCytCGOwPY3Wqq6wvN0S3CdgzRYP5ANQ+82kUR47NeZTHUZbbRJ3Dv9aCeae/fphLHB8K3bt2a62lrjha8faXqLgnjxPOMeVK/7Dwrs1l5UqCTFRBgd/LqtNG2GAM852OfSWdkP2WJt4psykJJfPKhDI984/q0I/uxdOGq23JovP+2L2ZvJNlt2Xnow/mZ97sMtBnjkW/+eXpm++EPSRoCtGd89DOVYRNF52HHe4kpKnuAMF5k+DqU7cjs8fbbEMPx4Ect6eunr79SeLQZo6p/AbatQHu3/QrYqAj8+derRQAD9m6++eZKwQEgYM4DZATs8847L913332lfZTFfgOpeNc9GMbGQC9tLUXAZh4euK0eW+y54oorfFaqmydz8+AcPdh4/avmxyBlFy0Ng7sD5s2DhHUJe+jTQ3YEXCAcWC1LcY3L6pAX1zHW8zrE8ev0L9OjTn/GLTvPoj06lgKdroAAu9NXqE323bD8hIaelmVe07pX8kUvtgfyCNiUeeD2A5WFQPCAJQ9a1sWCV0Gn79vvR5At87rHiwzz4N+/5Qt5aIzvz+/7ufv8VvZj3zzU+NPMS+4vYPBqL7zqqwO6E2APkKQtGf0M2BFMATO8jCQ8it7DjQcSqMUjCRBZeAkeTaDJUhWYAWT+VW5W3wMXfTOmB1rz2O7YsaPhYgCwNe95nIf1zZj0ZbZavm8LyK9atarBU05Ig4F1bOvBskojxmEuwDIXA+b1tfHLtnibIxQzP2zxF0G0jZpHwLX+yzQHVm+99VarUrmtWkdr4HWoGh8vdzPbqzRsdp6ZHdpKgW5RQIDdLSs1DDsJwQBofbpq4/P+cMB+hGgPgbGMxhZOcWD/3uyd2nsa+vMwD1T7By1pNz17W8nESVMyb/bXG+K/fbuGDisOfCgGVWL7W654X7LY54nHTE0Xr9uevwPcwy72zPn4Z7J3gk9L+7J49V2P35N706u8/RWmFNmbV84t9Jg0bWZauuaQlynaCnhHT78Au5CxrTsC7MN3jaqgCcgDSs1rGj2Oa9euzYGUhYlgRp94ymkLNOER915aQgAoJxFWQFiHJfItpIC2eJ4NeP3FQARswBToY0s7ANp7xD3URw90tJd+PCR6jeK4vl+bA+ObbpYXt9gWvep+7lwkRK++HysCLmBLHuPSln3TjbH9ekVb4nFcz1ZisOP46O+T7wPb7MJrMOeZ70/7UqAbFBBgd8MqDdPGCNhTTj4jXfj5r9X2GiHat4llwOqSDBwBRAD6tuxhSg/ZhJDwbm1SbOvLaOvftjFYz3EMATln6dVp9vxL83GjBlYW88vGxK4Iv3mnTX7Fvm1MmsWQljKPvQC7icBDLBZgHwZsC58ACn3csYdZZI7lPuwgApn3FtM2wiQQTB2Shy2fnxeG8jrQJU6beGtL0TvsQ0wAWw/f0V4A1QOiHzdeEOD1Rgu7KLDxm21jP7YOvl2s49fE60YbD+ccx4sID+eU16W4nh6OrV0cP/aPPf4ixfqI54KfE33H8yyuq42vrRToBgUE2N2wSsO0MYIecc6ER9SlCMJ1gB2hNLb1YOk9t96ja7b40BQ/5mNbD30gWz2/nXH2Renod0/O4d4Duve6x1AN825HbQDp2RcsTzPOWlQaQlNnx/t+4/caQHzbhhXp2QfvLEy1McmIIS1layLALqRr604/A3YELwPleNu+meDWjnoeyPBIArcx+TqUGXDF/NjOH3sAb+ZJjuVV9vo+/VjeLg/YACLw6L3DtMNzDGQDhM2819SP9kVApU5cE29HM8BtpX/GKEt+7pTbWvm6cfxYp6o8zsn3Wbbv162sXHlSoJMVEGB38uq0ybYIkXTbLEQkAql/IDACtPdC03f0JHsAj7Hg1K9KHrAf/psvp0f/2w2lVZf/P4/mgE1hjLO2eXqw9+BNG1/GsSW87jxwad538rd/7U/SE98aGC9NmQ/zGBAK8xa4U8/SI397+EFR8jyAcyzARoX2p34FbLyDV155Ze4lNFXN8zkc8PFA5iHQxmDr63BsQBbzKatL1q4ZQMZyA7XoIW3F3liHcBcuVCJkYzdwDSzj2a5L8eFGAXa5WrZu5aXKlQKdrYAAu7PXp23WRbD1bwUpG8THDlPuQxiaAXYE+tEE7Aj3wP/ESVMbHir0c2FueJO3bfhcw6sJybfk67cK2NEO66tu6z391BNg16k19LJ+Bezovfbe5gjYQGXdg3q+3EMygBkfqIthAd5rHNsSMlCXAC5SBOgIqLHcg1ocs5m9EbAZH1AHtO2HPEvxgUTL99ton13o+DpxTXzsevQQN5t/LPfjxH2vD2V2UePrxfFjnaryOCd/Hvn+bb9ZudXTVgp0ogIC7E5clRGwKXp28coCn2WJB/u23nhZQ5H3rrYLsAkROXfpNQ3j+APCqKhftgAAQABJREFUNew91Pte+B++qGF//Lv+feHBpsA/zIjnnXG+s3FF3oaHGC+76R8a2tsBFwY7s5COHz3+reJhSMqww95VXWfHxMnHFSEiUW8bo24b33wiwK5Ta+hl/QbYAC5hG/5BQ9TzcckRgluBRFuBCGQR5iJM1oEisNuOEIs4pgds3privc/N7C0DbJs7W7QDKNlaisBp+baND42WxWDHCyIfjxwBttkcYrnZUbaN+pTNJY4f61SVD+c8K7NVeVKgkxUQYHfy6rTRtuhVpmu+Bv2cJVcXUEge9YBrQhwsxVftDQewYziGD6uw8Ya79bHPhJkAyLue2JZ3G+dSNhZz/3bm0bY21IlhMGXtLC/GVwP1p1xw6GFLq2PbZ7P3b/sHQu3VgZQLsE2l9m57HbAJTzBIxdPKQ3sxlQFdhCIeGly0aFFDX7z/Gc+3fyd1BGw81EAa9XjQLcZke29tfJAP2wF/H2LBhQFzMO81c4kAHQEylnvAjuBq9qIZ40R7PWADxnwZywc+8IFCF+xhTvZQn78zQFlZYl38G1Kog96sC4m3bHg76BO7bV3jWjWbfyzPB6n4FftGOy6KsNnuasQ6rQI2Q8a28TwDwtGZdfHnWYW5ypYCHauAALtjl6b9hnnwtN7NS8z21SxUgndU++TfEGL5wwHs2BYP9blLrininIHTPdm3SO559nuVXyFudlRtoweeORjEeoClPRcUu57clmaff0nDQ43RAz2YC4H4hpA6qI9r4mPdBdhVKzy8/F4H7GbqAHEAsMGa1Qcu/dszLD9uPaxSFgE71vfHHlbJB9oALoNTX9fvx3YRoCNAxnJvc7N5ArPew+3H9v0abNIf87AU34xh+XEbvdix3B/7uw3kR0htNv9Y7vuO+/ECxMq9DnH8wQB2DBOx/uPWr1ks07EU6AYFBNjdsEpttDGCY13XgOnCq24pwjSsboTk6N2N3nIfg413+K7rljS879r69Vv/gKPPb3Xfh4lYG+bDu6998rbaxQZ5Pg3Wluilr4t3jxcDPkxEgO1XoX37rQD2rY+Nywe85ANvtG/gEeopwk7VMAAS0GJgWFYPbzGA5QEz1ovgEwEbgC/7pkP/rmTfJx5L3tNcB9ke7mjrQZfjCJCxPNpcB7fArP82Sz927JexfcLTS/t48eLr+H3swFNdpTewz8VQ9OTGNW82/1jubYj7Zd516ngd4viDAWz6ajZv6sQ1I09JCnSTAgLsblqtNtkKID91z4aGOOPYNV5X4qOBzpiGA9j0hZf67264rBay45s+og3NjuNbUKjvQd/ae8C2PL8FrvmWxTIdfD3b5w7AltWND2rZm0ysTtzGiwG7YBFgR6Xac9wMsIHr7/3ol/LB/nThv6ZJ43/RnoFHqBdgxcf/xmG41Q5Us20lAVj0SZiC98zSBwDpwzfoLwI2sEV7fkjAJu0iJOaF7hf18W76uZjtsS317MtK6IJyP79YDhzGCws8z4So2Hi0J1yB+QHSlsi38anLN0xGbWjDHOMY1kfdtkxvNMNmxi2DdbQyu+mbenXzj+V19lCGTWiARpbogx9SHB8Y9qlZOXXL5k1+1XlGmZIU6CYFBNjdtFpttBVP8g+zhxn3PvdwBry7856BSB5+POnUDzeES8RhgdI9WTtLMzMY91+7DkDzsKClqSefWYSAWB5bvLf0s//Fp/PsQx7kDASyhxJPPO0CX3XQ+9EGOoh2kocOzIdvbdybhaVYmjL9g6nKbqtTto3aTMi85jPnLiqrWuTxxpED7tsvbdzH//4rRR12JvzKlPQf5nwkz3vu4W+m1/7lfzaUn/afPt1wrINyBeoA28P1sjk/TWcd//PyTpRbKFAG2EWhdqSAFJACfaqAALtPF17TlgL9qkAZYL/+s5Ru/N470z/+r7flsgiuWz87BNita6WaUkAK9I8CAuz+WWvNVApIgUyBCNjA9bp735n2/Mvb0lFHpvTJ2fJcD+ZEEWAPRi3VlQJSoF8UEGD3y0prnlJACuQKeMCOcL1i3r+mae/q7JjrTltGAXanrYjskQJSoBMUEGB3wirIBikgBUZNAQPs3a+8Ld2648jCcy24HtoSCLCHpptaSQEp0NsKCLB7e301OykgBYICAPa0I/8prc/CQl7/aUpvz14YMueEf0sTx70Zao7O4Xnv+Xk6psPfVFKnBG/s8Gkob9Lw7bUvBaSAFOgFBQTYvbCKmoMUkAItKzDll49If7PjlRyuW240ghW/lL0KsJsBewSlUddSQApIga5VQIDdtUsnw6WAFBiKAniw08GX0qYsPIQ0PttMfOeb6W1HjI0H+6q5PxVgD2Uh1UYKSAEp0MEKCLA7eHFkmhSQAu1XwGKwH3rh7QVkT333L9LK7AHHo97R/vHUoxSQAlJACvSfAgLs/ltzzVgK9LUCBtiI8Ny+X0o3PzQuDxcRZPf1aaHJSwEpIAXaqoAAu61yqjMpIAU6XQEP2NjK20TsgUdBdqevXr19TzzxxIAKp5566oC8Xst44403Gr46nvl1+rz5qnd++Er4uvToo482fGW7/9r7Vvuo619lUmCkFBBgj5Sy6nfQCvzwiW+ln73+WkO76Wdd2HDcygFff75/9zNp4jFTGr7CvZW2sc6zD93VkDUUexo60MGYKxABG4M8ZPNlM3pl35gv05AMeOihh9L1119ftF2yZEm68MLB/w8pOuiSHQD7937v9wprf/3Xfz396Z/+aXHcaTv33ntvuvnmm3OzJk+enFavXp3Y+vTaa6+lNWvWNMA15YsXL85/WunD96d9KTDaCgiwR1vxDhhv5/Y700/+ec8AS+Z89DMD8nzGUNv5Pur2Aey7b/h0UWXOR/9zOuPjf1Qct7Kz/8Wd6a61SxKQTfrQ8vVp5txFrTQtrbPhj+akg//yUl72zqPflT51w1Ol9ZTZPQqUATbWA9k3PTgu/fNrR+Tf6CjI7p41NUsF2IeU6HTABqh37txpy1ZAc5GR7axcuTK98MILPivfN8BupY8BjZUhBUZRAQH2KIrdKUPddd0n097nHhlgztLVd6dJx1XfrrvlivelN14/MKDdVRufH5A3lIx2APYj37g+7ch+LE05+Yx04ee/ZoeD3gqwBy1ZxzeoAmwM55sd12bvx9771temdwtkP//88zmwcDv9zTffzD2CHb8QI2CgAPuQqJ0O2DfddFO67777ijMAmD799NOLY87jdevWFcfjx4/PIfzoo4/OPd28az32ce211ya9g72QTDsdoIAAuwMWYbRNqALsU+YvT+cuvabUnF2P35O23nhZadloAjZ27P3BI+mH2XbG2Z9IZ3ys0ese7aybU+lkQqYAOwjSA4d1gM30DLLZX9UFbxbhi164lW6JGFW8e/2YBNiHVr3TAZvwD0I88GID1vPmzWs4Xe+4447Ej6XLL798QB3rg4vLOXPmNAC6tdNWCoylAgLssVR/jMb2gD3uqAmFV3rCpGlp2doHSq3atmFFevbBO0vLRguw9z77cB7+YUbMyeA6AjZlQPa+3TvTuPET0uz5l1r1IW0F2EOSraMbNQNsjAeySd3w2j4B9qG14rcA+5AWnQ7Yh1esfC8CtrzT5Topt7MVEGB39vqMiHUesAmh8OEiZWEixDPfcuX7C1tim04D7MLQNuwIsNsgYod10Qpgd5jJpebgwSNGFQ/epk2bijonnHBCWrZsWX7Mg2P8AOFHHHFEnsftdur4VFdub2qw+scff3ziVr0l7MATiR1srX/G8Lf9rf5Qt4zz2GOPpZdeOvRMBP1gB2PYA3KtALa312yJ/Vi+3zI/QhcOHjyY28D8qtqhGXUZi3amNyEMzd6a4ce0fdbHxyyT79+m0epDjmX9oJ/ZZ+P57WDnwryZM+1IrA39Y6+dN1XnFO3QF+82P5Y4n81G06+qD2tjW7Pf7OH8xGNu54zVY2vjWx5jsYaEs7Ctamf1tZUCXgEBtlejT/Y9YOMF9jHLZSEVPNz4nY0rcnWAax4k9LHYHrDjg5AzzlrU8CYPvNB7f/BwobQvr4rBPrB/T3rmoTvTq/v3pme2H75tOGX6mWlq9kOyfqyuDTDlvWcm6pUlPN379+xMezKbSPQ14Zhp6cRT52fe74l5XjPA5uKDC5T9u59O+9AlO6Yf2p9y/vK8j7Jf2LnryXvyse2BzMnHzUqTps3MQ198G8p3PbEtG2NnNsbTedHE7G7D5ONmpOmZvmarb6P9agV6BbCBJR8aUjZjeyAMQAFcSIAF8auWgIpVq1bZ4YBy700ETjzMA0GbN2/O4aPowO0wFvG1BkeuaFC72HD33XeXjgMEWUhMM8DGdvqpSqaXLwes1q9fn1+k+Hzbj222bt2aa2LlcXv77bfHrMpjYBWbDQ5jReurGWDTD2/tYC5lCfBfsWJFAcBWx6+95fmtjU8e5xFx01W2ei907NfKWMd4IeHHY9/GrOrD169b74ULF6aLL77YV8/PIz/+2rVr878x083sbGikAylQoYAAu0KYXs72gA0wk8yLXRYmsvWGT+WAR71zll6dHtjyRXaL5AHb902FC1fd1gC48SFEX14F2DE0pBjY7Vg/sW5ZGAl1tm1cmQ7s3+16OLxrfZFTB9jxjSWHezi0h5YfufIrAx4cjRrEdl7PZmN4W2M/Oi5XoB8BOz4QxrF58MqA8NZbby1gy0MP3k6AmQRc26vWypU+lIvXEjAZKmRH2+NYgwFsP5fYjx0DXcCXpWZtPGADsv7hPOvDbw0QfV7Zfiv6Wl/NADvCaNl4QDbrZGkw4wPVXKQZiFoffuvhNNpjZc20pj+bc1UfNibrwHrUpQULFiQuPi3F8Tm3PHCbnVZfWylQp4AAu06dHi3zEAxgTz97ceGhZso+TATvqQ8PWbZ2e9q0am6DMh4Ifd9UigAY4dKXjwZge298wyTcgbepDrAjzLsuit14wRIfwiwquh3TE+03Z1qbh9tVKXa9rUWmdmoV6BXABmqAIEIm/BsZAGduZZMABMApwpJ/a0OECtr5cu/9tofNGPuKK66gapGAb35IQLt/xVr0mheNmuxEu6nOGMwLcMdrii0G/c082MwVW3goDuBnn/bksyXRLxcYJIDxkksuyff5xbiMRR3KDL5s3lFLD2TYii5Rt6Jzt1OmL6E5Zjd9oY3diWgFsKnPxQDz5gf78fD6cwePLWWkwcwlXqT5iw7mwthoZH1XwTH1qB9DWc4777x07LHH5nbRN6mqD8rox1/80d50jxds/mIzzpm+SKw7evG3wN+TkhRoRQEBdisq9VgdD8EA9sKrvtoA0dPPXpTmX7o+n7UHUsIXlq65O92w/IQGRQwIyfR9cxwBcCiATTjFzuwBy1ezrX/QEtunTP8gw6SZmc0TJk1NEXq9B7sMWPOQjLmfSJOnzUqv5P3fkT04+dnC694MsL+dPfx5ehZmM43wkmx8UpwjXn972NLfDaCuv5ixedqDmxHGY/gO5ZOz1yrauPSn1FyBXgFsm2kMFQEGAAWfIrB5ALrooovyqoR/WBiJefZiO/NsR7gBngxybVyAhvaWPGxaXrNt7CN6l2P7ZoANJAHHMcX52DyjttHLG/uJgOaBNdatO44QWKavb98MsFkHLiZiiutrF1DUi3PxIBr7ifo1W6dYP54bzcoZv64O56Jd4MWLO84Bf9HkbY1zpi15ZdpFDXQsBaICAuyoSB8cewgGUnlPtAc/Ynovu/H7uRI+3760ZbQB25akDp5bqfPktg0N4S02d2tbtq0D7LL6luc18pDvtaduvACx9mwjqM+Yuzidv3ydr6L9ISjQj4CNTB46DMI9QALdQAsJTyNw6EMe8KBa+EMEkQhI9BEByEM937b4zDPPUG1A+spXvpLnRfjz4w9o9FZGM8CmGv3iecbzzxbgwiPsk5+P9+BTB9jCk4xX1Dyy1rYsLAE45o4C21ZTvLCog1v6bAbY1GGeQCdrzny5mGLfJ79GZXOxecS5RI8xfaIN9dEpXtTEc8PrTdtm5XV1IkBjR7TXznP68d7teF77Cw7qKkmBwSggwB6MWj1S10OeQab3VDPNhVfekntxfXgI0A18e3ik7kh7sBmDNFzA9vOmvzq4pZzUDLDxihO/vve5h/MHEHkQM8Z2m8b0d/+WL6Sntm1kt0h8uQ/fNhkfWCT+esvqBUU9dgg5Oem0C9Ls8y+R57pBmdYP+hWw/QNfFgbh84A4u42OmnhxeSDQYMTDVwQRi4v1qxAhybcHsIHhmN75znemv/qrv8qzPfyTASRFL3ls3wywo02xvR174IvhD1aHLR5t/3Ag4Io2difA1wXMAbZWQgzsroK1L9PXytg2A2wulFhrLi7qkl+jwc7FX8D5MTjXuCNC35biOni9qdOsvK5OPG9szKqtXWxSHs9ru5NR1Vb5UqBOAQF2nTo9WuZB08Of/6ZGwhEIn7C3h/BmDUJJSL0C2P7CoGqp6wAb4K97WNL69BoD5HddtyR/I4iV25aLl7lLrmn4ave694/j0T5nydV6i4gJ2OK2XwE7ggceaqAar6Z5hz0kAT2AjsUZ+3CHVkAkQpKHNwD75ZdfLl2xL37xi3l+tHe4gB1BGbACdtkCk7wNxVIEPgAVyLWwA6vHFg8p2lgCYqnrY5utjK3X0ef7/XYCNvb4BxABfTzLbPnxb6Lxa4Q9zebiPet4jrkgQ+eyCwzfdzw3ot7NyrGtqk48b6hbl+oAu9mFTV2/KpMCAuw+PAeqANvDHJ7SydOmF28PsfAQ5OpWwN5y7YIGsDWPfN0pUAXYMZ574jFT0/TMCz315EOvBLxr7ZKiWw/YZNL2ycyL/dQ9Gxped2gNuHtwYualtsTdBbzevKYvJh8vH8t0XK5AvwI2avhwB2JPDSoNfrxHmzyDpfh6vhg+AJjH2/DAl4fMsjrlK3QoF+ite31gWds6D7a/KIiwXgVrcQxsArZNFyuPgEg+wGl1PZib1ta2bOvXifJmUF7nwY5z815ZANrftaiyzeZCX7SxVFWfeRM2wtYSMM85QYo2Rf2aldf1EecU1zo3oOKXP0eoIsCuEErZLSkgwG5Jpt6qVAXY8aE6PKrAIMnDaB1gxxCIGIZRV171FhFTv90hIv6iwcaI2yrAjiE1Xh/68BpFwPZjoDmx4faaRMr83QJf1x6C9O8tp7wVT7zvp9/3ew2wY8ypB5m41h6M8dyZd9oAByCiDsmX+zhVyqI3uCxUAs8otlnyYGd5zbYRNPG8EmbhE0DFnEl1gO29wvYQp/XjPffkmR5mf4whjjHHvn6sG9cnamk2+G28OMFLzhi+bz/vOsD2a0571sFSBFkPzNjtx6NNnIvV97ZY32yrgDWOa/pZ22bl1Kur42PYmQP9o2FM0e4qe2M7HUuBVhQQYLeiUo/VqQJspunDRGzaEfg8PFLHA158MA9PLB5ZEoBITLFBO3kewAcL2HyBDO19qoPw+JBj2XuqsZFkb+aoAuw4Tw/YcRwP2PRvfXu7vaZWnxhs6nKh45O/00C+19/X0365Ar0G2MzSwyPHwAQ/hH74dzpHMKSu905HgKKcFB/2oh4Q40MBgFxAmzJ7ePBQ65TH4ALLg01l9to49IVHGfsBI1IdYEeIRhfaYithBT4Z8JHPl8zgBeUHWGN+wB1jW7JQCeygDjqgP4k58GPJv7XC8uIW8POeZcrpl4seXlfH2NhmHtY6wI4gyjywjTG8XYxhwMx+2Vy4sPJeaTsvGIO+uABi7iTs818O5D3J0SbTO2+Y/WpWTr26OtjiX9OHdtjGelsye5mzJeZsF53kmb5Wrq0UGIwCAuzBqNUjdesAO3qYmXL09HoYpNwDXgRcynmID0gEGD1cUzYYwAZO4zu4re8zPvqZ/KHMOL5/gwdjb1559oCwDP9Nj7T3NlUBdvRgYwcPH8Zvm2SOBszso/2rL/84i7P+RBbjfuiD6JkH70h4si3Za/2AeEJDTjztw1kfZ6aJvIbwue+lJ+/ZWOhIaMrF67ZbU21bUKAXATt6PE0GD0zklYGbBx/qRBAlr8z7DOQBJB6yqRuTj3GNZa0cV83N2vr+6wA7Apm1Z8uFiA/jMOADEn2Msm9j+94bHgHN6tiWcagD8DVLERLL6hsA1gE26+RDbXw/rL0HZn++DGYuddraeD7MJdY3va1us3LqNavjPffWb9z6+VIW52z6xnY6lgKtKCDAbkWlHqtTB9hlb67w3lmkqANsyn3/HFsad9SEHIL52m9LHmabebBpU9W39VMH2LRnfvThv+qdfJ+sL/KqALsK1mnDw6E+XjoCtg8Hob5PtL3w87flFyTRS+7r2b631fK0rVegFwEbryoeOw9LqBABgjx/+5zj6FH1cdiU2wOQ7McEsAMhPtba6uAtxEuMDcNNeE4BqjKY94BbB9jYUAbrZh/9WzLgA04BNeYZU9n8yvq3doSG4MVvBa6tDYCPXd6ramV+XeoAm/rAOuvq9QOu8UD7d0L782Uwc6m7GOACiHmbRx97msFxs/JW+rA6MV6efBL6YZd53MkTYKOCUrsUEGC3S8ku6sdDqoc/m8LmlXMzL+ue/LDsIbpmgA188v5sD5J4WhdedUv64RP3JB9D7AERwP77mw7HV37gI1ekMz7+R2ZWvi3rmwLrpxlgUxdP+MOZd9h/aQ35JAD3I5mdFsYBYP/rq4fednDkUf8ufeqGpw5VzH4D61tvuKzQigL0jF/c4zUuu0NAOy4+3pN5qs9dek0REoKX/NHMTlsL6lkibIcvxMFzrjQ4BXoRsE0BQJD3O5MAOQ81VmektsCgpZEaG+DlYsIS4RIWf02eL7M6EWh9H+gTy62d33pdya+bHzYwhk8e4nx+q/vNxgewf/7znzd0F+fl7Yq6NTR0B76NZdfNxZ8D1G91HOt7pLZ+zTvJrpGar/rtDAUE2J2xDj1pBTAMhBIe0m4QtL4RjtAJA+LBCgmQW7JwEztudcscsWcwdlgbxmimDxcEfIulJR/SYnnatq5ALwN26yqophSQAlJACoykAgLskVRXfUsBKdBxCgiwO25JZJAUkAJSoOcUEGD33JJqQlJACtQpIMCuU0dlUkAKSAEp0A4FBNjtUFF9SAEp0DUKCLC7ZqlkqBSQAlKgaxUQYHft0slwKSAFhqKAAHsoqqmNFJACUkAKDEYBAfZg1FJdKSAFul4BAXbXL6EmIAWkgBToeAUE2B2/RDJQCkiBdiogwG6nmupLCkgBKSAFyhQQYJepojwpIAV6VgEBds8urSYmBaSAFOgYBQTYHbMUMkQKSIHRUECAPRoqawwpIAWkQH8rIMDu7/XX7KVA3ykgwO67JdeEpYAUkAKjrkBXA/YLr08adcE0oBSQAt2vwPFH7e/+SWgGUkAKSAEp0LEKdDVgd6yqMkwKSAEpIAWkgBSQAlKgbxUQYPft0mviUkAKSAEpIAWkgBSQAiOhgAB7JFRVn1JACkgBKSAFpIAUkAJ9q4AAu2+XXhOXAlJACkgBKSAFpIAUGAkFBNgjoar6lAJSQApIASkgBaSAFOhbBQTYfbv0mrgUkAJSQApIASkgBaTASCggwB4JVdWnFJACUkAKSAEpIAWkQN8qIMDu26XXxKWAFJACUkAKSAEpIAVGQgEB9kioqj6lgBSQAlJACkgBKSAF+lYBAXbfLr0mLgWkgBSQAlJACkgBKTASCgiwR0JV9SkFpIAUkAJSQApIASnQtwoIsPt26TVxKSAFpIAUkAJSQApIgZFQoKsB++kD/34kNFGfUkAK9LgCsyb8rx6foaYnBaSAFJACY6lAVwP2Pbt/Le078Iux1E9jSwEp0GUKvGfy29JZk3/cZVbLXCkgBaSAFOgmBQTY3bRaslUKSIFhKyDAHraE6kAKSAEpIAWaKCDAbiKQiqWAFOgtBQTYvbWemo0UkAJSoBMVEGB34qrIJikgBUZMAQH2iEmrjqWAFJACUuAtBQTYOhWkgBToKwUE2H213JqsFJACUmBMFBBgj4nsGlQKSIGxUkCAPVbKa1wpIAWkQP8oIMDun7XWTKWAFMgUEGDrNJACUkAKSIGRVkCAPdIKq38pIAU6SgEBdkcth4yRAlJACvSkAgLsnlxWTUoKSIEqBQTYVcooXwpIASkgBdqlgAC7XUqOQT8/+d//lHb8txsaRv7N3/+ThuNWDg7s35OeeejONG78xDT9rEX5tpV2ZXW++1d/3JB96vxL0y//6okNeTqQAqbAI9+4Ph1xhB2lNOejnzl8MEJ7AuwRElbdSoEKBZ5//vn06KOPpqOPPjotWLCgotboZd977725PQcPHkyvvfZabte11147egZopL5QQIDdwjLv3H5n+sk/7xlQc0YGoxMmTR2Qbxm7Hr8n7d+z0w6LbbsgAsC+9XNnFv0e/7556aOf3VQct7IDXG9ZvSC9cfDVvPqk42ampavvbqVpaZ2//4vL0z8+erj97/7Jd7sSsO2io3SSb2VOOGZaOvHU+cO6IKnrvx/Kblh+QsM0r9r4fMPxSBz0GmADC/v27Sukmjx5cpo3b15xHHdi/cWLF8cqOpYCbVOAc/OKK64o+jvvvPMajouCUdgBptevX5+efvrpAaPdfvvtA/KUIQWGo4AAuwX17rruk2nvc48MqHnO0qvT7MxDW5U2r5ybXn15IJi3CyIGA9gANNAIQPu099mH011rl/isNBz7egWwy3RpEMkdzL7g0nTOkqtdTvfuMu8p0w9ftI30TATYw1d49erVaefOxgt5vHGzZs0q7TzWF1iUyqTMtxSwizcu3IaStm7dmjZv3lw0HT9+fNq0aVNxPJo7d9xxR+KnLI3V3wHe/RNOOKHMJOV1uQIC7BYWsAqw67y9+1/cmXuGy7ofDsD6/loB7Ce3bUi7ntiWcnA6+Yx04ee/5rvIPde3XbuguBDAI7vwqq821BnMQT8CNvqcMn95OnfpNYORqmPqcofmR0/ek58jXIi16/xsZYIC7FZUqq8TgZnawNDatWvzW9+xdaw/VmAR7dJx5yiAp/fOO+/MPb0AIHc5hnqnA0BfuXJlIhyDRIjIsmXL8v3R/sW4ZgdjX3zxxWnhwoVptCGXiw4uigmbmTlzZuJvUqn3FBBgt7CmHrDHHTUhvfH6gaLVsrXbS8NE7t/yhfTUto15vdimXQDTCmB7gJlSAtgYCFQB4MRgD9d72auAPWnazAKg9+1+Ou342+sbzgO0u+zG7xfnRTft+HMEu9t1fraiwViM3WshIhGYTfcqkIn1BdimmLamACEUa9asscNhATadAOxALDHYY+mtveiii4o5jSXYdoodhRjaGREFBNgtyOoBG0j14SJVYSI+PAQ427/78C3cdgFMuwC7BQlartKrgB0vTspCSNq1ri2L3aaKYwG5ZvpYjN0vgI3GeLEj0Aiw7ezTtkqBdgN21Tijnd8pYNspdoy2/v02ngC7hRWPgE0Tg+wJk6alZWsfaOjFh4fgvSaUxOpT0YMYt+efzd7gYYm3eMycu8gOE33df9sXimNfXgfY9naGRzJPqyVsnTn3E/nhlPeeWXirmZ+9ysF7aq0dW7zc2PnD7MHNV/fvzeK5d+ft8dyekb35wWK7mwE2ceC7snAE+qFP5ofXnH5mn7+8sMmPzT5A+8xDX0/7Xnw6b0MeY04+blY2/h823EWg36e+vTHtydrQjsTcJ06aktvaipc+AnQEbMa45cr3533zi3W+7KZ/KI5th3qE6Dzz4B253RyTsOGk0y5o+tYWmzdzQXMS8546/YOZXpc0zDsvfOtXWTvGnJjpYHqVnSM0P+Pjh97k8e9+ZWrDuUiZ9WvrYGsw46xPVK5dXA/aMPdTsvX2GtK//9vgeCRSrwM2Ma52Gxy4BrJ9ahWw7U0L3MrGA0kirvv0009PPKiGNzIm+j7iiCPy7OOPPz4PBaCfu+++u7gNj03cmi9rH/uLx3hBfTwvdvBAJ7fcH3vssTykgfAY7CSkoSxumD64Nc+8XnrppVwrbKIuXn/2fYpjMi/6Zk4W+25voCAcgr4BVPZpiy2sCXaiXUxlD53SllhhtiRs8/NhDNrZOuONxfYqTenH1gDb/Hx9rD71zB62lqjDGCTmgT1RF9aC+TGOhT4QkhHr2XlhfdvWxrZ14Zhx49ytfqtb65f6Pv6afpkLibmZDth733335XbH88OvQd7Q/UIv5k17xiTRJ1pzvpNsfNuSV2UHZUrdrYAAu4X184ANHMw4+xPpgS1fLFry1g3yLfnwkOlnL8rBqAqwgZwd2Y+lOR/7TDoj+7EUQc+X1wF29Axaf7b1/fi6ESSpz9tQvr1xRQ7E1t5vfV/NANuP5fuwfd+X5UWNLN+2F666rYA7gJ2HNg1krY5ty/q3Mr+Nukddok1ldzKw5e9u/HQBxr5/2wf8P3LlVxrOH8qwH83Rvi59aPn6ARC8bcOK9OyDhy/aYnvTq9laxDn78zr2yXGZLXUa8DdDuU8CbK9Ga/sRmPkw9xAKFPBjKdaPISLAwbp163JQsDZxC8gBlcCDT9EzBzwAKzHRLoJ/rFN2XOZZBWgAm5jKbIztYxuOLS7XymIbYIy+/ZimoZ+/tfdbAJR4ZJ/ielB+8803Fxc1Vpcx0YyxBqMp4Ed/VQnIvPzyy/PiONeyNvYAbazLOUaeXXSgE3OL9Szf9x0fhPRl7NuYMb+V4zh+WRv7G2mlLloZmFtfN910U+maWHmr54fZYe207W4FBNgtrJ8HbKoTd71p1dyiZXzAzYeHADOPfOPPKj3YEdQiAEbQ8+WjAdgA3tYbLyvmWrbjbWoVsM2zT39Alo9rNwC0Ml4jaMm3e+PggTz0xtf32tMGj/y48RPy5lzkeFutz7Jt1N28/3jvzXtr7crgOr7+0OqWbfHeX5ydU2wtbb3hU7nn247rtguvvCWdmHmESdHuOr0GA9jxPJ14zNTMez4lu3jYWzwgy/h+LTjekj1A68OjyKtLAuw6dcrLIqDxYR7zfKhILLMPf3rHU03sLdDaLJUBbDPA9H2WgYovL9uPAIQN5l0vqw/gAz+WfHvK+CEZFLJvIGtlvo2VxzFNQ5s/Hmu7+EBL8zTTHoD2nuy4HnVzwibzjtJXTFHTCNc2Z/rw/ZhNca6xf44NdmNdvLXkWTKQjvUs3+pFGy3fb21Mn9fqfhy/rJ2Bra9rWlE/nh833nhjcbcgXhz4dqZzPD/KbCDP7KgqV353KSDAbmG9ImADAR6AgC8LE/FACoRcvG57Kmtvw0ZwiQAYgcmX1wE27Uj+FXw+/GNi9v5ue4e3B63otYzASvkZH/ts7jEGIne/NY6FtTQDbLSw9qZBDqPX/lYB2Xj951+6Pi+O+niYpAJtScyF/boLH7zC1PF3G/LGJb+i7iVVciAmnKLsy3miFxntmZN5bSn34OnnHMcGks+/9Ms5RDMHNLEHaLHLn39Rrwi8Xq+yc4T+aEMC+LE36so5wJtmKMeeqrfQEP70ncwLb8nPg7HxiHsNqCfANrVa30ZA48OcD3agycDOe4zL6tto3Lr2t6/tdj7wRJ/AqoeN6JE1wLT+CBsgTICEF9V7fWPbvFKTXx6ArCrAxhjMEVjjFXA2b+oYPLJPe+pgJyBkKfbrQSeWWRsLySCMwN7zjLa0tXAD6gLjq1atKoA2AmZcDzSnP+xjLtFbTTn9AfHNNEUX04J2jAXAk/y4jMXaYisXBPz4uyAWikM7dKaPZrpgH2/oiPX8/BnvyiuvbLhI4rxAWzvnWC/yGHcoyeZEW//gpp3b5B977LG53tjayvnh73J4Henr1ltvLTTmmD7tfGCf1MyOvJJ+db0CAuwWlrAMkCM8WJiIByvzbJe1t2EjEHmApk6ELV9eB9jWfx08N6vjLxaoCyQuXXO3NSvdNgPsskY5bGVx5hYu4CE/6jNj7uIihjj2RXvv7QY85y9fV4SPxPp1x1H3urqA5vlZqIZ5kal7yxXvKy4YOI5vm4nQSh/2FhJ/8UbbstCLeOFj51+ZXryjm/6rkj9HqBMhl1c91oVExTGtfbN5xPUqG7vK5uHk93oMtnnLomfNoDECgdVHU8DOezaj5xBYoY5BG208UHjANmijDol+DUQ59qD15ptvpl27dpFdmk466aQ8P8IaEAcYGjRSKV4kAGuAZlViTi+88EIOnFYHoDRb45jUibpYu7It7e2Vd5RHXeJ6+L6x7ZJLLmnolvnSB6lOUy5mCPWx5C80yIs6+X7jnO3csb7Yxjrkeds5JsV6ft3jOerLDrVu729/frY6Vtn54fWI60eZXXxVWT8UO6r6Un7nKiDAbmFtygAZz51/QMtg2oOVQU9Zexs2wokHaOpE0PPlIw3Y0bYy0LN52LYZYKNbfADR2trWw2acv9UBZk889cN5PLzlsY3gSZ6Fd/BQXR1oUtdSHNd7/1/JvOB7n3t4QJyzrXds6y8YrH+2MXwCwMa+mG/A6tvGtbEwlXjhZ23Qa8bZi/NvnrQ82zYD7AjK8SFRe+jV+rN5+L8Fysrm0Uod67dd234BbPTyH/5AKKEigJT3QnvA9h/8EQRNf98neR6qfPsygKkqB7A/+cnsYeuS9KEPfSh9+tOfzksirJV5wWOdaAfeWXuIjbplybeJ/eH59ODq2wNj9A3cMg7HZclrHvX0ZbT1mnFcV+7tjgBtnmf6IGEbNlry6xjn7IHS6sc6VbrEet7GOHdvg43Tzq3X0tvhx0CTwZwf8SLB+iJOmws1815bPttW7PD1td+dCgiwW1i3KkD24GHeUgvJsPAQuq9qT1kEJQ/QlEdY8+UjDdh+ftgSww3Ii6kOsPGI1z0s6fvyMFb3cB0hDNhl4IxXFL19TLf1Sx1CTCIgWrnfRt3LIDnqYxdZrbRlrHheGKA3A17a1p03/i4KdX1i7mhgelHWbLxop++vbN/Ok2b90raVi4myMYaT10+AjYfTh4rwYQ/MlgE2wOW9pVUAUgdFzcChqnyogN0K+Pl5cPu/7oE/O698mzpAtPpsATNu/VdBta/rITnq6cto4zXjuK7c2x0Bm7Z1ycNtnHMrOnuvvx8n9uVtjHP3XnTfR7v2vZbeDut/KOcHbbng8uFP1h9bQNseIrX8ZnZYPW27WwEBdgvrFwHD4C+GUBBHa29vMNii+6r2lNWBEuUR1kYTsCPYxvhn7IupDrC9dxmPMG9LmZwBMvHTdRoxBuD8RPbFPT96/FsD4NnHL1MXL/mTWd1nsxjg+FX1gGV8oJA2MUXdywA7eoutTittGc/rwbGdV1X51LEUzxvzYFs5NuzM3iRi56Pls416NQPhuDbMsy6du/TaPHa7Wb/00UqdurGGUtZPgI0+zUDLA5v/4MfjWfamjwhFHsx8+zKAqSoHsL/5zW9WLufHPvaxvCzCWhnUxdAIqxPDKSzW1zy7VbbFMcvmhXFcyBBqQiJ0hRAT+uZOALHUvL7Oktc86unLqO/t4riu3NsW1x0Psw+loS+fLI6dvDjnVgC7rE5ZX97GOHd/Lnnb2rXvtfR20P9Qzw+zDc3sFYWWZ9v48GmdHdZG2+5XQIDdwhpGwDAQomm8xW3dmTeS47r2EdI9mNM2xr+OJmDHsQkzALLrUhVgR+g0L6f1NRhPJmC7fcuaBtD2a2J9sgXM4wOFcWxf3/ajvQbPVs42XoD4r5n34FgG9TEG29/xiJ7xMnujXv588zayH9cxzsXbSv2oZZxnmT20iyleKMR2UWPax7Fjn+047jfARjMPf1FDD2xAIaBhKXoU8c7Gh9J8+2bg0Kzcxq3aRvAruwjAZv9goD2QFm/l+9jxOu99HDOCGbbG9jHeOYKk16yujL69Zhz7trHc2xbnGyGPtlUpzrkMnlupQ/+xnrcxrlWzePkqe1vN91p6O2gf9fI6x/WNbf341AW0ucCxFOt7O8rOYWunbXcrIMBuYf3qALnsdryHJbqvax8hAxgDROxtE/E9ysMBbPoGxPAY4+XlmOQBy8NXhEDqzr7g0jQn+2IXa4v99GUP+LUK2D6eO3qCGcdAi/79G08oI3nA5O0UfMkLMP3qy3sHxBnH/iPoHeqx8XdcF9YDz6ylvc99L/kv8SHfe5EjJPNwpj1siF6EEmGvJb+u0V7Gnp89RMmWFL3X/nxrRS+/xvTnteTY9LFzJF4EYsdHsgstewsNbajLF+rwjnhL8W/Dt8svfLI3jHgNaGfrbn2MxLYfAZvwBd48UZY8SETg4cMfryKeT8CBN0twG92SeYft2INDhArqNCu3fqq2Edaox5sqgGhS2S1+u0iIHl3Lp12ct7c9junLaEuKdTyQRo869b3mIwXY0SOLJ52x7AFJ7CChGaFDPj/Oh/IVK1YU5wHnQ6zj53yo50O/Yz2vXyyjRewH+zgP+SE9/vjj+dZ+nXbaabbbdFt3/g31/GB9mVO8O+DHis8K+DKMttdo8jcW+2k6KVXoWAUE2C0sTR0gR/igOw9LHNe1B0w2rzy7wRtLm6rk+24lBjt6Ea1f308VYFM3ei+tPYCN7STfVxVgl8G6xUIDhTEZaBlMAmd8eyHvtOZbID2YWciDQTHx8FPzby2ckn/r5C7CSt6y1WA8jhePra+YX3Uc37CCfax7WSx47ANAXrLm7uKihfIIvbGNP/ahO14vvi2RFPWKd0nixYD17UG8zB5bP+rbGtq6kYcG/q0u5DVLvn2zukMt70fARqsIEKafhz2gzMdsWx0DbDtmSxgEsacezDw4eJCyds3KrV7VtgzIquqS7z2iEXSZE9AG1HAB4pO3PY7py3wbPzfy7eE22sfkNR8pwGbMeOFAntnFPvNm/jE0I8I5dS1Z3ahLBGOrH+tF/eL8rR3nld1NsTEp+9KXvpSeeOKJvNopp5yS/viP/9iaNN36NYp2DPX8MPvRlT55w86OHTsK2zEq3j2wNt5gzkfOV3RU6g0FBNgtrGMdINM8Qmx8JVuz9gZF0RRgcE72nmX/ijQPs60AdvSG2hi+nzrApn70RFoftvV9VQE2datADjgFnMu+7bJKGxubthd+/tBDjs2g2L+H2dpXbZv15dsBovZeaJ/Pxde3N3yuFrKB64VX3VJ4p609FwR3XZd5uXcf9nJbmd/6OwHkD0Yv66dqrh6wuUD6uxsua2pPBOSq84+x/TMLZktsb/nt3LYC2A+98PZ8yLOO/3k7hx6RvuKHtYe3OGBZqEisD3TRp38VX+wHuKaOeRWtvA5gqNOs3Pqp2kZYwzNIXpmtxBxjo/cIls2fsfDE+7ASD19xTF/m7Yxx1lYGKGLnaMdgMz7wjAYWG242xa0HWCsrg3PKrG7UZaiAjY1cqPkHb80G29qYHH/qU59Kr7zySl5k4T9Wr9m22fk3lPMDfetsj3d5sDHCvNldpaGVa9tdCgiwW1gvvLj7Xzzshbjw819raEWMKzBFyt8m8taXpFilZu2pB4jw5SEGVMCNhSTcn8UbW5qevWrNvtQFwL7nlj+0ovT2I49KH/3spuLYdmLfgObcLNzB+uECwNKk42Zl415jh8UWCMvnmYUBWKKf95z24XTq/OUFIALYB1992aqk3/j9L6Vf/tUT82OgEQB8dvsdOXRae8Yjv0xjG5et9wajj716zsJVgMCHs35+nNX1DzcCsb+WebTPzB6q9GENhZElO3hfve4lVbK3kXwwnXTqBcXcy+pgEw9nYr+tLfW4MMD+2Zl2Zn9Ze9bu2QfvaLj4qJuP6UW4hk/o5c8dX8Y+7fw3jrI207OwFn8usH48PMq57udi9qCFhQr5/qmfr+9bFwvUnz53Uf6Qqz/3aBP/tnw/7dpvBtjA9aYdR+bDfW7eG+nkyf/WrqFHpB/ADii2xAd+VaIe9X0qqw/0AIR44jycAa1z5sypfM+v7wv4ju+fblbu7SrbL4M64JU5GeQAtLy5ocwTyLyo+/+3d64xc1TnHR9oGgKRSSoMHwADTaVySVsgAoe0BFBbI4GjRCJAQVHFTU0lLmqaBvhEgeRDFQgRUgxSicLlS00AgVoVqgKVgBBRAyqhbRygSoAYIjWAWmzCrQnu/tZ+luf9e+bMzuv1+87u/o/0ei7nnOc8z++c9fzn7JlZBA6iPJdt8k2Z1cUVvvItAUsaYuYVcQUD2uN8pNwW/pT6L5elfum4yTfa5i8YYYebJGZd4cdfvhEhP8YA9SIeeCFCaUe5wJw/TVquq4/4BkP840d9eAYg0je/+c0KETNuyuzq/CBObji7jI82tnVM8Jc2aCs+X/QHcTaVHzdGl+sPAQvs/vSFPTEBE1gCAiWBncX1p37zV9V5x7yzBB65iXEJ1AnsOiE9rj2Xmy4CGzZsqK677rqh0/vvv391/fXXT1cA9nauCFhgz1V3O1gTMIEmgX3Lk3tUjz3/a0NAFtf9HCcW2P3sl6Xyav369dU999wzbO6UU05Z8N72pfLB7ZjAuAQssMcl5XImYAIzQaBOYGdx/Znf+WX12cPfnYlYZy0IC+xZ69Fu8fAmk0hf+MIXqqOPPjoOvTWB3hGwwO5dl9ghEzCBXUlABXYW1+eufreahgcbdyWfPtu2wO5z79g3EzCBTMACO9PwvgmYwMwTCIH91v9V1brHPlT913/vPozZ4rr/XW+B3f8+socmYALbCFhgeySYgAnMFQEE9tEf/Vl17UMfql76323i+g9/+5eDt4W8tywcDvroe9U+ey1P28sS8E40ylseeGtDJF6Zl9/rHOe9NQETMIHlJmCBvdw94PZNwASWlMABv7Fb9ch//s9IXC9p4zWN/c3aty2wa7j4lAmYgAlMMwEL7GnuPftuAibQmQAz2M+++NrojSGdDUy4ggX2hIHanAmYgAn0gIAFdg86wS6YgAksHYFYg50fblx98K+qP/3E8r3zeo8PLF38bskETMAETGDXE7DA3vWM3YIJmECPCITAxiX/sEyPOsaumIAJmMAMEbDAnqHOdCgmYALtBLLAprSK7LOOfKfa89fb7biECZiACfSJwJYtW3ZwZ8WKFTucW6oTffNnqeKOdiywg4S3EyPwzpubq3vXfbF6+Zl/rVasXFV95uK/rVYedMSi7G989K7q0du/WmHzqJMvqD591hWLsuNKJhAEVGBzPovsAwdv9bj0pLctsgPYGNsnnniiuvHGG6tf/OIX1dq1a6tzzjlnjFrtRXhrCHZ5PR9vC+GHRj784Q+3V5yCEvfee2911113DT09/fTTh9ymwO2Zc3GWXv3IeLrjjjtGffTlL3+5Ou6440bHi9m56qqrqo0bN46qZvujkw07zz//fHX55ZePck844YTq4osvHh3P+o4F9hT0MEL17mvOKnqKkD3wsOOqjx198uBvTbHsrs7c8PfXV48P/iIdcOgnq9Mu/24cdtp+6/xDFpQ/+6r7Fi3WFxjaBQd3f/1Pqpef3dDZ8iU3v9C5jissnkCdwMbaUz/7QHXrEx+s3hr8iKNFdje+5557bvXmm2+OKl166aXVscceOzpe7M4NN9xQPfzww6Pqp556akVb0564ETnvvPMWhEGs++6774JzPtj1BCywy4wtsMt8SrkW2CU6PckbR2BnV3d21jjbWsz+Dx74TvW99V8bVUXwr73k26PjLjs3XfS71Ttvvf+1lwV2F3ouW0egSWBTdtPru1ffGLwfexpFNkLhySefrLZu3brkIvSiiy6qmG2OdOWVV07k/dTTLLBfeOGFipl9xLTeFNQJ7FtuuWVmZudjHEzD1gK73EsW2GU+pVwL7BKdnuR1Fdjh9nKK0Qe+85XqZ4OZ95UHHV598nN/uehZ55/82/0VM+Kkw4//fHXUmgsivN5tPYPduy6pdagksKkQIpv9K9a8Xa3s+Y/AIOSuvvrqoZDDZ358hYviUiZ8uPXWW4ci+6STTqrOOOOMiTSPEGWJCPaJC6E6DUtE8DNm9Jv6A/EdX7ezrAZuTktPwAK7zNwCu8ynlGuBXaLTkzwV2LHkgnXJr/50Y/X6qy9Vz3z/zh2WJ+yx197VOdc8WrF1Wh4CusTFy0GWpx9yq20Cm7KIbNKqj/T/FxZVIDQJumFA/mdJCJx55pmjdtwfIxS93NHPDzeHk7pBXOqAvQZ7qYmX27PALvPpRW6TwFbndO0z+Z8++4rGWV/E+ebXXq5e3fTD6oBDP1WtXHX4WGJ8y0DQR70V+6wa1kH0d020/8qmH1VbXts0tLFy1cfH9qGuLTi9+tLG4QOR+LXvIJ6mhysjhrCz9z4HDB7IPHBYl3XUXZmEHd3ujMCGD8tjwje1Hfzefev1Abfx2A1vygbMie+De36kyEjbm5XjcQT2NMWqAqGLoGOG+MUXXxyGS72cWPLBH7PHhxxySLXXXnsNt7nMOPthJ8oefPDBw1nosM0W+3E+yrVts++UZf0yf3EeLjwYGedL9ijLg1z77bffcKacOsQds9DUVT4le7tKYEdstF0XV/Z5HJ5hL/q4zqbGSVlYUZcE43HaUjsxLqLt0vjKcWEn+gIb8QAevsd5bSuOtU3K6+enq8AOhtghMYbafIk6+EOifBPDXBZGfIPTFOe4Als5lNg3zWAHe/zj81vnU5eHHCPOGA/jjMUhvB7/Y4Hd484J18YV2JTXZQox2x222CLMHhm8mQO7mnhTx+rP/kWt0Kb8hn+4vrYes+RfXPfvQ3Pq7+rPfWmwTORLo6Zo/x/X/Xm15dVNo3N5J8/yajw5L+pwY/H0AzcPxXGciy3r0U8YvHnkY584OU4Nt3ozgo8r9jmw+pebv7Kg3M6uZx9HYKsvp112e7Xx+3cNvpXY9oYB5ceymUdu/9oO/OiDo04+f9B/77POwTRxIsY1519bHTB4SHYe0qwIbBUGdX2XxYKKPpYxXHvttaO107Fcgbdb3HfffaPz2S4XvQsvvHCH9dVNF2Hq3nnnncO/sMP6bEQR53PiIo1ttpHq6iLoSBo/sVI33mYSNhAk2K176DK//STKs2XJBuIgxBvngg/7TSkzriuDCIEVSctm+3U8WYJDv+QUb2xB7MTbViKfuHmbS93SE8QMYoy+1gRf6uV+oAy8aSNEodbL/mtePiYOuNfZoU3ajj6OesqD9er4gp2c6GP6WpcREe9tt91WPfTQQ7n4UNgybrAVKX9m4lzTVsdnLldnh5ipo35EvWuuuWbEnTL0N4JTE/Hx+T3xxBMXZLUJ7BL7cT/bTeypz8PNedyMI7Cb+obAGAfT/OYgC+wFw7OfBypY60RzeI744hV5OWVRii3ymclsSggtRF5OvC5PxWfOZz/aUX+zQERcr7/qVK264DjscLJNYN/7rT+rfvLUAwvq1x380fnfqI44/vRRlopaZrrxrS4hQM+95nt1Wa3nFiOwD/uD00fimgYyv3H6IZcPB8fhRJ/Pg8i2wN42A8jFPoucEEgqZmIM5W0WApzXOmGLPBUhiCAVRpQjcUFFgEfSuvnhSRXYJbsIknXr1i0QXvjADUZTog4X/0g5pjinWxXNmr9YgY1IbhJlCDnycl/mdvXtJMTEmv064Rb1iJ0+RjSRlHWUy9tx+FC+jRFl1GcdX208YJKT1s952s91wjiXj/26G57IY6t24J2flchlYz+P75LPUZ6biXwD1SawJ8G+y+esTWCPMxYZg4xF+mnakgX2FPSYCtaSwCacJlGHqL7tsuNH4nqPPVdUf3zBdcPZ6h8/df9wFjhwZEGq7VNm78FsLw8wrjzo44MZ7ceG679DGGv5LPh4+DFmZrFD3oGHbps5jbXk+ZV+JYGtIhl7vLEEn7jReHXTQsF87mA9OstASHV1iemwgQiPeIYFt/+zWPHZ1BfZdp0vOT/4KdeVq46oTjj7r4dFeXNLvtEoxYqAP2Lwx3h48Dt/NXpLy87cSGR/+74/KwKbCzYXeS5SscwD9vnrXi6+cQFuu7iGQOLCjlijHrNR2KMtZgAjqRBWMRC2KK8imXNcNLGPcMuzxORl8a51swCpE334il2Y5Nf7YQQZPD8AABJ2SURBVJeZUWZ8I+UHETmHT4iHurrk55g4rktwIOWYcn/Ak3ZJ2h/ZvvKkPOIcH7kxyEtXyCPhO/mI7ZzPLCdveYmkwhAhiG36nLyoiz1mJEnciOSbojzDTT3ajNijnaYtcUf/0y5Jv9HIbZPfxINxSNv55gIhxixrJGbp89jlPPbpC61LngpjztWl3H8s7+CYvibBihhjvDGmeP8z20gxLoiBzxfjGd4ck4iZ84xnzlFe+0hjHUdgT4I98a5evbr6+c9/vsPnLL9Ks01ga7/GuCJuPvsxFnUMB8O+by2w+95DA/9UWHUV2CzdYPmAvj5PReP6K08didL8aj2d/SQvhHngw8eY/VR/QyBSVgVz+BZ2dKvlQ8RTru0VfirmEZZrLvjGsAkVtYjrs66+b7Q0RmPOMaiPpePFCuy8dp714twYZJ+4OTr76n8a3TAglm+6+PdGrmR/MycdOzojrmNiZHCGdmZFYEeXqNDMs6RRhm0WBBxzkbzsssuGQoALNxdeEvbiIj88sf0fFWYlQZjzVCSrf4iKfIOQRbTWzXkaNwKEmc+Y6dK6WTghrPKyAGLPs2RqGwQ5psylbj+z1nijfC7DuWxfhUf2vc63Un5un37OYluFi/ZxvDpQ/YnzEUuXbdP4yq9kpD84jqTt53gRrcQUYow6maW+QlJvtEq2o33dah8oRy2vIp/4aDc+c5SPm4Q418RJx26e7W8T2Ah/bi40ZfYq2pWP3vzUxRZ9VxLYiGj+/4mU+5RzJZ+iTt+3Fth976GBfypYVSTlEFRokReiVMVqnimm3CPrrx4tk8hrqlUktoli9TeLvSwSaZNZ06NPPq/62FEnj8Qi5yOpzxGLtpHFc9RFmN46mLGPlGdoVWBnHymvS200P2y2bZVd+J/rqS91sVA+C2WWtJxw9vtfpZPPDUWsa48xopxYY8+PEUXiYcdH/u6rcVh8KHZUaMp3LLC3dWC+MNd1KRdAhC+zVFz8ueCHCKB8rq8X4SxwVOjq19qanwWQ5pUEtl74VQRloZkv3sSiPnFOhX+OifxSyuI5t5vr5DKcz/aVZ57Vp6zWVcGb87NYVXHGbDozuZHoc0R2pOCtM9jYZHYW5iEIo8642zy+2Ocvj68uPJRX+I29fEORWYSfiM68VEiFXpTLW0S9/lAQN6XMNh9zzDGjm7yoo/4xtuqEbpTPW2LIfGI/ykSsHLcJ7KiDjfhsh73FssdmHm8cR9+VBLZ+tumnPJYYq/xF0s9AnO/z1gK7z72z3TcVSSGe6lxXYchM5xdv+I9h0TxDXVdXzyEGdc10qe2or/5mcap5UYctM+BrBmulYxkH55oEts685jaoF6lJ4Kqo1frqp+aH/bZtU/u5nvqy9uKbdngok/JqK9vQ/egnHQ9aTo8XG6fa6fOxBfa25RCIzLqE4GB2N3+dXVcuX9hVQMQFlnp6Ic316vKzwCnVVQGd62FX87PQVX/VJ+prmRwT+aWUBUduN9fJZTif7be13ZbfZFt5Zn/q9oOLssxlEZbcoGRxlPN1nxlPhGDb+OrCQ2+Ymvyu6wuNTceR+h/HOtsf55kBZpkEdiLpcqQcW5TRLX7xOczCV8twHLGy3yawx2WfbbaNNdptKlMS2Npn2Cml7FOpXJ/yLLD71BsNvqjYC/FUV1wFaZ4NzTOgdXX1HAK7S9tRX+uoaCP/wcFs6+bXXooqoy0z5/xATohsjSdmgHW5i7YRBlWURn0VtVq/LYaw37Ztaj/XU1+almmorWxD92OMKCctp8fKQfNn4dgCe9t6Xi6KmnQ2D9HEHyKKC35eW5wveE0XWOyrqMv16vKzwCnVbRNGmp/FlfrLxV4FopYZRxQFzyxwc7uRzzaX4Tjbb2u7Lb/JtvKk3VLKfVUSfIhKZhiVodquW07ADDp/Or668NC4wm8dA3VLObRMHn/qvx7TLjHl5SlRJn+j0tQfUVa3zCrn5RN5vbaue45YsVES2F3YZ5ttY412dRlO9F1JYKtd7JRS9qlUrk95Fth96o0GX1TshXjS4jqrS34WaypWySslZpRLyyya6qq/TaKN2fHh6+gevXP0oB02j1xz/ujhPfU5BLK2keuEX+o7DwWePVhnTVJRqz6qfc2PNtq2KorD/1xPfcl9lstlW/kBx1wm9rlRYRmJxsENFw84NqW9B2u94+amqcy0n7fAbhbYeVlEFgj0eZOIIU8vlnGBbatXl58FTqnNNmGk+Vnoqr91F+/MAj9zTByXUhZUiEfEp6ZchrxsX/3LeZRty2+yrTxhDZemhO+xpj3KwJWv7rkZy8IyP9wWZXWbZ3J1fOmMcI65LV6NK/qzNAbCN72pzOMvypS2zMRjAyb5BpQ6saxBBWjdDV1uIy/J4aYFO9EP6m/ESv2SwO7CPttU9rocqW65TPRdSWBrn8UDjplD3q8bizm/j/sW2H3sFfFJRZIKbITqDx68pfrRQKjmpOX0ob8mIZdtsH/bpccvmG3OD+BFWdZ+I+pI6m+bONV149nvJoGtdWhbf7VShWsW4ZqnPnaNITjoNoti8nZGYOclPjrTr+3Gsd5kcNPUdmMVdWd1O2sCW9eZjiPostjM/ZyFmQoNvdCWLsJxgcW2Xkhzvbr83G6proqnXA+7mp9jVjGnYk9FDPZyTByXUuZIubq6pTLKWuu35TfZViYadykmzdNZ1jZbKsTyWntsqwjNMbfF2zROtE3aCdHLPklt6zjaVmq8f5v80OUQTazwFyGdb+60rNrKn6cmga3/R7Sxzzbb+GjM2d+SwOaGJD9ovDPcx+udpS9lgb30zDu3qGJvHAPMcJ52+e0j0UsdtYNI44dJ+BXHSC8/+1j1yos/rNZe8u04tcNsLxn8cMuBA7HGrwHyOjzE/Tg/NMNDjnvve1B1+O9/fjjDii2dec9CuElgU09vGCIe8l4Z3HSw/jin0qvrpkFg63IPHtrkB3yYdY70o8GP02x+5aeDvv9unNphHTsi+6g1F4zGBjcr2P7k4AdqyJv1NGsCm/5SQcUDaPG1Mhc8LZPF5jBz+z/ZDjNnzHpxwediyF9OpYtwFkd6Ac71sKf5+UKrebmuisVcD7uan2NWcUh5OHFzwqwsX6dryjFpnh6rKME2y2y2bt06enVbZk39bF/r5zzKtuWXbKuQ5cE8/iIhxogfEYbPJGZUYROMOKdLDpQ/ZTRlv7DH+CJhi5uanHLMbfGWxonWZTwz206iTcZCTuPEwdhCzMKNccVnBW7cuOU4YqZaxyLtZZ74wOeLtewwzz7jL+fZYodYc8qfiSaBTfku7LPN7Eu0S9wRs/6/kB8YLglsbiYYi/lbEOzyFwmmjAM4TmOywJ6CXlNh3OZy3Wv0oo6K0jift3kGOc6r0I3zeRuzs+pvFq9tdvT1c1o+2qBdhOHdXz9r9GrB7Ivu5/d6kzeNM9j4nWexOa5L2n98wwFHfna9lMb9RqNkYxryZlFg110E6YssFvIFNovN3Gf5q+l8nn0Ee74Yli7CWRyVxA92NT/7rHm5TRUtuR52NV9jLsVK/RAQ7JNyTNvONP+rfkfJ7EPuD/Kzfe3PnEfZtvySbeWCvbqUWWt7Wp6xAU+YlVKemdVyvDYyv64xx6zt5zzsKO/se93NVG5bx7WOo1w29sdhyM0I8UZSH+N83obf+g1LLqOcog5lSgK7C/tsU9lnX3Q/j2/ySgKbfMR5nsXmXF3S/q4r08dzFth97BXxSQWrZA8PEaYxM9k2C/nI+q9Wz8i652wzPxiZz1OPnySvS/ltJervuAKbd1GvveSm0cw27ZQENvmIbH4sJf/ICucjYfPTgx9jGeen0vPPuZdiCNvjbCe5RIT2iHfYf9t/Rr3Oh/wNQOQjsnkN48vPbohTC7b0H7PerNue9TSLAhsRgcBhxienLBay6NILYdShPhdUtcPDYfvtt9+C2bPSRThfEFVY5Hq0q/nZZ83LdVXk5HrY1XyNmRk0mOm6WQQXM6tc/HNejgn7pYRtOGbBSPnsQ+4P8rJ96pbabssv2aYt2JTeUIFQRpAxy0rS9oYnt/+jZXOe7jNOsZVv1CjDjDIztPR3pC48SuMEe/QlolXbRazCinEQScdRnM9bHVs5j/14iwgx5dTkR5SJ5Sul8cO3U9nf/JkoCewu7LNN7XtmqOtYMrYZMznmNoFN3G1MGF+ewY4RUlW7vfHGG1vfPxxv7913362ee+658QpvL3X/pv2rV7a816nONBZGVCGQmhJLI7oKI2z+eLCEYkt6k8eKgRhdNVgiUHrIjTW9mwZvAcn19h0sR8kCVv3VB+dYuvGK/Moiv+ZYd2NA3NiLVFeGPPz68VP/PCj7/ixtk80ov3lQJ5L62BZD1GvbItRzqvMf37Mv9CV9Wkp18dJ/vzVYulOqC09+tTOnEqdcblb2Z1Fg0zdclPl6OsQxQpFfW4tZRURBJC6CIZ7iXGzVDhdOvrbGLm8wiET9uJhy8aZepFhawHGpXl0+Qj58LtWlPdqNlOtxTvObYsZGfK0PM76ipiwiOwQZ5xEVXZMuQchLLHJ/YDczK/GkbFt+yTb1I1EuC3nOZx+jHCwpR7uRYILPTeMoyukWW3mcRnva1114aN08NqN9bZcytK3jRMdR1NctbcKPbSTGLX7H+I3zeVvHkvL4Ep+nKJ/HT5O/Odann346qo62Rx555GifthG0Ma6b2GebdWONmB9//PEd7Iwa2r6DwN68+f3rN6ezP1G+jslix1fY7MPWM9h96AX7YAImsGQEZlVgLxnAOWgIYZNnCREi+ev+OUDgEE3ABHaSgAX2TgJ0dRMwgekiYIE9Xf21q7xlWQGzlSx/yQlxzfIJZtUi5a/M45y3JmACJlAiYIFdouM8EzCBmSNggT1zXbqogPLa0vhKXL8Ox3Ddj5MsqkFXMgETmCsCFthz1d0O1gRMwALbYwACWWA3EUFcsxZb18Y2lfd5EzABEwgCFthBwlsTMIG5IGCBPRfd3BokD3uxHIS/nHjgjAc7edgxP2iXy3jfBEzABNoIWGC3EXK+CZjATBGwwJ6p7nQwJmACJtBLAhbYvewWO2UCJrCrCFhg7yqytmsCJmACJhAELLCDhLcmYAJzQcACey662UGagAmYwLISsMBeVvxu3ARMYKkJWGAvNXG3ZwImYALzR8ACe/763BGbwFwTsMCe6+538CZgAiawJAQssJcEsxsxARPoCwEL7L70hP0wARMwgdklYIE9u33ryEzABGoIWGDXQPEpEzABEzCBiRKYaoH92i8/MlEYNmYCJjAfBPb5wOvzEaijNAETMAETWBYCUy2wl4WYGzUBEzABEzABEzABEzCBAgEL7AIcZ5mACZiACZiACZiACZhAVwIW2F2JubwJmIAJmIAJmIAJmIAJFAhYYBfgOMsETMAETMAETMAETMAEuhKwwO5KzOVNwARMwARMwARMwARMoEDAArsAx1kmYAImYAImYAImYAIm0JWABXZXYi5vAiZgAiZgAiZgAiZgAgUCFtgFOM4yARMwARMwARMwARMwga4ELLC7EnN5EzABEzABEzABEzABEygQsMAuwHGWCZiACZiACZiACZiACXQlYIHdlZjLm4AJmIAJmIAJmIAJmECBgAV2AY6zTMAETMAETMAETMAETKArAQvsrsRc3gRMwARMwARMwARMwAQKBCywC3CcZQImYAImYAImYAImYAJdCVhgdyXm8iZgAiZgAiZgAiZgAiZQIGCBXYDjLBMwARMwARMwARMwARPoSsACuysxlzcBEzABEzABEzABEzCBAoHeCOz33nuv4KazTMAETMAETMAETMAETGB6COy+++4TdXa3N954Y+tELdqYCZiACZiACZiACZiACcwxAQvsOe58h24CJmACJmACJmACJjB5AhbYk2dqiyZgAiZgAiZgAiZgAnNMwAJ7jjvfoZuACZiACZiACZiACUyegAX25JnaogmYgAmYgAmYgAmYwBwTsMCe48536CZgAiZgAiZgAiZgApMnYIE9eaa2aAImYAImYAImYAImMMcELLDnuPMdugmYgAmYgAmYgAmYwOQJWGBPnqktmoAJmIAJmIAJmIAJzDEBC+w57nyHbgImYAImYAImYAImMHkCFtiTZ2qLJmACJmACJmACJmACc0zAAnuOO9+hm4AJmIAJmIAJmIAJTJ6ABfbkmdqiCZiACZiACZiACZjAHBOwwJ7jznfoJmACJmACJmACJmACkydggT15prZoAiZgAiZgAiZgAiYwxwQssOe48x26CZiACZiACZiACZjA5AlYYE+eqS2agAmYgAmYgAmYgAnMMYH/B7zP7htJ/cEDAAAAAElFTkSuQmCC)

추측하기

우리가 가지고 있는 제약 조건들을 고려할 때, 우리가 다른 접근 방식들을 통해 우리의 방법을 추론할 수 있는지 알아보자.

- 신경망은 너무 무겁다. 깨끗하지만 최소한의 데이터 세트와 노트북을 통해 로컬로 훈련을 실행하고 있다는 사실을 감안할 때 신경망은 이 작업에 너무 무겁다.
- 2진 클래스 분류기가 없다. 우리는 2진 클래스 분류기를 사용하지 않기 때문에, 그것은 1-vs-all을 배제한다.
- 의사 결정 트리 또는 로지스틱 회귀 분석이 작동할 수 있다. 의사결정 트리가 작동하거나 다중 클래스 데이터에 대해 로지스틱 회귀 분석을 수행할 수 있다.
- 다중클래스 Boosted Decision Tree는 다른 문제를 해결한다. 다중클래스 부스트 결정 트리는 예를 들어 순위를 구축하도록 설계된 작업과 같은 비모수 작업에 가장 적합하므로 우리에게 유용하지 않다.

Scikit-learn 이용하기

Scikit-learn을 사용하여 데이터를 분석할 것이다. 그러나 Scikit-learn에서는 로지스틱 회귀 분석을 사용하는 여러 가지 방법이 있다. 전달할 파라미터를 살펴보자.

기본적으로 Scikit-learn에게 로지스틱 회귀 분석을 수행하도록 요청할 때 지정해야 하는 multi_class와 solver라는 두 가지 중요한 파라미터가 있다. multi_class 값은 특정 동작을 적용한다. solver의 값은 사용할 알고리즘이다. 모든 solver를 모든 multi_class 값과 쌍으로 구성할 수 있는 것은 아니다.

문서에 따르면, 다중클래스 사례에서 훈련 알고리즘은 다음과 같다.

multi_class 옵션이 ovr로 설정된 경우 one-vs-rest(OvR) 체계를 사용한다.
multi_class 옵션이 multinomial로 설정된 경우 cross-entropy loss를 사용한다. (현재 multinomial 옵션은 'lbfgs', 'sag', 'saga' 및 'newton-cg' solvers에서만 지원된다.)"

Scikit-learn은 solvers가 다양한 종류의 데이터 구조에서 나타나는 다양한 문제를 처리하는 방법을 설명하는 이 표를 제공한다.

![solvers.png](data:image/png;base64,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ARAAARAAARAAARAAARAAARAAARAISgLRIPQH5XlDpUEABEAABEAABEAABEAABEAABEAABEAABEAABEAABEDAIAChHxcCCIAACIAACIAACIAACIAACIAACIAACIAACIAACIAACAQxAQj9QXzyUHUQAAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEQgNCPawAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEgpgAhP4gPnmoOgiAAAiAAAiAAAiAAAiAAAiAAAiAAAiAAAiAAAiAAAhA6Mc1AAIgAAIgAAIgAAIgAAIgAAIgAAIgAAIgAAIgAAIgAAJBTABCfxCfPFQdBEAABEAABEAABEAABEAABEAABEAABEAABEAABEAABCD04xoAARAAARAAARAAARAAARAAARAAARAAARAAARAAARAAgSAmAKE/iE8eqg4CIAACIAACIAACIAACIAACIAACIAACIAACIAACIAACEPpxDYAACIAACIAACIAACIAACIAACIAACIAACIAACIAACIBAEBOA0B/EJw9VBwEQAAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEI/bgGQAAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEQCCICUDoD+KTh6qDAAiAAAiAAAiAAAiAAAiAAAiAAAiAAAiAAAiAAAiAAIR+XAMgAAIgAAIgAAIgAAIgAAIgAAIgAAIgAAIgAAIgAAIgEMQEIPQH8clD1UEABEAABEAABEAABEAABEAABEAABEAABEAABEAABEAAQj+uARAAARAAARAAARAAARAAARAAARAAARAAARAAARAAARAIYgIQ+oP45KHqIAACIAACIAACIAACIAACIAACIAACIAACIAACIAACIAChH9cACIAACIAACIAACIAACIAACIAACIAACIAACIAACIAACAQxAQj9QXzyUHUQAAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEQgNCPawAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEgpgAhP4gPnmoOgiAAAiAAAiAAAiAAAiAAAiAAAiAAAiAAAiAAAiAAAhA6Mc1AAIgAAIgAAIgAAIgAAIgAAIgAAIgAAIgAAIgAAIgAAJBTABCfxCfPFQdBEAABEAABEAABEAABEAABEAABEAABEAABEAABEAABMJd6L969SoogwAIgAAIgAAIgAAIgAAIgAAIgAAIgAAIgAAIgAAIgMAzJ5AmTZpnXofwqACE/vCgijJBAARAAARAAARAAARAAARAAARAAARAAARAAARAAAQiHAEI/Q5PyY+//+UwJ7KBAAiAAAiAAAiAAAiAAAiAAAiAAAiAAAiAAAiAAAiAgHsEEsSO6V5hEaikcLfoh9Afgc42qgICIAACIAACIAACIAACIAACIAACIAACIAACIAACUZgAhH6HJx9Cv0NwyAYCIAACIAACIAACIAACIAACIAACIAACIAACIAACIOAqAQj9DnFC6HcIDtlAAARAAARAAARAAARAAARAAARAAARAAARAAARAAARcJQCh3yFOCP0OwSEbCIAACIAACIAACIAACIAACIAACIAACIAACIAACICAqwQg9DvECaHfIThkAwEQAAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEQcJUAhH6HOCH0OwSHbCAAAiAAAiAAAiAAAiAAAiAAAiAAAiAAAiAAAiAAAq4SgNDvECeEfofgkA0EQAAEQAAEXCbw999/03fffmspNV68eJTgpZcscdgAARAAARAAARAAARAAARAAARAAgchKAEK/wzMLod8hOGQDARAAARCIFAQeP35MF86fpzt37tL33z+kf/75hxImTEiJXklE72bKSHHixHlqv/Pc559Tw3r1LccrVbo0DRs5whKHDRAAARAAARAAARAAARAAARAAARCIrAQg9Ds8sxD6HYJDNhAAARAAgaAmcO/ePZo1fQZt37aNHj165PW3lChZkpq3aklp06b1msatHRD63SKJckAABEAABEAABEAABEAABEAABIKVAIR+h2cOQr9DcMgGAiAAAiAQtARY3O/VvUeo6t++40fUqHHjUOUJbWII/aElhvQgAAIgAAIgAAIgAAIgAAIgAAKRjQCEfodnFEK/Q3DIBgIgAAIgEJQE9uzeQ107dXJU9y7du1GdunUd5Q0kE4T+QCghDQiAAAiAAAiAAAiAAAiAAAiAQGQmAKHf4dmF0O8QHLKBAAiAAAgEHYHffvuNShYt5tVVT/r06Y3fdOnSJa+/bduunZQ0aVKv+8OyA0J/WOghLwiAAAiAAAiAAAiAAAiAAAiAQGQgAKHf4VmE0O8QHLKBAAiAAAgEHYF1a9fSoP4DPOrd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Eu0mBf3RkmI5EkidBNasXiP733/fObgKlSrKgw89lDoP9D8+qmNHj0mzxo3lokwXy8033yK578gtNWvVkmuuueY/7hl3TwIkQAIpnwDvsSn/HPII4kdg08ZNMnrECLn0ssvk9ttvl3vvv09q1KwpGTNmjF9DLE0CJJCsCHBsJ6vTkeY6A58QmzdtksyXZ5a77rpbChUpIiVLlUxzHHjAyYMABf0ezwMF/R7BsRoJkAAJkAAJkAAJkAAJkAAJkAAJkAAJkAAJkAAJkICvBCjo94iTgn6P4FiNBEiABEiABEiABEiABEiABEiABEiABEiABEiABEjAVwIU9HvE+eWXX3qsyWokQAIkQAIkQAIkQAIkQAIkQAIkQAIkQAIkQAIkQAIk4B+BXLly+ddYMmop3QUVErM/FPQnJl22TQIkQAIkQAIkQAIkQAIkQAIkQAIkQAIkQAIkQAIkEC0BCvqjJcVyJEACJEACJEACJEACJEACJEACJEACJEACJEACJEACJEACSUYg0TX6k+xIuCMSIAESIAESIAESIAESIAESIAESIAESIAESIAESIAESSIMEKOhPgyedh0wCJEACJEACJEACJEACJEACJEACJEACJEACJEACJJB6CFDQn3rOJY+EBEiABEiABEiABEiABEiABEiABEiABEiABEiABEggDRKgoD8NnnQeMgmQAAmQAAmQAAmQAAmQAAmQAAmQAAmQAAmQAAmQQOohQEF/6jmXPBISIAESIAESIAESIAESIAESIAESIAESIAESIAESIIE0SICC/jR40nnIJEACJEACJEACJEACJEACJEACJEACJEACJEACJEACqYcABf2p51zySEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABNIggWQv6P/777/l22+/k+PHj8vRY8flr7/+kuzZr5Prr79ectx0o2TKlCkNnraUfciff/GlfP3113Ly++/lkkyXyPXqfD7wwANy6aWXpOwDS+a979y1h7y1ZYvTy1EjhkmRwoWSeY/ZPRIgARIgARIgARIgARIgARIgARIgARIgARIggWgIJFtB/08//yxLly6X6TNnO0L+UAfzv//9T+rVrS21alZXQv+bgor8+eefUvSx4iZ94Iv9pHSpkmY7mkjD55rKRx9/7BR9tm4dad2qhRPfvWevtGrTLs4msl6bVW7KcaOU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BEiABEiABEiABEgg2RDo3aOHrF61OqA/9Rs2lDbt2sbbrn5AI9wgAQ8EKOj3AA1VKOj3CI7VSCCFEfjwww9l1fIVTq/vvf8+qVK1ago7gpTR3WNHj0mzxo3lokz/Z+88wKsoujB8KGIDQaRKUTqiWECqgCK9KUUQDSC9Iz10AQVFSCD0KlWKdKSHUEKR3kUBEZCOFBGR4q/mn7M4k9m9e0tuNnCTfPM8yc7OnJndfffO7uyZmXNSUPbsz1GevHno/fr1DQfH8eMKcJYgAAIgAAIgAAIgAAIgAAIgAAKJjcCosJG0cf16SpkqJeXP/wIVL1mSypUvl9gw4HoDhAAU/X7eCCj6/QSHYiAAAiAAAiAAAiAAAiAAAiAAAiAAAiAAAiAAAiAAAo4SgKLfT5xQ9PsJDsVAAARAAARAAARAAARAAARAAARAAARAAARAAARAAAQcJQBFv584T5486WdJFAMBEAABEAABEAABEAABEAABEAABEAABEAABEAABEAAB5wjkzJnTucoCqKYkUSLE5flA0R+XdFE3CIAACIAACIAACIAACIAACIAACIAACIAACIAACICArwSg6PeVFORAAARAAARAAARAAARAAARAAARAAARAAARAAARAAARAAAQeGIE4n9H/wK4EBwIBEAABEAABEAABEAABEAABEAABEAABEAABEAABEACBREgAiv5EeNNxySAAAiAAAiAAAiAAAiAAAiAAAiAAAiAAAiAAAiAAAgmHABT9Cede4kpAAARAAARAAARAAARAAARAAARAAARAAARAAARAAAQSIQEo+hPhTcclgwAIgAAIgAAIgAAIgAAIgAAIgAAIgAAIgAAIgAAIJBwCUPQnnHuJKwEBEAABEAABEAABEAABEAABEAABEAABEAABEAABEEiEBKDoT4Q3HZcMAiAAAiAAAiAAAiAAAiAAAiAAAiAAAiAAAiAAAiCQcAhA0Z9w7iWuBARAAARAAARAAARAAARAAARAAARAAARAAARAAARAIBESCHhF/z///EPnzp2nS5cu0YWLl+jvv/+mTJkyUubMmSlb1iz06KOPJsLbhksGAfcEgnv2oU2RkYZA2PAQKlmiuHth5IAACIAACIAACIAACIAACIAACIAACIAACIAACMR7AgGr6L/x+++0ePFS+mraDEPJb0f6ySefpIYNPqT679cVSv+sLiL/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axa57rrrJH369DGXFVSGXw4flgNK7L/88svlGrVuQEKdFkHVm5hy/v33X9mrRvYfUm3/49gfDsvMWTJL9uzZXSaPElNHpLyYqbF371754fsf1GLAFznrLWRX5/G8886LlI1xJEACJEACJEACJEACJEACJEACJEACJEACJEAC5zABCv1xnvyUJvTHeRjMRgIkQAIkQAIkQAIkQAIkQAIkQAIkQAIkQAIkQAIkkMoJUOiP8wRS6I8THLORAAmQAAmQAAmQAAmQAAmQAAmQAAmQAAmQAAmQAAkESoBCf5w4KfTHCY7ZSIAESIAESIAESIAESIAESIAESIAESIAESIAESIAEAiVAoT9OnBT64wTHbCRAAiRAAiRAAiRAAiRAAiRAAiRAAiRAAiRAAiRAAoESoNAfJ04K/XGCYzYSIAESIAESIAESIAESIAESIAESIAESIAESIAESIIFACVDojxMnhf44wTEbCZAACZAACZAACZAACZAACZAACZAACZAACZAACZBAoAQo9MeJk0J/nOCYjQRIgARIgARIgARIgARIgARIgARIgARIgARIgARIIFACFPrjxLlz5844czIbCZAACZAACZAACZAACZAACZAACZAACZAACZAACZAACQRHIGfOnMEVloJKSndKuaRsD4X+pKTLskmABEiABEiABEiABEiABEiABEiABEiABEiABEiABKIlQKE/WlJMRwIkQAIkQAIkQAIkQAIkQAIkQAIkQAIkQAIkQAIkQAIkkGwEknxEf7IdCSsiARIgARIgARIgARIgARIgARIgARIgARIgARIgARIggXOQAIX+c/Ck85BJgARIgARIgARIgARIgARIgARIgARIgARIgARIgATSDgEK/WnnXPJISIAESIAESIAESIAESIAESIAESIAESIAESIAESIAEzkECFPrPwZPOQyYBEiABEiABEiABEiABEiABEiABEiABEiABEiABEkg7BCj0p51zySMhARIgARIgARIgARIgARIgARIgARIgARIgARIgARI4BwlQ6D8HTzoPmQRIgARIgARIgARIgARIgARIgARIgARIgARIgARIIO0QoNCfRs7liRMn5edffpbrsmdPI0eU9g5j167dcvXVWeXiiy9OewfHIyIBEiABEiABEiABEiABEiABEiABEiABEiABEjhrBCj0nzX0sVXctl0HWbFy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AAmQAAmQAAmQAAmQAAmQAAmQAAmQQNAEKPQHTZTlkQAJpGgCqUXoB8Qvv/xKni1TzuGZkE3oFA09xsbBxFmhJ4o65m6wFsDSDxanmlHwPx06JC+36yDffPudy04kRvNjTYOE1nqIERWTkwAJkAAJkAAJkAAJkAAJkAAJkAAJkIBDgEI/LwQSIIFzikBqEvpxYl7r3NUsrvvOtMnOwt1p/YTZi4gPHzpInixaJNUc8k61BotesFg3Gp0VY0cPl1tvuUUHcUsCJEACJEACJEACJEACJEACJEACJEACgRKg0B8oThZGAiSQ0gn89ttR6duvv9PMDBkySNs2rVJ0k3/55RcZNGSYYFHXBx/IJyVLFE/R7Q2icW/1HyhYzCtTpkwCczepye3du0+qvVBLLr74IrnxxhvktltvlerVqjoLa6am42BbSYAESIAESIAESIAESIAESIAESIAEUhcBCv2p63yxtSRAAiRAAiRAAiRAAiRAAiRAAiRAAiRAAiRAAiRAAiTgIkCh34WDOyRAAiRAAiRAAiRAAiRAAiRAAiRAAiRAAiRAAiRAAiSQuggkudC/c+fO1EWErSUBEiABEiABEiABEiABEiABEiABEiABEiABEiABEkiTBHLmzJkmj4tCf5o8rTwoEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABLwEK/V4iUe7/dvLvKFMyGQmQAAmQAAmQAAmQAAmQAAmQAAmQAAmQAAmQAAmQAAkkHYGM6S9MusLPYslJPqKfQv9ZPLusmgRIgARIgARIgARIgARIgARIgARIgARIgARIgARIwBCg0G9QxOah0B8bL6YmARIgARIgARIgARIgARIgARIgARIgARIgARIgARJIGgIU+uPkSqE/TnDMRgIkQAIkQAIkQAIkQAIkQAIkQAIkQAIkQAIkQAIkECgBCv1x4qTQHyc4ZiMBEiABEiABEiABEiABEiABEiABEiABEiABEiABEgiUAIX+OHFS6I8THLORAAmQAAmQAAmQAAmQAAmQAAmQAAmQAAmQAAmQAAkESoBCf5w4KfTHCY7ZSIAESIAESIAESIAESIAESIAESIAESIAESIAESIAEAiVAoT9OnBT64wSXwrIdP35c0qdPL+edd14KaxmbQwIkQAIkQAIkQAIkQAIkQAIkQAIkQAIkQAIkQALREaDQHx2nkFQU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larKOgEdCSQXAQr9cZKm0B8nOGZLEwTGjh7jssGLdRpghglTwTASHaO0N6hFaWGn99tvvjHHHO2UNZOBHhIgARIgARIgARIgARIgARIgARKIgoA2N6zXpkOW4aNHyYMPPhhFbiYhgcQR+OPYHzJ50iTB+pbarfhoddyz/nUZ3JJALAQo9MdCy0pLod+CQe85RwBmb0o+/YwZ1W8DgK17P9vtWOS5XPnydlL6SYAESIAESIAESIAESIAESIAESCBRBDasXy8T1PppWAfMXtew4COPyKChQxJVNjOTQEIEJowfLx+tXi1bNm12JW3eqqXUqFnTFcYdEkhqAhT64yRMoT9OcMyWZgj8ouyz9+jRQ5arhWYiOSyq2X/QQMmbL1+kZIwjARIgARIgARIgARIgARIgARIggZgJzJ0zV7p26uTKh1H8vd7qy9HULircSQoCr3XoIAsXLHQVXaNWLWnavFnE9QxdGbhDAgERoNAfJ0gK/XGCY7Y0R2DtmjXy+Wefy84d38m3330nf//5l+TKlUtuv+MOya1M+dx7372SSS2+R0cCJEACJEACJEACJEACJEACJEACQRNYumSpDHjrLbnkf/+T3Llzy5133yUVK1VyFoMPui6WRwJeAlgTYvnSpZLhsgxy2223y0MFCkiRokW8ybhPAslCgEJ/nJgp9McJjtlIgARIgARIgARIgARIgARIgARIgARIgARIgARIgAQCJUChP06cFPrjBMdsJEACJEACJEACJEACJEACJEACJEACJEACJEACJEACgRKg0B8nzp07d8aZk9lIgARIgARIgARIgARIgARIgARIgARIgARIgARIgARIIDgCOXPmDK6wFFRSulPKJWV7KPQnJV2WTQIkQAIkQAIkQAIkQAIkQAIkQAIkQAIkQAIkQAIkEC0BCv3RkmI6EiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiCBZCOQ5CP6k+1IWBEJkAAJkAAJkAAJkAAJkAAJkAAJkAAJkAAJkAAJkAAJnIMEKPSfgyedh0wCJEACJEACJEACJEACJEACJEACJEACJEACJEACJJB2CFDoTzvnkkdCAiRAAiRAAiRAAiRAAiRAAiRAAiRAAiRAAiRAAiRwDhKg0H8OnnQeMgmQAAmQAAmQAAmQAAmQAAmQAAmQAAmQAAmQAAmQQNohQKE/7ZxL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xBOlWLGL5Zqrr5L69erI8ccfH/GZWB0JkAAJkAAJkAAJkEBUBCjoj4ok6yEBEiABEiABEiABEiABEiABEiABEiABEiABEiABEiCBP4EABf1/AnSekgRIgARIgARIgARIgARIgARIgARIgARIgARIgARIgASiIpDlgv7PPvssqrayHhIgARIgARIgARIgARIgARIgARIgARIgARIgARIgARJImkDRokWTPjYnH0hBf06+O2wbCZAACZAACZAACZAACZAACZAACZAACZAACZAACZBAZAQo6I8MJSsiARIgARIgARIgARIgARIgARIgARIgARIgARIgARIgARKIikCWjRreogAABEhJREFUa/RH1VDWQwIkQAIkQAIkQAIkQAIkQAIkQAIkQAIkQAIkQAIkQAIkEEuAgv5YJkwhARIgARIgARIgARIgARIgARIgARIgARIgARIgARIggZQhQEF/ytwqNpQESIAESIAESIAESIAESIAESIAESIAESIAESIAESIAEYglQ0B/LhCkkQAIkQAIkQAIkQAIkQAIkQAIkQAIkQAIkQAIkQAIkkDIEKOhPmVvFhpIACZAACZAACZAACZAACZAACZAACZAACZAACZAACZBALAEK+mOZMIUESIAESIAESIAESIAESIAESIAESIAESIAESIAESIAEUoYABf0pc6vYUBIgARIgARIgARIgARIgARIgARIgARIgARIgARIgARKIJUBBfywTppAACZAACZAACZAACZAACZAACZAACZAACZAACZAACZBAyhCgoD9lbhUbSgIkQAIkQAIkQAIkQAIkQAIkQAIkQAIkQAIkQAIkQAKxBCjoj2XCFBIgARIgARIgARIgARIgARIgARIgARIgARIgARIgARJIGQIU9KfMrWJDSYAESIAESIAESIAESIAESIAESIAESIAESIAESIAESCCWAAX9sUyYQgIkQAIkQAIkQAIkQAIkQAIkQAIkQAIkQAIkQAIkQAIpQ4CC/pS5VWwoCZAACZAACZAACZAACZAACZAACZAACZAACZAACZAACcQSoKA/lglTSIAESIAESIAESIAESIAESIAESIAESIAESIAESIAESCBlCFDQnzK3ig0lARIgARIgARIgARIgARIgARIgARIgARIgARIgARIggVgCFPTHMmEKCZAACZAACZAACZAACZAACZAACZAACZAACZAACZAACaQMAQr6U+ZWsaEkQAIkQAIkQAIkQAIkQAIkQAIkQAIkQAIkQAIkQAIkEEuAgv5YJkwhARIgARIgARIgARIgARIgARIgARIgARIgARIgARIggZQhQEF/ytwqNpQESIAESIAESIAESIAESIAESIAESIAESIAESIAESIAEYglQ0B/LhCkkQAIkQAIkQAIkQAIkQAIkQAIkQAIkQAIkQAIkQAIkkDIEKOhPmVvFhpIACZAACZAACZAACZAACZAACZAACZAACZAACZAACZBALAEK+mOZMIUESIAESIAESIAESIAESIAESIAESIAESIAESIAESIAEUoYABf0pc6vYUBIgARIgARIgARIgARIgARIgARIgARIgARIgARIgARKIJUBBfywTppAACZAACZAACZAACZAACZAACZAACZAACZAACZAACZBAyhCgoD9lbhUbSgIkQAIkQAIkQAIkQAIkQAIkQAIkQAIkQAIkQAIkQAKxBCjoj2XCFBIgARIgARIgARIgARIgARIgARIgARIgARIgARIgARJIGQIU9KfMrWJDSYAESIAESIAESIAESIAESIAESIAESIAESIAESIAESCCWwP8D1AhUInmEQWMAAAAASUVORK5CYII=)

### 데이터 분할하기

train_test_split()를 호출하여 데이터를 훈련 및 테스트 그룹으로 나눕니다.


```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

로지스틱 회귀 적용하기

다중 클래스 케이스를 사용 중이므로 사용할 scheme과 설정할 solver를 선택해야 한다. 다중 클래스 설정 및 liblinear solver와 함께 로지스틱 회귀 분석을 사용하여 훈련한다.

multi_class를 ovr로 설정하고 solver를 liblinear로 설정한 로지스틱 회귀 분석을 만든다.


```python
lr = LogisticRegression(multi_class='ovr',solver='liblinear')
model = lr.fit(X_train, np.ravel(y_train))

accuracy = model.score(X_test, y_test)
print ("Accuracy is {}".format(accuracy))
```

    Accuracy is 0.8056713928273561


80%가 넘는 정확도가 나왔다.

하나의 데이터 행(#50)을 테스트하면 이 모델이 작동하는 것을 볼 수 있다.


```python
print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
print(f'cuisine: {y_test.iloc[50]}')
```

    ingredients: Index(['cane_molasses', 'chicken', 'onion', 'peanut_butter', 'scallion',
           'sesame_oil', 'soy_sauce', 'tabasco_pepper'],
          dtype='object')
    cuisine: thai


더 자세히 살펴보면 다음과 같은 예측의 정확성을 확인할 수 있다.


```python
test= X_test.iloc[50].values.reshape(-1, 1).T
proba = model.predict_proba(test)
classes = model.classes_
resultdf = pd.DataFrame(data=proba, columns=classes)

topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
topPrediction.head()
```

    /usr/local/lib/python3.7/dist-packages/sklearn/base.py:451: UserWarning: X does not have valid feature names, but LogisticRegression was fitted with feature names
      "X does not have valid feature names, but"






  <div id="df-0ecdf417-245f-4e79-b393-ed389ec63ad9">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>thai</th>
      <td>0.689479</td>
    </tr>
    <tr>
      <th>chinese</th>
      <td>0.259955</td>
    </tr>
    <tr>
      <th>korean</th>
      <td>0.029757</td>
    </tr>
    <tr>
      <th>japanese</th>
      <td>0.018253</td>
    </tr>
    <tr>
      <th>indian</th>
      <td>0.002555</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-0ecdf417-245f-4e79-b393-ed389ec63ad9')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }
    
    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }
    
    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }
    
    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-0ecdf417-245f-4e79-b393-ed389ec63ad9 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';
    
        async function convertToInteractive(key) {
          const element = document.querySelector('#df-0ecdf417-245f-4e79-b393-ed389ec63ad9');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;
    
          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




분류 보고서를 출력하여 더 자세히 알아보자.


```python
y_pred = model.predict(X_test)
print(classification_report(y_test,y_pred))
```

                  precision    recall  f1-score   support
    
         chinese       0.79      0.70      0.74       242
          indian       0.89      0.92      0.90       251
        japanese       0.73      0.75      0.74       239
          korean       0.81      0.76      0.79       230
            thai       0.80      0.89      0.84       237
    
        accuracy                           0.81      1199
       macro avg       0.80      0.80      0.80      1199
    weighted avg       0.81      0.81      0.80      1199


​    

## Cuisine Classifiers 2

이 두 번째 분류 과정에서는 숫자 데이터를 분류하는 더 많은 방법을 살펴볼 것이다. 또한 다른 분류기를 넘어 한 분류기를 선택하는 데 미치는 영향에 대해서도 배울 수 있다.

분류 지도

이전에는 Microsoft의 치트 시트를 사용하여 데이터를 분류할 때 사용할 수 있는 다양한 옵션에 대해 배웠다. Scikit-learn은 추정기(분류기의 다른 용어)를 더 좁히는 데 도움이 될 수 있는 유사하지만 보다 세분화된 치트 시트를 제공한다.


계획
이 지도는 데이터를 명확하게 파악하면 의사 결정에 이르는 경로를 '걷기'할 수 있으므로 매우 유용하다.

- 50개 이상의 샘플을 가지고 있다.
- 어떤 범주를 예측하고 싶다.
- 레이블이 지정된 데이터가 있다.
- 10만개 미만의 샘플을 가지고 있다.
- Linear SVC를 선택할 수 있다.
- 만약 그게 안 된다면, 우리는 수치 데이터를 가지고 있기 때문에
 - KNeighbors 분류기를 사용해 볼 수 있다.
 - 그래도 문제가 해결되지 않으면 SVC 및 앙상블 분류기를 사용해 본다.


```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
import numpy as np
```


```python
#데이터 분할하기
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

Linear SVC classifier

SVC(Support-Vector clustering)는 머신러닝 기술인 Support-Vector machine의 하위 모델이다. 이 방법에서는 'kernel'을 선택하여 레이블을 군집화하는 방법을 결정할 수 있다. 'C' 파라미터는 파라미터의 영향을 조절하는 '정규화'를 의미한다. 커널은 여러 가지 중 하나일 수 있다. 여기서는 linear SVC를 활용하도록 'linear'로 설정한다. Probability는 기본적으로 'false'로 설정되며, 여기서는 확률 추정치를 수집하기 위해 'true'로 설정한다. 확률값을 얻기 위해 데이터를 섞기 위해 무작위 상태를 '0'으로 설정했다.

Linear SVC 적용

분류 어레이를 만드는 것으로 시작한다. 테스트하는 대로 이 어레이에 점진적으로 추가한다.



```python
C = 10
# Create different classifiers.
classifiers = {
    'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)

}
```


```python
#모델 훈련
n_classifiers = len(classifiers)

for index, (name, classifier) in enumerate(classifiers.items()):
    classifier.fit(X_train, np.ravel(y_train))

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
    print(classification_report(y_test,y_pred))
```

    Accuracy (train) for Linear SVC: 79.4% 
                  precision    recall  f1-score   support
    
         chinese       0.74      0.70      0.72       247
          indian       0.89      0.89      0.89       222
        japanese       0.73      0.76      0.74       237
          korean       0.84      0.78      0.81       243
            thai       0.79      0.85      0.82       250
    
        accuracy                           0.79      1199
       macro avg       0.80      0.80      0.80      1199
    weighted avg       0.79      0.79      0.79      1199


​    

K-Neighbors classifier

K-Neighbors는 ML 방법의 "Neighbors" 계열의 일부로, 지도 학습과 비지도 학습 모두에 사용할 수 있다. 이 방법에서는 데이터에 대한 일반화된 레이블을 예측할 수 있도록 미리 정의된 수의 점이 생성되고 이러한 점 주위에 데이터가 수집된다.

K-Neighbors 분류기 적용하기

이전의 분류기는 좋았고, 데이터도 잘 작동했지만, 아마도 우리는 더 나은 정확도를 얻을 수 있을 것이다. K-Neighbors 분류기를 사용해 보자.

분류기 배열에 줄을 추가한다(Linear SVC 항목 뒤에 쉼표를 추가한다.)


```python
C = 10
# Create different classifiers.
classifiers = {
    'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0),
    'KNN classifier': KNeighborsClassifier(C)
}
```


```python
#모델 훈련
n_classifiers = len(classifiers)

for index, (name, classifier) in enumerate(classifiers.items()):
    classifier.fit(X_train, np.ravel(y_train))

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
    print(classification_report(y_test,y_pred))
```

    Accuracy (train) for Linear SVC: 79.4% 
                  precision    recall  f1-score   support
    
         chinese       0.74      0.70      0.72       247
          indian       0.89      0.89      0.89       222
        japanese       0.73      0.76      0.74       237
          korean       0.84      0.78      0.81       243
            thai       0.79      0.85      0.82       250
    
        accuracy                           0.79      1199
       macro avg       0.80      0.80      0.80      1199
    weighted avg       0.79      0.79      0.79      1199
    
    Accuracy (train) for KNN classifier: 73.9% 
                  precision    recall  f1-score   support
    
         chinese       0.70      0.70      0.70       247
          indian       0.87      0.81      0.84       222
        japanese       0.61      0.84      0.71       237
          korean       0.90      0.57      0.70       243
            thai       0.73      0.79      0.76       250
    
        accuracy                           0.74      1199
       macro avg       0.76      0.74      0.74      1199
    weighted avg       0.76      0.74      0.74      1199


​    

Support Vector Classifier

Support-Vector 분류기는 분류 및 회귀 작업에 사용되는 ML 메서드의 Support-Vector Machine 제품군의 일부이다. SVM은 두 범주 간의 거리를 최대화하기 위해 "공간 내 지점에 훈련 예제를 매핑"한다. 후속 데이터는 해당 범주를 예측할 수 있도록 이 공간에 매핑된다.

Support Vector Classifier 적용하기




```python
C = 10
# Create different classifiers.
classifiers = {
    'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0),
    'KNN classifier': KNeighborsClassifier(C),
    'SVC': SVC()
}
```


```python
#모델 훈련
n_classifiers = len(classifiers)

for index, (name, classifier) in enumerate(classifiers.items()):
    classifier.fit(X_train, np.ravel(y_train))

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
    print(classification_report(y_test,y_pred))
```

    Accuracy (train) for Linear SVC: 79.4% 
                  precision    recall  f1-score   support
    
         chinese       0.74      0.70      0.72       247
          indian       0.89      0.89      0.89       222
        japanese       0.73      0.76      0.74       237
          korean       0.84      0.78      0.81       243
            thai       0.79      0.85      0.82       250
    
        accuracy                           0.79      1199
       macro avg       0.80      0.80      0.80      1199
    weighted avg       0.79      0.79      0.79      1199
    
    Accuracy (train) for KNN classifier: 73.9% 
                  precision    recall  f1-score   support
    
         chinese       0.70      0.70      0.70       247
          indian       0.87      0.81      0.84       222
        japanese       0.61      0.84      0.71       237
          korean       0.90      0.57      0.70       243
            thai       0.73      0.79      0.76       250
    
        accuracy                           0.74      1199
       macro avg       0.76      0.74      0.74      1199
    weighted avg       0.76      0.74      0.74      1199
    
    Accuracy (train) for SVC: 81.8% 
                  precision    recall  f1-score   support
    
         chinese       0.80      0.73      0.77       247
          indian       0.90      0.91      0.90       222
        japanese       0.76      0.82      0.79       237
          korean       0.87      0.78      0.82       243
            thai       0.78      0.86      0.82       250
    
        accuracy                           0.82      1199
       macro avg       0.82      0.82      0.82      1199
    weighted avg       0.82      0.82      0.82      1199


​    

Ensemble Classifiers

지난번 테스트는 꽤 좋은 결과를 갖지만 끝까지 그 길을 따라가자. 앙상블 분류기, 특히 랜덤 포레스트와 AdaBoost를 사용해 보자.


```python
C = 10
# Create different classifiers.
classifiers = {
    'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0),
    'KNN classifier': KNeighborsClassifier(C),
    'SVC': SVC(),
    'RFST': RandomForestClassifier(n_estimators=100),
    'ADA': AdaBoostClassifier(n_estimators=100)
}
```


```python
#모델 훈련
n_classifiers = len(classifiers)

for index, (name, classifier) in enumerate(classifiers.items()):
    classifier.fit(X_train, np.ravel(y_train))

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
    print(classification_report(y_test,y_pred))
```

    Accuracy (train) for Linear SVC: 79.4% 
                  precision    recall  f1-score   support
    
         chinese       0.74      0.70      0.72       247
          indian       0.89      0.89      0.89       222
        japanese       0.73      0.76      0.74       237
          korean       0.84      0.78      0.81       243
            thai       0.79      0.85      0.82       250
    
        accuracy                           0.79      1199
       macro avg       0.80      0.80      0.80      1199
    weighted avg       0.79      0.79      0.79      1199
    
    Accuracy (train) for KNN classifier: 73.9% 
                  precision    recall  f1-score   support
    
         chinese       0.70      0.70      0.70       247
          indian       0.87      0.81      0.84       222
        japanese       0.61      0.84      0.71       237
          korean       0.90      0.57      0.70       243
            thai       0.73      0.79      0.76       250
    
        accuracy                           0.74      1199
       macro avg       0.76      0.74      0.74      1199
    weighted avg       0.76      0.74      0.74      1199
    
    Accuracy (train) for SVC: 81.8% 
                  precision    recall  f1-score   support
    
         chinese       0.80      0.73      0.77       247
          indian       0.90      0.91      0.90       222
        japanese       0.76      0.82      0.79       237
          korean       0.87      0.78      0.82       243
            thai       0.78      0.86      0.82       250
    
        accuracy                           0.82      1199
       macro avg       0.82      0.82      0.82      1199
    weighted avg       0.82      0.82      0.82      1199
    
    Accuracy (train) for RFST: 83.6% 
                  precision    recall  f1-score   support
    
         chinese       0.81      0.79      0.80       247
          indian       0.90      0.91      0.90       222
        japanese       0.80      0.80      0.80       237
          korean       0.86      0.81      0.83       243
            thai       0.81      0.88      0.84       250
    
        accuracy                           0.84      1199
       macro avg       0.84      0.84      0.84      1199
    weighted avg       0.84      0.84      0.84      1199
    
    Accuracy (train) for ADA: 69.0% 
                  precision    recall  f1-score   support
    
         chinese       0.72      0.46      0.56       247
          indian       0.83      0.83      0.83       222
        japanese       0.63      0.58      0.60       237
          korean       0.63      0.84      0.72       243
            thai       0.67      0.75      0.71       250
    
        accuracy                           0.69      1199
       macro avg       0.70      0.69      0.69      1199
    weighted avg       0.69      0.69      0.68      1199


​    

이 머신러닝 메서드은 모델의 품질을 향상시키기 위해 "여러 기본 추정기의 예측을 결합"한다. 이 예에서는 랜덤 트리와 AdaBoost를 사용했다.

- 평균화 방법인 랜덤 포레스트는 과적합을 피하기 위해 무작위성이 주입된 '결정 트리'의 'forest'을 구축한다. n_estimators 파라미터는 트리 수로 설정된다.

- AdaBoost는 분류기를 데이터 집합에 적합시킨 다음 해당 분류기의 복사본을 동일한 데이터 집합에 적합시킨다. 잘못 분류된 항목의 가중치에 초점을 맞추고 다음 분류자가 수정할 적합도를 조정한다.

## 모델 빌드

분류 모델 훈련

첫째, 우리가 사용한 정제된 요리 데이터 세트를 사용하여 분류 모델을 훈련한다.

read_csv()를 사용하여 CSV 파일을 읽어 이전과 동일한 방식으로 데이터를 처리한다.


```python
data = pd.read_csv('cleaned_cuisines.csv')
data.head()
```





  <div id="df-30ec884e-2b80-4af5-87a6-861c6aa2bb70">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>cuisine</th>
      <th>almond</th>
      <th>angelica</th>
      <th>anise</th>
      <th>anise_seed</th>
      <th>apple</th>
      <th>apple_brandy</th>
      <th>apricot</th>
      <th>armagnac</th>
      <th>...</th>
      <th>whiskey</th>
      <th>white_bread</th>
      <th>white_wine</th>
      <th>whole_grain_wheat_flour</th>
      <th>wine</th>
      <th>wood</th>
      <th>yam</th>
      <th>yeast</th>
      <th>yogurt</th>
      <th>zucchini</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>indian</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>indian</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>indian</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>indian</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>indian</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 382 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-30ec884e-2b80-4af5-87a6-861c6aa2bb70')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }
    
    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }
    
    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }
    
    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-30ec884e-2b80-4af5-87a6-861c6aa2bb70 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';
    
        async function convertToInteractive(key) {
          const element = document.querySelector('#df-30ec884e-2b80-4af5-87a6-861c6aa2bb70');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;
    
          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
#불필요한 첫 두 열 삭제하고 나머지만 X로 저장
X = data.iloc[:,2:]
X.head()
```





  <div id="df-3fedfe63-ecb8-4920-8ca8-b0c6c3142441">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>almond</th>
      <th>angelica</th>
      <th>anise</th>
      <th>anise_seed</th>
      <th>apple</th>
      <th>apple_brandy</th>
      <th>apricot</th>
      <th>armagnac</th>
      <th>artemisia</th>
      <th>artichoke</th>
      <th>...</th>
      <th>whiskey</th>
      <th>white_bread</th>
      <th>white_wine</th>
      <th>whole_grain_wheat_flour</th>
      <th>wine</th>
      <th>wood</th>
      <th>yam</th>
      <th>yeast</th>
      <th>yogurt</th>
      <th>zucchini</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 380 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-3fedfe63-ecb8-4920-8ca8-b0c6c3142441')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }
    
    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }
    
    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }
    
    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-3fedfe63-ecb8-4920-8ca8-b0c6c3142441 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';
    
        async function convertToInteractive(key) {
          const element = document.querySelector('#df-3fedfe63-ecb8-4920-8ca8-b0c6c3142441');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;
    
          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
#cuisine을 레이블로 사용하기 위해 y로 저장
y = data[['cuisine']]
y.head()
```





  <div id="df-a55cfba9-f785-41ac-b023-07a791c4e366">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cuisine</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>indian</td>
    </tr>
    <tr>
      <th>1</th>
      <td>indian</td>
    </tr>
    <tr>
      <th>2</th>
      <td>indian</td>
    </tr>
    <tr>
      <th>3</th>
      <td>indian</td>
    </tr>
    <tr>
      <th>4</th>
      <td>indian</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-a55cfba9-f785-41ac-b023-07a791c4e366')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }
    
    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }
    
    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }
    
    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-a55cfba9-f785-41ac-b023-07a791c4e366 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';
    
        async function convertToInteractive(key) {
          const element = document.querySelector('#df-a55cfba9-f785-41ac-b023-07a791c4e366');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;
    
          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




훈련 하기

정확도가 가장 높은 SVC 라이브러리를 사용할 것이다.
Scikit-learn에서 적절한 라이브러리를 가져온다.


```python
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report
```


```python
#데이터셋 분할하기

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
```


```python
#SVC 분류 모델 구축

model = SVC(kernel='linear', C=10, probability=True,random_state=0)
model.fit(X_train,y_train.values.ravel())
```




    SVC(C=10, kernel='linear', probability=True, random_state=0)




```python
#predict()를 호출하여 모델 테스트
y_pred = model.predict(X_test)
```


```python
#모델 성능 확인

print(classification_report(y_test,y_pred))
```

                  precision    recall  f1-score   support
    
         chinese       0.69      0.68      0.68       229
          indian       0.89      0.88      0.89       228
        japanese       0.71      0.78      0.74       226
          korean       0.88      0.77      0.82       257
            thai       0.79      0.83      0.81       259
    
        accuracy                           0.79      1199
       macro avg       0.79      0.79      0.79      1199
    weighted avg       0.79      0.79      0.79      1199


​    