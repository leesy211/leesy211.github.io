# 머신러닝 모델 사용하여 Web App 만들기

### 1. 데이터 정제

데이터 적재


```python
import pandas as pd
import numpy as np

ufos = pd.read_csv('https://raw.githubusercontent.com/codingalzi/ML-For-Beginners/main/3-Web-App/1-Web-App/data/ufos.csv')
ufos.head()
```





  <div id="df-56fae63c-6ef7-4d79-9f44-7c448403134a">
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
      <th>datetime</th>
      <th>city</th>
      <th>state</th>
      <th>country</th>
      <th>shape</th>
      <th>duration (seconds)</th>
      <th>duration (hours/min)</th>
      <th>comments</th>
      <th>date posted</th>
      <th>latitude</th>
      <th>longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10/10/1949 20:30</td>
      <td>san marcos</td>
      <td>tx</td>
      <td>us</td>
      <td>cylinder</td>
      <td>2700.0</td>
      <td>45 minutes</td>
      <td>This event took place in early fall around 194...</td>
      <td>4/27/2004</td>
      <td>29.883056</td>
      <td>-97.941111</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10/10/1949 21:00</td>
      <td>lackland afb</td>
      <td>tx</td>
      <td>NaN</td>
      <td>light</td>
      <td>7200.0</td>
      <td>1-2 hrs</td>
      <td>1949 Lackland AFB&amp;#44 TX.  Lights racing acros...</td>
      <td>12/16/2005</td>
      <td>29.384210</td>
      <td>-98.581082</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10/10/1955 17:00</td>
      <td>chester (uk/england)</td>
      <td>NaN</td>
      <td>gb</td>
      <td>circle</td>
      <td>20.0</td>
      <td>20 seconds</td>
      <td>Green/Orange circular disc over Chester&amp;#44 En...</td>
      <td>1/21/2008</td>
      <td>53.200000</td>
      <td>-2.916667</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10/10/1956 21:00</td>
      <td>edna</td>
      <td>tx</td>
      <td>us</td>
      <td>circle</td>
      <td>20.0</td>
      <td>1/2 hour</td>
      <td>My older brother and twin sister were leaving ...</td>
      <td>1/17/2004</td>
      <td>28.978333</td>
      <td>-96.645833</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10/10/1960 20:00</td>
      <td>kaneohe</td>
      <td>hi</td>
      <td>us</td>
      <td>light</td>
      <td>900.0</td>
      <td>15 minutes</td>
      <td>AS a Marine 1st Lt. flying an FJ4B fighter/att...</td>
      <td>1/22/2004</td>
      <td>21.418056</td>
      <td>-157.803611</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-56fae63c-6ef7-4d79-9f44-7c448403134a')"
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
          document.querySelector('#df-56fae63c-6ef7-4d79-9f44-7c448403134a button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';
    
        async function convertToInteractive(key) {
          const element = document.querySelector('#df-56fae63c-6ef7-4d79-9f44-7c448403134a');
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




ufos.csv 스프레드시트에는 목격된 city, state 와 country, 오브젝트의 shaper과 latitude 및 longitude 열이 포함되어 있다.

칸의 인덱스를 새로운 이름으로 변경하고 데이터프레임으로 변환한다.

그리고 Country의 값이 unique 한 값인지 확인한다.


```python
ufos = pd.DataFrame({'Seconds' : ufos['duration (seconds)'], 'Country': ufos['country'], 'Latitude': ufos['latitude'], 'Longitude': ufos['longitude']})
ufos.Country.unique()
```




    array(['us', nan, 'gb', 'ca', 'au', 'de'], dtype=object)



null 값을 drop하고 1-60초 사이 목격만 가져와서 처리할 데이터의 양을 줄인다.


```python
ufos.dropna(inplace=True)

ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]

ufos.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 25863 entries, 2 to 80330
    Data columns (total 4 columns):
     #   Column     Non-Null Count  Dtype  
    ---  ------     --------------  -----  
     0   Seconds    25863 non-null  float64
     1   Country    25863 non-null  object 
     2   Latitude   25863 non-null  float64
     3   Longitude  25863 non-null  float64
    dtypes: float64(3), object(1)
    memory usage: 1010.3+ KB


Scikit-learn의 LabelEncoder 라이브러리를 Import해서 국가의 텍스트 값을 숫자로 변환한다.
- LabelEncoder는 데이터를 알파벳 순서로 인코드한다.


```python
from sklearn.preprocessing import LabelEncoder

ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])

ufos.head()
```





  <div id="df-c6d4bd2f-5c94-4eb8-945f-46346cc22d5e">
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
      <th>Seconds</th>
      <th>Country</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>20.0</td>
      <td>3</td>
      <td>53.200000</td>
      <td>-2.916667</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20.0</td>
      <td>4</td>
      <td>28.978333</td>
      <td>-96.645833</td>
    </tr>
    <tr>
      <th>14</th>
      <td>30.0</td>
      <td>4</td>
      <td>35.823889</td>
      <td>-80.253611</td>
    </tr>
    <tr>
      <th>23</th>
      <td>60.0</td>
      <td>4</td>
      <td>45.582778</td>
      <td>-122.352222</td>
    </tr>
    <tr>
      <th>24</th>
      <td>3.0</td>
      <td>3</td>
      <td>51.783333</td>
      <td>-0.783333</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-c6d4bd2f-5c94-4eb8-945f-46346cc22d5e')"
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
          document.querySelector('#df-c6d4bd2f-5c94-4eb8-945f-46346cc22d5e button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';
    
        async function convertToInteractive(key) {
          const element = document.querySelector('#df-c6d4bd2f-5c94-4eb8-945f-46346cc22d5e');
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




### 2. 모델 만들기


데이터를 훈련하고 테스트할 그룹으로 나누어서 모델을 훈련할 준비를 한다.

입력 데이터로 훈련할 특성 3가지를 선택하고, 출력 데이터는 Country이다.

즉, Seconds, Latitude 와 Longitude를 입력하면 국가 id로 반환된다.


```python
from sklearn.model_selection import train_test_split

Selected_features = ['Seconds','Latitude','Longitude']

X = ufos[Selected_features]
y = ufos['Country']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

logistic regression을 사용해서 모델을 훈련한다.


```python
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(classification_report(y_test, predictions))
print('Predicted labels: ', predictions)
print('Accuracy: ', accuracy_score(y_test, predictions))
```

                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00        41
               1       0.83      0.23      0.36       250
               2       1.00      1.00      1.00         8
               3       1.00      1.00      1.00       131
               4       0.96      1.00      0.98      4743
    
        accuracy                           0.96      5173
       macro avg       0.96      0.85      0.87      5173
    weighted avg       0.96      0.96      0.95      5173
    
    Predicted labels:  [4 4 4 ... 3 4 4]
    Accuracy:  0.960371157935434


    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_logistic.py:818: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG,


Country 와 Latitude/Longitude가 상관 관계있어서, 정확도 (약 96%) 가 나쁘지 않다.

만든 모델은 Latitude 와 Longitude에서 Country를 알 수 있어야 하므로 매우 혁신적이지 않지만, 정리하면서, 뽑은 원본 데이터에서 훈련을 해보고 웹 앱에서 모델을 쓰기에 좋은 연습이다.

### 모델 'pickle'하기

pickled 되면, pickled 모델을 불러와서 Seconds, Latitude 와 Longitude 값이 포함된 샘플 데이터 배열을 대상으로 테스트한다.


```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('/content/ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

    [1]


    /usr/local/lib/python3.7/dist-packages/sklearn/base.py:451: UserWarning: X does not have valid feature names, but LogisticRegression was fitted with feature names
      "X does not have valid feature names, but"



```python
cd web-app
```

    /content/web-app



```python
!pip install -r requirements.txt
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Requirement already satisfied: scikit-learn in /usr/local/lib/python3.7/dist-packages (from -r requirements.txt (line 1)) (1.0.2)
    Requirement already satisfied: pandas in /usr/local/lib/python3.7/dist-packages (from -r requirements.txt (line 2)) (1.3.5)
    Requirement already satisfied: numpy in /usr/local/lib/python3.7/dist-packages (from -r requirements.txt (line 3)) (1.21.6)
    Requirement already satisfied: flask in /usr/local/lib/python3.7/dist-packages (from -r requirements.txt (line 4)) (0.12.2)
    Requirement already satisfied: scipy>=1.1.0 in /usr/local/lib/python3.7/dist-packages (from scikit-learn->-r requirements.txt (line 1)) (1.4.1)
    Requirement already satisfied: joblib>=0.11 in /usr/local/lib/python3.7/dist-packages (from scikit-learn->-r requirements.txt (line 1)) (1.1.0)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /usr/local/lib/python3.7/dist-packages (from scikit-learn->-r requirements.txt (line 1)) (3.1.0)
    Requirement already satisfied: pytz>=2017.3 in /usr/local/lib/python3.7/dist-packages (from pandas->-r requirements.txt (line 2)) (2022.1)
    Requirement already satisfied: python-dateutil>=2.7.3 in /usr/local/lib/python3.7/dist-packages (from pandas->-r requirements.txt (line 2)) (2.8.2)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.7/dist-packages (from python-dateutil>=2.7.3->pandas->-r requirements.txt (line 2)) (1.15.0)
    Requirement already satisfied: itsdangerous>=0.21 in /usr/local/lib/python3.7/dist-packages (from flask->-r requirements.txt (line 4)) (1.1.0)
    Requirement already satisfied: click>=2.0 in /usr/local/lib/python3.7/dist-packages (from flask->-r requirements.txt (line 4)) (7.1.2)
    Requirement already satisfied: Werkzeug>=0.7 in /usr/local/lib/python3.7/dist-packages (from flask->-r requirements.txt (line 4)) (1.0.1)
    Requirement already satisfied: Jinja2>=2.4 in /usr/local/lib/python3.7/dist-packages (from flask->-r requirements.txt (line 4)) (2.11.3)
    Requirement already satisfied: MarkupSafe>=0.23 in /usr/local/lib/python3.7/dist-packages (from Jinja2>=2.4->flask->-r requirements.txt (line 4)) (2.0.1)



```python
!pip install flask==0.12.2
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting flask==0.12.2
      Downloading Flask-0.12.2-py2.py3-none-any.whl (83 kB)
    [K     |████████████████████████████████| 83 kB 714 kB/s 
    [?25hRequirement already satisfied: itsdangerous>=0.21 in /usr/local/lib/python3.7/dist-packages (from flask==0.12.2) (1.1.0)
    Requirement already satisfied: click>=2.0 in /usr/local/lib/python3.7/dist-packages (from flask==0.12.2) (7.1.2)
    Requirement already satisfied: Werkzeug>=0.7 in /usr/local/lib/python3.7/dist-packages (from flask==0.12.2) (1.0.1)
    Requirement already satisfied: Jinja2>=2.4 in /usr/local/lib/python3.7/dist-packages (from flask==0.12.2) (2.11.3)
    Requirement already satisfied: MarkupSafe>=0.23 in /usr/local/lib/python3.7/dist-packages (from Jinja2>=2.4->flask==0.12.2) (2.0.1)
    Installing collected packages: flask
      Attempting uninstall: flask
        Found existing installation: Flask 1.1.4
        Uninstalling Flask-1.1.4:
          Successfully uninstalled Flask-1.1.4
    Successfully installed flask-0.12.2





```python
import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open("/content/ufo-model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]

    countries = ["Australia", "Canada", "Germany", "UK", "US"]

    return render_template(
        "index.html", prediction_text="Likely country: {}".format(countries[output])
    )


if __name__ == "__main__":
    app.run(debug=True)
```

     * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
     * Restarting with stat



    An exception has occurred, use %tb to see the full traceback.


    SystemExit: 1



    /usr/local/lib/python3.7/dist-packages/IPython/core/interactiveshell.py:2890: UserWarning: To exit: use 'exit', 'quit', or Ctrl-D.
      warn("To exit: use 'exit', 'quit', or Ctrl-D.", stacklevel=1)

http://127.0.0.1:5000 링크에 들어가면

![](https://github.com/leesy211/leesy211.github.io/blob/master/assets/images/UFO-webapp.JPG?raw=true)

다음과 같은 웹이 만들어진 것을 확인할 수 있다.

다음과 같이 Seconds, Latitude 와 Longitude를 입력해보았다.

![](https://github.com/leesy211/leesy211.github.io/blob/master/assets/images/ufo-input.JPG?raw=true)

그 입력값에 해당하는 UFO가 발견된 해당국가 id를 예측하여 알려준다.



![](https://github.com/leesy211/leesy211.github.io/blob/master/assets/images/ufo-predict.JPG?raw=true)

UK 영국으로 정확하게 예측한 것을 확인할 수 있다.
