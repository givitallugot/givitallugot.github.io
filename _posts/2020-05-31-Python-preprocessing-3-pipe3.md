---
layout: post
title:  "[Python] 전처리 파이프라인 만들기 3 (범주형 변수 및 최종 합치기) with sklearn pipeline processing StandardScaler OneHotEncoder"
date:   2020-05-31 15:10:48
categories: [Preprocessing]
comments: true
---
<br>
> 1. **`waterfront`, `yr_renovated` 범주형으로 변환 및 인코딩**
> 2. ~~보조변수 `yr_sold (date)`와 연식 `h_age` 변수 추가~~
> 3. ~~연속형 변수 `min-max` 변환~~
> 4. **`id, zipcode, date` 제외**

<br><br>

저번 포스팅에서 연속형 변수의 전처리까지 수행했고, 이번에는 1.에 해당하는 범주형 전처리와 최종 파이프라인을 구현해보도록 합시다.
<br><br>

이전 포스팅 내용을 바로 확인하고 싶다면 클릭하세요!  <br><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [>> 전처리 파이프라인 만들기 1][next-1]<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [>> 전처리 파이프라인 만들기 2][next-2]

<br><br><br><br>

### 범주형 변수를 위한 파이프라인
연식 `h_age`을 구할 때 사용되었던 `yr_renovated`를 이번에는 개조된 경우 **1**, 개조되지 않은 경우 **0**인 범주형 설명변수로서 변경하고자 합니다. 현재로서는 개조된 경우 **1**이 아닌 네 자리 개조 연도가 들어가있기 때문에 int형 변수로 인식됩니다. 이를 먼저 0과 1로 표현하기 위해 다음과 같은 `LevelChanger` 클래스를 생성합니다.

{% highlight Python %}
from sklearn.base import BaseEstimator, TransformerMixin

class LevelChanger(BaseEstimator, TransformerMixin):
    def fit(self, C, y=None):
        return self  # nothing else to do
    def transform(self, C, y=None):
        
        for i in range(len(C)):
            if C[i,1] != 0:
                C[i,1] = 1
        
        return C
{% endhighlight %}  
<br><br><br><br>

즉, **0**이 아니라면 값을 1로 변경하는 노드를 추가했습니다. `housing` 데이터셋에서 범주형 설명변수는 `binary`형인 `waterfront`와 `yr_renovated` 뿐이기 때문에 추가적인 변환은 필요 없습니다. 이제 인코딩을 진행해야 하는데, 만약 회귀분석을 이용한다면 이제 진행할 인코딩은 건너뛰어도 됩니다. 
<br><br><br><br>


회귀분석에서는 **# of levels - 1**로 범주형 설명변수 개수를 지정합니다. 그러나 머신러닝에서는 **# of levels**로 설명변수의 개수를 지정하는 경우가 있는데, 예를 들어 `renovated` 한 변수에서 `yes`와 `no`를 **1**과 **0**으로 구분하는 것이 아닌, `renovated_yes`, `renovated_no` 두 변수를 만들어서 각 변수가 `yes` 또는 `no`를 의미하도록 하는 것입니다. 이를 위한 인코딩은 사이킷런의 `OneHotEncoder`를 이용하면 됩니다.
<br><br><br><br>

이제 범주형 파이프라인 `cat_pipeline`을 다음과 같이 정의합니다. 연속형 파이프라인 'num_pipeline'과 비슷하게 사이킷런의 `Pipeline`을 이용하여 클래스들과 각각에 맞는 이름(`selector`, `level`, `cat_encoder`)을 적고 노드를 연결합니다. 이때 `Pipeline`은 연속된 변환을 순서대로 처리할 수 있도록 돕는 클래스입니다.

{% highlight Python %}
from sklearn.preprocessing import OneHotEncoder

cat_attribs = ["waterfront", "yr_renovated"]

cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('level', LevelChanger()),
        ('cat_encoder', OneHotEncoder(categories='auto', sparse=False))
    ])
{% endhighlight %}  
<br><br><br><br>

위의 코드를 보면 연속형 파이프라인에서 사용했던 `selector`를 마찬가지로 이용하여 범주형 파이프라인에서 처리할 설명변수 `cat_attribs`를 인자로 넘겨 변환된 배열로 반환받습니다. 그리고 연결된 두 노드를 통해 인코딩까지 수행합니다.
<br><br><br><br>

### 최종 파이프라인
이쯤에서 파이프라인의 전체적인 구조를 한번 확인해보겠습니다. 지금까지 글로 설명해서 복잡해보일 수 있지만 다음과 같이 도식화하면 매우 간단해보입니다. 또한 이렇게 재사용 가능한 파이프라인을 생성하여 전처리를 수행하는 것이 얼마나 효율적인지 알 수 있습니다.

![proces](/!contents_plot/2020-05-31-1.png){: width="50%"}  
<br><br><br>

이제 두 파이프라인을 정의했으니 `housing`의 `train set`으로 확인을 해봐야겠죠? 포스팅에서는 순어에 맞춰 파이프라인을 모두 정의한 다음 최종 데이터셋 변환을 실행했으나, 각 파이프라인의 사실 클래스를 하나씩 추가할 때마다 제대로 수행되는지 계속 확인해봐야 합니다. 나중에 한번에 실행할 경우 에러를 찾고 코드를 변경하는데 오랜 시간이 걸리니 미리 확인은 필수입니다.


{% highlight Python %}
from sklearn.compose import ColumnTransformer

full_pipeline = ColumnTransformer([
        ("num_pipeline", num_pipeline, num_attribs),
        ("cat_pipeline", cat_pipeline, cat_attribs),
    ])
{% endhighlight %}  
<br><br><br><br>

먼저 `num_pipeline`과 `cat_pipeline`을 연결한 `full_pipeline`을 정의합니다. `num_attribs`와 `cat_attribs`는 앞에서 정의한 변수 이름이 들어갑니다. 다음으로 설명변수와 반응변수를 `housing`과 `housing_labels`로 나누고, 파이프라인은 `train_set`의 설명변수만 있는 `housing`만 넣고 수행합니다. 나중에 모델 생성하고 이를 테스트할 때는 물론 `test_set`으로 이 파이프라인을 돌린 후 모델에 넣고 진단을 하면 됩니다.

{% highlight Python %}
housing = train_set.drop("price", axis=1) # X
housing_labels = train_set["price"].copy() # Y
{% endhighlight %}  
<br><br><br><br>

{% highlight Python %}
housing_full = full_pipeline.fit_transform(housing)
housing_full
{% endhighlight %}  
<br>

![housing_full](/!contents_plot/2020-05-31-2.png){: width="50%"}  
<br><br><br>

`housing_full`은 배열로 반환됩니다. 원하는 대로 결과가 반환되었다면 성공입니다. 마지막으로 `data.frame`으로 변환 후 이번 포스팅을 마치도록 하겠습니다.

{% highlight Python %}
housing_tr = pd.DataFrame(
    housing_full, 
    columns=list(housing.drop(["id", "date", "waterfront", "yr_renovated", "zipcode"], axis=1)) + ["h_age"] + ["watf_no"] + ["watf_yes"] + ["reno_no"] + ["reno_yes"])
housing_tr.head()
{% endhighlight %} 
<br>

![housing_tr](/!contents_plot/2020-05-31-3.png){: width="120%"}  
<br><br><br>


[next-1]: https://givitallugot.github.io/articles/2020-03/Python-preprocessing-1-pipe1
[next-2]: https://givitallugot.github.io/articles/2020-03/Python-preprocessing-2-pipe2