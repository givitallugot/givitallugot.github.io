---
layout: post
title:  "[Python/Jupyter] 전처리 파이프라인 만들기 2 (연속형 변수) with sklearn pipeline processing StandardScaler OneHotEncoder"
date:   2020-03-15 12:33:17
categories: [Preprocessing]
comments: true
---
<br>
> 1. ~~`waterfront`, `yr_renovated` 범주형으로 변환 및 인코딩~~
> 2. **보조변수 `yr_sold (date)`와 연식 `h_age` 변수 추가**
> 3. **연속형 변수 `min-max` 변환**
> 4. ~~`id, zipcode, date` 제외~~  

<br><br>

저번 포스팅에서 정리된 전처리 아이디어 중 이번 포스팅에서는 2.와 3.에 해당하는 연속형 변수 파이프라인을 만들어봅시다. 
<br><br>

저번 포스팅 내용을 바로 확인하고 싶다면 클릭하세요!  <br><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [>> 전처리 파이프라인 만들기 1][next-1]
<br><br><br><br>

### 연속형 변수를 위한 파이프라인
연속형 파이프라인을 생성하기 전에 훈련셋의 X와 Y를 housing과 housing_label로 분리합니다.

{% highlight Python %}
housing = train_set.drop("price", axis=1) # X
housing_labels = train_set["price"].copy() # Y
{% endhighlight %}

* housing에는 훈련셋이 담겨있습니다.  
<br><br><br><br>

사이킷런은 `pandas`의 `data frame`을 바로 사용할 수 없기 때문에 파이프라인에서 `array`를 사용하게 됩니다. 또한, 연속형 변수와 범주형 변수를 나눠서 처리할 것이므로 연속형 변수만 선택하는 전처리가 필요합니다. 이들을 쉽게 노드라고 부르고 먼저 `data frame`에서 인자로 받은 변수를 선택하고 배열을 반환하는 클래스를 만들어 봅시다 **>_<**

{% highlight Python %}
from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
{% endhighlight %}  
<br><br><br><br>

이 클래스는 연속형 변수 선택 뿐만 아니라 범주형 변수 선택에도 사용할 수 있습니다. 따라서 어떤 형태이든 파이프라인의 가장 처음 노드가 되어야 합니다. 이 클래스의 이름은 `DataFrameSelector`로 이를 추후에 최종 파이프라인에 다음과 같이 추가하게 됩니다.  

{% highlight Python %}
num_attribs = list(housing.drop(["id", "zipcode", "waterfront"], axis=1)) # exclude id, zipcode

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs))
    ])
{% endhighlight %}  
<br><br><br><br>

`num_attribs` 변수에 선택할 연속형 변수명을 나열합니다. 머신러닝에 필요없는 `id`, `zipcode`과 연속형 변수가 아닌 `waterfront`를 제외한 변수명을 `num_attribs`에 넣었습니다. `yr_renovated`도 범주형 변수로서 변환하고자 하나 개조 연도 정보를 연식 `h_age` 변수 처리에 먼저 이용해야 하므로 연속형 변수로도 선택합니다.  
<br><br><br><br>

다음으로 가장 복잡한 `CombinedAttributesAdder` 클래스를 생성해봅시다. 이 클래스를 통해 2. 전처리를 수행합니다. 코드를 조각으로 나눠서 확인해봅시다. `DataFrameSelector`로 거쳐 반환된 `array`를 X로 받고 `yr_sold`와 `h_age`를 먼저 초기화합니다. 다음으로 `date`의 연도인 맨 앞 네 자리를 추출해서 `yr_sold`에 저장합니다.

{% highlight Python %}
yr_sold = np.zeros(len(X))
h_age = np.zeros(len(X))
date = X[:, date_ix]

for i in range(len(date)):
    yr_sold[i] = date[i][0:4]
{% endhighlight %}  
<br><br><br><br>

두 번째 for문에서 집을 개조한 적이 없다면 거래 연도인 `yr_sold`에서 건축 연도인 `yr_built`를 빼서 연식을 구합니다. 집을 개조한 적이 있다면, 즉 `yr_renovated`가 0이 아니라면 `yr_sold`에서 개조 연도인 `yr_renovated`를 빼서 연식을 구합니다. 이는 다음과 같은 코드가 됩니다.

{% highlight Python %}
for i in range(len(X)):
    if X[i, yr_renovated_ix] == 0:
        h_age[i] = yr_sold[i] - X[i, yr_built_ix]
    else:
        h_age[i] = yr_sold[i] - X[i, yr_renovated_ix]
{% endhighlight %}  
<br><br><br><br>

이제 목적을 이뤘으니 `date`와 `yr_renovated`변수를 제외하고 `h_age`를 추가한 X 배열을 반환하면 되겠지요? 따라서 최종 `CombinedAttributesAdder` 클래스를는 다음과 같이 작성할 수 있습니다.

{% highlight Python %}
from sklearn.base import BaseEstimator, TransformerMixin

date_ix, yr_built_ix, yr_renovated_ix = 0, 11, 12

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, h_age = True): # no *args or **kargs
        self.h_age = h_age
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        yr_sold = np.zeros(len(X))
        h_age = np.zeros(len(X))

        date = X[:, date_ix]
        for i in range(len(date)):
            yr_sold[i] = date[i][0:4]
        
        for i in range(len(X)):
            if X[i, yr_renovated_ix] == 0:
                h_age[i] = yr_sold[i] - X[i, yr_built_ix]
            else:
                h_age[i] = yr_sold[i] - X[i, yr_renovated_ix]
        
        return np.concatenate((np.delete(X, [0, 12], axis=1), h_age.astype(int)[:,None]), axis=1)
{% endhighlight %}  
<br><br><br><br>

다음으로 3. 전처리를 수행하려고 합니다. 다만 모든 연속형 변수의 `min-max` 변환은 따로 클래스를 생성하지 않아도 됩니다. 왜냐하면 이미 구현된 `StandardScaler` 클래스가 있기 때문이죠. 이를 사용하면 매우 편리합니다.  
<br><br><br><br>

이렇게 연속형 변수의 전처리 노드를 모두 생성했습니다. 이제 이 노드들로 구성된 파이프라인을 구현하면 됩니다. 앞에서 잠깐 확인했던 것처럼 `num_pipeline`을 정의합니다. 사이킷런의 `Pipeline`을 이용하여 클래스들과 각각에 맞는 이름(`selector`, `attribs_adder`, `std_scaler`)을 적고 노드를 연결한다는 느낌으로 연속형 파이프라인이 구성됩니다. 

{% highlight Python %}
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_attribs = list(housing.drop(["id", "zipcode", "waterfront"], axis=1)) # exclude id, zipcode

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler())
    ])
{% endhighlight %}  
<br><br><br><br>

연속형 변수를 위한 전처리 파이프라인이 완성되었습니다. 이제 범주형 전처리 파이프라인도 처리해야겠죠? 다음 포스팅에서 범주형 전처리 파이프라인과 최종 파이프라인을 구현해보도록 할게요!  
<br><br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [>> 전처리 파이프라인 만들기 3 (범주형 변수 및 최종 합치기)][next-3]


[next-1]: https://givitallugot.github.io/articles/2020-03/Python-preprocessing-1-pipe1
[next-3]: https://givitallugot.github.io/articles/2020-03/Python-preprocessing-3-pipe3

