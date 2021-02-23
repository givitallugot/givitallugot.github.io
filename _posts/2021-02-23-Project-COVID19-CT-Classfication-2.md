---
layout: post
title:  "[Python] Kaggle COVID-19 CT image Classification 2 (폐 CT로 코로나 감염 여부 분류) with CNN, Inception V3 (딥러닝 프로젝트 매뉴얼 2)"
date:   2021-02-23 20:14:13
categories: [Deep Learning]
comments: true
---
<br>
[Kaggle COVID-19 CT 이미지][kaggle-data]의 분류를 딥러닝으로 수행한 프로젝트입니다. CNN 기반으로 코로나 감염 여부를 분류하는 것이 목적이며, 원래7개의 모델 `Logistic Regression`, `Decision Tree`, `Random Forest`, `XGBoost`, `LightGBM`, `KNN`, `SVC` 을 사용했습니다. 독성인지 아닌지 구별하는 이진 분류 문제이며, X 변수는 모두 범주형 변수로 이루어져있습니다. Kaggle 사이트에 많은 notebook이 올라와있어서 머신러닝 공부 시 수행해보기 좋습니다.
<br><br><br>

![slide-1](/!contents_plot/2021-02-23-covid19-1.jpeg){: width="60%"}
<br>

{% highlight Python %}

{% endhighlight %}
<br><br><br>

![slide-2](/!contents_plot/2021-02-23-covid19-2.jpeg){: width="60%"}
<br>

결측치를 확인하고, 결측 비율이 30%에 가까운 ‘stalk-root’ 변수를 제거했다. 또한 단일 카테고리인 ‘veil-type’ 변수 또한 제거했다.
<br>

{% highlight Python %}

{% endhighlight %}
<br><br><br>

![slide-3](/!contents_plot/2021-02-23-covid19-3.jpeg){: width="60%"}
<br>

여기서 중요 변수를 탐색해보기 위해 두 가지 방법을 사용했다. ~~두 방법이 최선이라고 것은 아니며 그냥 이런 방법도 가능하다는 것!~~
* ① Y와 상관관계를 확인한다. 이때, 모든 X 변수는 범주형이지만, 변수 간 상관성을 짐작해보기 위해서 Label Encoder를 이용해 숫자값을 가지도록 변환하고 상관계수로 선형 관계성을 확인해본다. <br>
* ② Extra Tree 적합 후 중요변수를 그려보았다.
<br>

{% highlight Python %}

{% endhighlight %}
<br><br><br>

![slide-4](/!contents_plot/2021-02-23-covid19-4.jpeg){: width="60%"}
<br>

EDA와 상관계수 등을 살펴보며 얻게된 Y에 따른 특징을 표로 정리한 결과다. 표에 나타난 것과 마찬가지로 독버섯은 식용버섯과 달리 주름 크기가 좁고 악취 냄새가 나며 멍이 없고 회색, 황토색, 초콜렛색을 띄는 경우가 많다는 것을 알 수 있다.
<br><br><br>

![slide-5](/!contents_plot/22021-02-23-covid19-5.jpeg){: width="60%"}
<br>

훈련 셋의 비율을 80%로 모델을 적합했다. 모두 5-fold CV Accuracy와 Test Accuracy로 비교를 진행하고, 먼저 간단하고 기본적인 Logistic, Decision Tree, Random Forest를 적합한 결과, 트리 기반의 모델이 굉장히 잘 적합됨을 확인할 수 있다.
<br>

그 전에 먼저 평가 기준을 위한 함수 작성 코드이다. 다음과 같이 `CV_check()`, `plt_roc_curve()`, `ROC_check()`, `error_result_check()` 함수를 커스터마이징하여 사용해보았다.


{% highlight Python %}

{% endhighlight %}
<br>

이 함수들을 이용하여 fitting 후 결과 확인에 사용했다.
<br>

{% highlight Python %}

{% endhighlight %}
<br><br><br>

![slide-6](/!contents_plot/2021-02-23-covid19-6.jpeg){: width="60%"}
<br>

앞에서 매우 높은 정확도를 보였기 때문에, 계산량을 줄이고 모델을 간결하게 만들어보고자 축차적으로 줄였다. 그 결과, 5개 `‘gill-size’, ‘ordor’, ‘bruises’, ‘ring-type’, ‘spore-print-color’` 변수로 충분히 잘 적합됨을 확인할 수 있다. 사용한 모델은 `Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, KNN, SVC`이며, 이 7개 모델 비교한 결과이다.
<br>

{% highlight Python %}

{% endhighlight %}
<br><br><br>

![slide-7](/!contents_plot/2021-02-23-covid19-7.jpeg){: width="60%"}
<br>

각 모델의 `Accuracy`, `F1-socre`, `Recall`을 test 셋에서 비교해본 결과이다. `Decision Tree`, `Random Forest`, `LightGBM`의 성능이 우수한 것을 확인할 수 있으며, 트리 기반에 세 모델을 ~~(내 마음속의 ㅋㅋ)~~ 최적 모형으로 선택했다.
<br>

{% highlight Python %}

{% endhighlight %}
<br>

피피티 디자인 저작권은 @PPTBIZCAM

[kaggle-data]: https://www.kaggle.com/engesraahassan/covid19-ct-image