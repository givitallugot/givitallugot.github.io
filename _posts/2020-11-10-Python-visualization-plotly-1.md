---
layout: post
title:  "[Python] Hover기능을 이용한 인터렉티브한 iris 데이터 산점도 시각화 with plotly"
date:   2020-11-09 15:52:20
categories: [Visualization]
comments: true
---
<br>
이번 포스팅에서는 **plotly 라이브러리**를 활용해 특정 포인트 위로 마우스 커서를 가져갈 때 정보가 보이는 산점도를 그려보도록 하겠습니다. `matplotlib`과 `seaborn`과 달리 hover 기능이 가능해서 해당 library를 사용해보게 되었는데, 추가적으로 plot의 확대와 축소, 3차원 회전 등이 자유롭게 가능합니다. 포스트와 [plotly-site][plt-site]를 참고하면 더욱 interactive 한 시각화를 시도해볼 수 있습니다.

<div class="plotly-div"><iframe src="/!contents_plot/2020-11-09-4.html"
    scrolling="no"
    width="500"
    height="400"
    frameborder="0" allowfullscreen>
</iframe></div>
<br><br><br><br>

가장 간단한 iris 데이터를 활용할 것인데, ~~iris 데이터 분류를 위해 중요한 변수는 petal width와 petal length라는 사실은 이미 안다고 가정하고~~ 해당 두 변수와 target을 이용하여 2차원 산점도를 plotly로 그려보도록 하겠습니다.
<br>

{% highlight Python %}
# iris 불러오기
from sklearn.datasets import load_iris
iris = load_iris()

# dataframe 변환 & target 설정
import numpy as np
import pandas as pd

iris = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])
iris['target'] = iris['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'}) 
{% endhighlight %}
<br><br><br><br>

x는 `petal length`, y는 `petal width`로 설정하고, color와 symbol을 `target`으로 설정해 `target`별 위치를 파악할 수 있는 그림을 그려보았습니다. 기본적으로 x, y축의 값이 hover에 출력되며, 추가적으로 `target`을 추가해서 붓꽃의 종류를 표시했습니다.
<br>

{% highlight Python %}
import plotly.express as px

fig = px.scatter(iris, x='petal length (cm)', y='petal width (cm)', hover_data=['target'], color='target', symbol='target')
fig.update_layout(
    width=600,
    height=500)
fig.show()
{% endhighlight %}

<div class="plotly-div"><iframe src="/!contents_plot/2020-11-09-1.html"
    scrolling="no"
    width="500"
    height="400"
    frameborder="0" allowfullscreen>
</iframe></div>
<br><br><br><br>

plot을 조금 더 발전시켜봅시다. hover에 좀 더 정보를 추가하고 싶다면 hover_data에 설정하면 됩니다. 이번엔 `sepal length`, `sepal width` 정보도 추가해보겠습니다. 그리고 회색 기본 배경을 바꾸기 위해 template를 `plotly_white`로 변경했습니다. 이 외에도 더 많은 종류를 [plotly-template]에서 확인할 수 있습니다. 또한, `target`별 색을 직접 color_discrete_map으로 지정하고, `title`을 설정했습니다. 이때 확인할 수 있듯이 dictionary 형태로 옵션을 설정하게 됩니다.
<br>

{% highlight Python %}
import plotly.express as px

color_discrete_map = {'CAT1': 'rgb(85,107,47)', 'CAT2': 'rgb(169,169,169)', 'CAT3': 'rgb(250,128,114)'}
fig = px.scatter(iris, x='petal length (cm)', y='petal width (cm)', hover_data=['target', 'sepal length (cm)', 'sepal width (cm)'], color='target', symbol='target', template= 'plotly_white', color_discrete_map=color_discrete_map)
fig.update_layout(
    width=500,
    height=400,
    title=dict(text="Iris Scatter Plot with hover", font=dict(size=13)))
fig.show()
{% endhighlight %}

<div class="plotly-div"><iframe src="/!contents_plot/2020-11-09-2.html"
    scrolling="no"
    width="500"
    height="400"
    frameborder="0" allowfullscreen>
</iframe></div>
<br><br><br><br>

마지막으로 ~~지금도 잘 보이긴 하지만~~ 붓꽃 군집을 배경색을 함께 활용해서 시각화해 봅시다. 배경을 지정하는 방법은 매우 직관적입니다. x0, y0는 각각 x축, y축의 시작점이고 x1, y1은 각각 x축, y축의 끝점입니다. `shapes` 옵션을 통해 배경으로 도형을 그린다고 생각하면 되는데, `type=rect`는 직사각형으로 좌측 아래 꼭잣점이 (x0, y0), 우측 위 꼭짓점이 (x1, y1) 입니다. `type='circle`는 원형으로 장축, 단축의 양 끝점이 각각 (x0, x1), (y0, y1) 입니다. 그리고 ~~사실 필요 없지만~~ setosa와 versicolor 사이의 경계에 `horizontal, vertical line`를 각각 하나씩 추가해보았습니다. 이는 `add_shape()` 함수로 이용하여 같은 방식으로 선의 양 끝점의 좌표를 지정하면 됩니다.
<br>

{% highlight Python %}
import plotly.express as px

color_discrete_map = {'CAT1': 'rgb(85,107,47)', 'CAT2': 'rgb(169,169,169)', 'CAT3': 'rgb(250,128,114)'}
fig = px.scatter(iris, x='petal length (cm)', y='petal width (cm)', hover_data=['target', 'sepal length (cm)', 'sepal width (cm)'], color='target', symbol='target', template= 'plotly_white', color_discrete_map=color_discrete_map)
fig.update_layout(
    width=500,
    height=400,
    title=dict(text="Iris Scatter Plot with hover and background", font=dict(size=13)),
    shapes=[dict(type="rect",xref="x",yref="y",
                 x0=np.min(iris[iris.target == "setosa"]['petal length (cm)'])-.3, y0=np.min(iris[iris.target == "setosa"]['petal width (cm)'])-.07,
                 x1=np.max(iris[iris.target == "setosa"]['petal length (cm)'])+.3, y1=np.max(iris[iris.target == "setosa"]['petal width (cm)'])+.07,
                 opacity=0.2,fillcolor="purple",line_color="purple"),
            dict(type="rect",xref="x",yref="y",
                 x0=np.min(iris[iris.target == "versicolor"]['petal length (cm)'])-.3, y0=np.min(iris[iris.target == "versicolor"]['petal width (cm)'])-.07,
                 x1=np.max(iris[iris.target == "versicolor"]['petal length (cm)'])+.1, y1=np.max(iris[iris.target == "versicolor"]['petal width (cm)']),
                 opacity=0.2,fillcolor="orange",line_color="orange"),
            dict(type="rect",xref="x",yref="y",
                 x0=np.min(iris[iris.target == "virginica"]['petal length (cm)'])-.1, y0=np.min(iris[iris.target == "virginica"]['petal width (cm)'])-.07,
                 x1=np.max(iris[iris.target == "virginica"]['petal length (cm)'])+.3, y1=np.max(iris[iris.target == "virginica"]['petal width (cm)'])+.05,
                 opacity=0.2,fillcolor="skyblue",line_color="skyblue")])

fig.add_shape(dict(type="line", 
                   x0=0,y0=0.8, x1=7,y1=0.8,
                   line=dict(color="red",width=1,dash="dash")))

fig.add_shape(dict(type="line", 
                   x0=2.45,y0=0, x1=2.45,y1=2.5,
                   line=dict(color="red",width=1,dash="dash")))


fig.show()
{% endhighlight %}

<div class="plotly-div"><iframe src="/!contents_plot/2020-11-09-3.html"
    scrolling="no"
    width="500"
    height="400"
    frameborder="0" allowfullscreen>
</iframe></div>
<br><br><br><br>

여기서 `shapes` 세 `rect`의 좌푯값을 바꾸면 가장 처음 봤던 그림도 쉽게 그려볼 수 있습니다. plotly를 활용하면 `zoom in`과 `zoom out` 또한 자유롭게 할 수 있습니다. 특히 3차원 그림을 자유롭게 돌려서 확인할 수도 있는데, 이는 다음 포스팅에서 확인해봅시다.
<br><br><br>

[plt-site]: https://plotly.com/python/
[plt-site-template]: https://plotly.com/python/templates/