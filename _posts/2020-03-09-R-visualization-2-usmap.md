---
layout: post
title:  "[R] R을 이용한 미국 지도 시각화 with ggiraphExtra ggplot2 ggmaps ggthemes albersusa viridis packages"
date:   2020-03-09 17:35:12
categories: [Visualization]
comments: true
---
<br>
미국 관련 데이터셋을 분석할 때 해석을 돕기 위해 시각화를 지도와 함께 나타내고 싶은 경우가 있습니다. 이번 내용을 통해 Alaska와 Hawaii 주를 포함하는 미국 지도의 주별 시각화에 도움이 되셨으면 좋겠습니다.

* 주의: 내용을 따라하기 위해서는 R과 R Studio가 필요합니다.  
<br><br><br><br>

### 미국 주별 지도 시각화 (V 1.0)

V 1.0에서는 미국의 주 51개 중 **Alaska, Hawaii** 주를 포함하지 않는 미국 지도를 그려보겠습니다. (이 두 주를 포함하는 시각화는 밑의 V 2.0에서 다룹니다.)

R에 내장된 `USArrests` 데이터를 사용하며 이는 1973년 미국 주별 강력 범죄율 데이터입니다. 시각화하고자 하는 데이터셋을 사용하셔도 문제없습니다.

{% highlight R %}
str(USArrests)
head(USArrests)
{% endhighlight %}  
<br><br><br><br>

`str()`와 `head()`로 `USArrests`를 확인해본 결과, 행 이름이 미국 주들로 이를 'state' 변수로 변환해줄 필요가 있습니다. 또한 (곧 나올) 이번에 이용할 지리 정보의 경우 'state'가 모두 소문자로 들어가 있기 때문에 동일하게 'state'를 모두 소문자로 변환합니다. 그리고 `crime` 데이터셋으로 저장합니다.

{% highlight R %}
library(tibble)

crime <- rownames_to_column(USArrests, var="state") #행 이름을 'state' 변수로
crime$state <- tolower(crime$state) #state를 소문자로
{% endhighlight %}  
<br><br><br><br>

지리 정보는 `maps` 패키지의 `state`를 이용합니다. 이를 data frame 형태로 불러오기 위해 `ggplot2` 패키지의 `map_data()` 함수를 이용합니다.

{% highlight R %}
library(ggplot2)

states_map <- map_data("state")
head(states_map)
{% endhighlight %}  
<br><br><br><br>

`states_map`의 region는 주를 나타내며 long, lat으로 주의 위도와 경도를 확인할 수 있습니다. 이제 시각화의 대상인 `crime`과 지리 정보인 `states_map`을 `ggiraphExtra` 패키지의 `ggChoropleth` 함수를 이용하여 시각화합니다.

{% highlight R %}
library(ggiraphExtra)
ggChoropleth(data=crime, aes(fill=Murder, map_id=state), map=states_map)
{% endhighlight %}  

![1-map1](/!contents_plot/2020-03-09-2.JPG){: width="60%"}  
<br><br><br>

`fill=Murder`를 통해 수치에 따라 색의 진하기가 다르게 나타납니다. 기본 그림도 나쁘지 않지만 위도와 경도축의 값들은 사실상 필요 없어서 `ggthemes` 패키지를 이용하여 배경을 깔끔히 변경해보았습니다.

{% highlight R %}
library(ggthemes)
ggChoropleth(data=crime, aes(fill=Murder, map_id=state), map=states_map) + 
  theme_map() + theme(legend.position="right")
{% endhighlight %}  

![1-map2](/!contents_plot/2020-03-09-3.JPG){: width="60%"}  
<br><br><br>

`interactive=T`를 추가해서 커서를 지도 위에 두면 해당 주에 대한 정보를 확인할 수 있습니다. 이는 시각화 해석 시 굉장히 유용합니다. 다만, theme이 있을 때는 적용되지 않아 기본 그림에 옵션을 추가해 보았습니다.

{% highlight R %}
ggChoropleth(data=crime, aes(fill=Murder, map_id=state), map=states_map, interactive=T)
{% endhighlight %}  

![1-map3](/!contents_plot/2020-03-09-4.jpg){: width="60%"}

* interactive는 rmarkdown을 html로 knit했을 때 특히 그 진가를 발휘합니다.

* interactive 캡처라서 시각화 화질이 별로임을 양해해주세요 :X  
<br><br><br><br>

### 미국 주별 지도 시각화 (V 2.0)

위의 미국 지도 시각화에는 **Alaska, Hawaii** 주가 나타나지 않습니다. `dplyr` 패키지로 `crime`과 `states_map`에 state를 확인해봅시다.

{% highlight R %}
crime %>% group_by(state) %>% summarise(n=n())
states_map %>% group_by(region) %>% summarise(n=n())
{% endhighlight %}  
<br><br><br><br>

시각화의 대상인 `crime`의 obs는 50으로 **Alaska, Hawaii**에 대한 정보도 존재합니다. 그러나 `states_map`의 obs는 49로 **Alaska, Hawaii** 지리 정보가 없으며 **district of columbia**가 존재합니다. 따라서 이번에는 다른 지리 정보를 이용해서 `ggplot2` 패키지를 이용하여 그림을 그려보도록 하겠습니다. 

먼저 `devtools` 패키지를 import 합니다. 그리고 `albersusa` 패키지를 다운받습니다.

{% highlight R %}
library(devtools)
devtools::install_github("hrbrmstr/albersusa")
{% endhighlight %}  
<br><br><br><br>

만약 다음과 같이 에러가 출력되는 경우 [rtools link][rtools-site]에서 recommended 버전의 `Rtools`를 설치해줍니다. 이는 컴파일 시 `Rtools`가 필요하기 때문입니다. 만약 `Rtools` 다운 후에도 폴더에 설치할 수 없다는 에러가 발생한다면 관리자 권한으로 Rstudio를 재실행한 후 시도하면 됩니다.

![error1](/!contents_plot/2020-03-09-1.JPG){: width="70%"}  
<br><br><br><br>

`alberusea` 패키지를 import하여 미국 지리 정보를 us 변수로 불러옵니다. 그리고 `ggmap` 패키지의 `fortify()` 함수를 이용하여 우리가 알 수 있는 좌표계로 변경하고 us_map 변수로 저장합니다.

{% highlight R %}
library(albersusa) #to use usa_composite()
us <- usa_composite()

library(ggmap) #to use fortify()
us_map <- fortify(us, region="name")
{% endhighlight %}  
<br><br><br><br>

us_map을 확인해봅시다. 미국 51개의 state에 대한 지리 정보가 모두 담겨있는 것을 확인할 수 있습니다.

{% highlight R %}
head(us_map)
us_map %>% group_by(id) %>% summarise(n=n())
{% endhighlight %}  
<br><br><br><br>

참고로 앞에서 이용했던 `ggChoropleth()` 함수로는 시각화 할 수 없습니다. 그 대신 훨씬 익숙하고 쉬운 `ggplot2` 패키지의 함수들을 이용해서 시각화를 진행해봅시다. 이를 위해 시각화 대상인 `crime`과 지리 정보인 `us_map`을 join 해야 합니다.

앞에서 `crime`의 `state`를 모두 소문자로 바꿔주었는데, `us_map`의 state 변수인 `id`는 맨 첫 번째 글자가 대문자입니다. join을 하기 위해서는 Key 값이 될 state 정보가 동일해야 하므로, `crime`을 다시 불러옵시다.

{% highlight R %}
crime <- rownames_to_column(USArrests, var="state") #crime 다시 불러오기
us_map_crime <- left_join(us_map, crime, by=c("id"="state")) #us_map에 시각화할 crime 내용 추가
head(us_map_crime)
{% endhighlight %}  
<br><br><br><br>

이제 시각화할 데이터셋의 준비가 완료되었습니다. 기본 틀을 그려봅시다.

{% highlight R %}
ggplot() + geom_map(data=us_map_crime, map=us_map, aes(x=long, y=lat, map_id=id), color="#2b2b2b", size=0.1, fill=NA)
{% endhighlight %}

![2-map1](/!contents_plot/2020-03-09-5.JPG){: width="60%"}  
<br><br><br><br>

**Alaska, Hawaii**가 미국 대륙 왼쪽 아래에 나타납니다. `fill=Murder`로 채색합니다.

{% highlight R %}
ggplot() + geom_map(data=us_map_crime, map=us_map, aes(x=long, y=lat, map_id=id, fill=Murder), color="grey", size=0.1) + 
  scale_fill_gradient(low = "#FFFFFF", high = "#f094b0", space = "Lab", guide = "colourbar") + 
  theme_bw() + labs(title = "Crime Rate by state in the US") + 
  theme(panel.grid.major.x = element_blank(), panel.grid.minor.x = element_blank(), panel.grid.major.y = element_blank(), panel.grid.minor.y = element_blank(), plot.title = element_text(face = "bold", size = 14, hjust = 0.5))
{% endhighlight %}

![2-map2](/!contents_plot/2020-03-09-6.JPG){: width="60%"}

* ggplot 코드: plot에 기본 틀을 넣고 ggplot의 여러 함수로 세부 옵션 조정. scale_fill_gradient()를 통해 숫자가 높으면 진한 색, 숫자가 낮으면 연한 색으로 구분할 수 있도록 변경, theme_bw()로 뒷배경 제거, labs()로 제목 추가, theme()으로 plot의 기본 격자 제거  
<br><br><br><br>

옵션 조절을 통해 시각화를 더욱 발전시킬 수 있습니다. 다만, 이 지도는 평면적으로 실제 둥근 지구의 모습과는 괴리감이 있습니다. 이 또한 다음과 같이 보완할 수 있습니다.

{% highlight R %}
ggplot() + geom_map(data=us_map_crime, map=us_map, aes(x=long, y=lat, map_id=id, fill=Murder), color="black", size=0.1) + 
  coord_map("albers", lat0=30, lat1=40)
{% endhighlight %}

![2-map3](/!contents_plot/2020-03-09-7.JPG){: width="60%"}  
<br><br><br><br>

이제 `Conic projections` 좌표계에 지도를 사영한 모습을 볼 수 있습니다. 지금부터 내용은 시각화를 다듬는 방법이니 참고하시고 다양한 활용을 해보시면 좋겠습니다.

{% highlight R %}
ggplot() + geom_map(data=us_map_crime, map=us_map, aes(x=long, y=lat, map_id=id, fill=Murder), color="black", size=0.1) + 
  coord_map("albers", lat0=30, lat1=40) + 
  scale_fill_gradient(low = "#E6E7EA", high = "#6A5ACD", space = "Lab", guide = "colourbar") + 
  theme_bw() + labs(title = "Crime Rate by state in the US") + 
  theme(panel.grid.major.x = element_blank(), panel.grid.minor.x = element_blank(), panel.grid.major.y = element_blank(), panel.grid.minor.y = element_blank(), plot.title = element_text(face = "bold", size = 14, hjust = 0.5))
{% endhighlight %}

![2-map4](/!contents_plot/2020-03-09-8.JPG){: width="60%"}  
<br><br><br><br>

{% highlight R %}
library(ggthemes)
library(viridis)

ggplot() + geom_map(data=us_map_crime, map=us_map, aes(x=long, y=lat, map_id=id, fill=Murder), color="black", size=0.1) + 
  coord_map("albers", lat0=30, lat1=40) + 
  theme_map() + scale_fill_viridis(name="murder (1973)") + theme(legend.position="right")
{% endhighlight %}

![2-map5](/!contents_plot/2020-03-09-9.JPG){: width="60%"}  

* `viridis` 패키지와 `theme_map()` 함수로 매우 깔끔한 그림이나 오히려 범죄율이 심각할수록 색이 옅어져 오히려 해석이나 발표 자료로는 적합하지 않을 수도 있겠네요.  
<br><br><br><br>



[rtools-site]: https://cran.r-project.org/bin/windows/Rtools