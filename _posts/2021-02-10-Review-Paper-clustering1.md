---
layout: post
title:  "[Review/Paper] Clustering 논문 리뷰 with 다양한 클러스터링 방법, 비정형 데이터 클러스터링 관련"
date:   2021-02-08 02:29:40
categories: [Machine Learning]
comments: true
---
<br>
(Clustering 관련 논문에 대한 짧은 리뷰입니다.)

#### Data Clustering: 50 years beyond K-means, Anil K. Jain 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>[click to browse the paper][paper-1] 
<br><br>

2009년 논문으로 Clustering의 간략한 설명과 클러스터링 알고리즘 설계 시 고려사항, 여러가지 Clustering 방법론 및 최신 동향이 매우 잘 정리된 논문이다. 내용 중 일부를 간략하게 정리했다.

<br>
> ① 서로 다른 클러스터를 구별짓는 요소에는 `shape`, `size`, `density`가 있다. <br>
② Clustering은 크게 `Hierarchical` 또는 `Partitional`로 나뉘며, `Hierarchical Clustering`을 제외하고 우리가 아는 대부분의 Clustering은 `Partitional Clustering` 이다. <br>

> ③ `Fuzzy c-means`(1965, Dunn)는 한 점이 여러 클러스터에 들어갈 수 있으며, 각 클러스터에 들어갈 확률을 계산하는 soft한 방법이다. <br>
④ `K-medoid`는 `K-means`와 비슷하나 centroid를 실제 데이터 포인트 중 하나로 잡는다. 이에 따라 시간복잡도가 상대적으로 크지만 `K-means`에 비해 outlier의 영향을 덜 받는 편이다. <br>
⑤ `DBSCAN`(1996, Ester et al.)은 density 기반의 알고리즘이며 데이터의 밀도가 임계값보다 높은 부분을 클러스터로, 그 외는 노이즈로 두는 알고리즘으로 클러스터의 개수를 파라미터로 지정하지 않는다. <br>
⑥ `Probabilistic Model`은 `Mixture Model` 중심으로 발전되었으며, 대부분의 경우 EM 알고리즘을 사용한다. <br>
⑦ `Bayesian`을 적용한 모델로 `Latent Dirichlet Allocation`(Blei et al., 2003), `Pachinko Allocation model`(Li and McCallum, 2006), `undirected graphical model`(Welling et al., 2005)가 있다. <br>

> ⑧ 클러스터의 개수를 정하는 문제는 모델 선택 문제와 같은 맥락이며, `MDL`(Hansen and Yu, 2001), `BIC`, `AIC`, `Gap statistics`(Tibshirani et al., 2001), `DP`(Ferguson, 1973; Rasmussen, 2000) 등의 기준을 활용할 수 있다. <br>
⑨ 클러스터에 대한 타당성은 크게 `internal` (한 알고리즘에서 구조의 변화에 따른 비교), `relative` (여러 알고리즘의 다양한 구조에 따른 비교), `external` (실제 y와 비교) <br>
⑩ `Supervised Learning`의 `Cross Validation` 평가와 같은 역할을 `Unsupervised Learning (Clustering)`에서는 `Prediction Accuracy`로 수행한다. <br>
⑪ 이미지, 텍스트 등 비정형 데이터, 대용량 데이터, `Rank data`, `Dynamic data`, `Graph data` 등과 같은 일반적이지 않은 데이터에 적용할 수 있는 Clustering 방법론들이 다양하게 제시된다.

<br><br>
개인적으로 깨닫게된 내용을 추가1, 2에 더 적어보았다. ⑪번에 정형 데이터가 아닌 데이터 클러스터링 방법론에 대한 아이디어를 더 얻고자 한다면 해당 논문의 References를 찾아보는 것이 좋을 것 같다.

<br><br>
<b> 추가1. 대부분의 Clustering 방법론은 군집 내 거리를 최소화하고 군집 간 거리를 최대화하는 것이 목적이다. 이때 유클리디안 거리를 사용하는 K-means에서 이 둘이 같기 때문에 군집 내 거리를 최소화하는 식을 이용하여 주로 정의한다. <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[click to browse relevent question and answer][stackoverflow-1] 
<br><br>

<b> 추가2. Semi-Supervised Clustering에서 `Supervised`는 주어지는 정보는 어떤 점이 특정 클러스터에 속한다는 정보가 아니라, 같은 클러스터에 속하는 점과 서로 다른 클러스터에 속하는 점에 대한 정보이다. ~~생각해보면 클러스터에서는 1번, 2번 이런 식의 명명이 불가능하니 이런 방식으로 `Supervised`되는 것이 당연한데..~~
<br><br><br><br>


[출처] Jain, Anil K. "Data clustering: 50 years beyond K-means." Pattern recognition letters 31.8 (2010): 651-666.

[paper-1]: https://www.sciencedirect.com/science/article/pii/S0167865509002323?via%3Dihub

[stackoverflow-1]: https://www.notion.so/K-means-clustering-64b4c9877c6140578c0cc70f3482e05c