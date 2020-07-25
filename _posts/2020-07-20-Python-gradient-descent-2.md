---
layout: post
title:  "[Python/Jupyter] 확률적 경사 하강법 Stochastic Gradient Descent / 미니 배치 경사 하강법 Minibatch Gradient Descent 정리"
date: 2020-07-20 18:56:37
categories: [Machine Learning]
comments: true
---
<br>
이번 블로그 내용은 머신러닝 도서 `『Hands-On Machine Learning with Scikit-Learn and TensorFlow』` 4장에서 다루는 경사 하강법(Gradient Descent) 내용을 정리하고자 하는 목적을 가지고 있습니다. 
<br><br><br><br>

경사 하강법, 배치 경사 하강법 내용을 보려면 다음 포스팅을 클릭하세요! <br><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [>> 경사 하강법 GD 이해, 배치 경사 하강법 BGD][next-1]
<br><br><br><br>

### 확률적 경사 하강법

저번 포스팅에 이어서 확률적 경사 하강법에 대해 알아봅시다. 앞에서 배치 경사 하강법의 경우 다음 스텝으로 가기 위해 매번 모든 데이터를 이용하여 그래디언트를 계산했습니다. 상대적으로 수렴 속도를 더 빠르게 하기 위해 매 스탭에서 한 개의 샘플을 무작위로 선택하고 해당 샘플에 대한 그래디언트만 계산하는 것이 바로 확률적 경사 하강법입니다. 그런데 문제는 계산 속도는 빨라지나, 전역 최솟값을 찾는데 문제가 발생할 수 있습니다. 
<br><br><br><br><br><br>

따라서 확률적 경사 하강법의 장단점을 나열하면 다음과 같습니다.
<br><br>

> * 장점: 배치 경사 하강법에 비해 **빠름**, 하나의 샘플에 대한 그래디언트만 계산, 즉 데이터가 매우 클 때 유용함, 배치 경사 하강법에 비해 **비용함수가 불규칙할 경우 알고리즘이 지역 최솟값(local minimum)을 건너뛸 가능성**이 높음  

> * 단점: 배치 경사 하강법에 비해 **불안정**, 일정하게 비용함수가 감소하는 것이 아니라 요동치기 때문에 평균적으로 감소 배치 경사 하강법에 비해 **전역 최솟값(global minimum)에 다다르지 못함, 하이퍼 파라미터 증가**(ex stochastic index, ~~learning schedule~~)    

<br><br><br><br>

이러한 단점을 극복하기 위해 전역 최솟값을 잘 찾을 수 있는 방법이 있는데, 이는 바로 학습률을 조정하는 것입니다. 즉 처음에는 학습률을 크게 하여 지역 최솟값에 빠지지 않도록 한 후 점차 작게 줄여서 전역 최솟값에 도달하도록 한다면 어느 정도 해결이 가능합니다. 즉, 학습률을 점진적으로 줄이는 **학습 스케줄**`(learning schedule)`을 설정합니다.
<br><br><br><br><br><br>

그렇다면 learning schedule을 적용하는 경우와 그렇지 않은 경우로 나눠 확률적 경사 하강법을 linear regression에 사용해봅시다. ~~앞의 포스팅에서 y=6x-2 모델을 계속 활용합니다.~~ 먼저, learning schedule을 적용하지 않는 경우부터 살펴봅시다.
<br><br><br><br>

{% highlight Python %}
# [ learning schedule을 하지 않는 경우 ]

n_epochs = 50
t0, t1 = 5, 50  # 학습 스케줄 하이퍼 파라미터 learning schedule hyperparameters
eta = 0.01

B_hat_SGD = np.random.randn(2,1)  # 초기화

for epoch in range(n_epochs):
    for i in range(m):
        stoch_index = np.random.randint(m)
        xi = X_b[stoch_index:stoch_index+1]
        yi = y[stoch_index:stoch_index+1]
        
        gradients = 2 * xi.T.dot(xi.dot(B_hat_SGD) - yi)
        B_hat_SGD = B_hat_SGD - eta * gradients   
{% endhighlight %}
<br><br><br><br>

`random index`를 받아 이를 이용하여 그래디언트를 계산하는 것 외에는 크게 다르지 않습니다. 그리고 샘플은 말그대로 하나의 데이터, 레코드를 의미합니다. B_hat_SGD 결과로는 [6.1265, -1.7448]이 출력되었습니다.
<br><br><br><br><br><br>

이번에는 learning schedule을 적용한 경우를 살펴봅시다.
<br><br><br><br>

{% highlight Python %}
# [ learning schedule을 하는 경우 ]

n_epochs = 50
t0, t1 = 5, 50  # 학습 스케줄 하이퍼 파라미터 learning schedule hyperparameters
eta = 0.01

B_hat_SGD = np.random.randn(2,1)  # 초기화

for epoch in range(n_epochs):
    for i in range(m):
        stoch_index = np.random.randint(m)
        xi = X_b[stoch_index:stoch_index+1]
        yi = y[stoch_index:stoch_index+1]
        
        gradients = 2 * xi.T.dot(xi.dot(B_hat_SGD) - yi)
        B_hat_SGD = B_hat_SGD - eta * gradients   
{% endhighlight %}
<br><br><br><br>

learning schedule을 위한 하이퍼 파라미터가 추가되어 점진적으로 학습률이 줄어드는 것을 알 수 있습니다. 이를 조금씩 조절하면 B_hat_SGD 결과 또한 달라지게 됩니다. B_hat_SGD 결과는 [5.9443, -1.8941]로 이 예제에서는 사실상 learning schedule을 적용하지 않은 경우와 그렇게 큰 차이는 나지 않았습니다.
<br><br><br><br><br><br>

위 두 예제를 통해, 확률적 경사 하강법의 배치 크기는 1이며 100번 뽑고 이를 총 50번 반복하는 것을 확인할 수 있습니다. 전체 훈련 세트에 대해 배치 경사 하강법에서는 10000번을 반복하는 동안 확률적 경사 하강법에서는 50번만 반복하고도 LSE(최적 추정치)와 비슷한 값에 도달한 것을 확인할 수 있습니다.
<br><br><br><br><br><br>


### 미니 배치 경사 하강법

마지막으로 미치 배치 경사 하강법에 대해 알아봅시다. 미니 배치 경사 하강법은 확률적 경사 하강법과 달리 랜덤한 파트가 없으나 이와 비슷하게 모든 데이터를 매번 사용하지 않습니다. 대신 매 스탭에서 **미니 배치**라는 임의의 작은 샘플 세트에 대해 그래디언트를 계산하여 감소하는 방향으로 파라미터를 업데이트합니다.
<br><br><br><br><br><br>

미니 배치 경사 하강법의 장단점을 비교하여 나열해보면 다음과 같습니다.
<br><br>

> * 장점: 배치 경사 하강법에 비해 **빠름**, 확률적 경사 하강법과 달리 랜덤 파트도 없어서 더욱 빠름, 미니배치가 어느정도 크면 **확률적 경사 하강법에 비해 전역 최솟값에 더 가까이 도달**

> * 단점: 확률적 경사 하강법에 비해 **지역 최솟값에서 빠져나오기는 더 힘들 수 있음**, 마찬가지로 **하이퍼 파라미터 증가**(ex minibatch size)    

<br><br><br><br>

그렇다면 이어서 같은 예제로 미니 배치 경사 하강법 추정치를 구해봅시다. 위에서 배운 learning schedule을 추가한 코드입니다.
<br><br><br><br>

{% highlight Python %}
n_epoch = 50
minibatch_size = 20

np.random.seed(42)
B_hat_MGD = np.random.randn(2,1)  # 무작위 초기화

# learning schedule을 추가
t0, t1 = 200, 1000
def learning_schedule(t):
    return t0 / (t + t1)

t = 0
for epoch in range(n_epoch):
    shuffled_indices = np.random.permutation(m)
    X_b_shuffled = X_b[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    
    for i in range(0, m, minibatch_size):
        t += 1
        xi = X_b_shuffled[i:i+minibatch_size]
        yi = y_shuffled[i:i+minibatch_size]
        gradients = 2/minibatch_size * xi.T.dot(xi.dot(B_hat_MGD) - yi)
        eta = learning_schedule(t)
        B_hat_MGD = B_hat_MGD - eta * gradients
{% endhighlight %}
<br><br><br><br> 

B_hat_MGD 결과는 [5.9447, -1.8872]로 LSE와 비슷하며, 배치 크기는 20으로 전체 데이터를 1/5씩 사용하고 전체 데이터에 대해 총 50번 반복합니다.
<br><br><br><br><br><br>

확률적 경사 하강법과 미니 배치 경사 하강법을 비교해보면, 확률적 경사 하강법은 **배치 크기가 1**이나 미니 배치 경사 하강법에서는 **배치 크기가 >1**로 조금 더 크게 설정하는 것이 일반적입니다. 또한, 미니 배치에서 한 반복(epoch)에 있어서 **한번도 사용하지 않은 데이터는 없으나**, 확률 경사법에서는 **한번도 사용하지 않은 데이터가 있을 수도 있다는 것**이 큰 차이점입니다.
<br><br><br><br><br><br>

그럼 다음 포스팅에서 만나요 X)

[next-1]: https://givitallugot.github.io/articles/2020-07/Python-gradient-descent-1