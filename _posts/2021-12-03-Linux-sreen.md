---
layout: post
title:  "[Linux] 스크린(Screen) 명령어 익히기 / 서버에 jupyter notebook 계속 띄워두기"
date:   2021-12-01 10:52:10
categories: [Engineering]
comments: true
---
<br>

스크린은 서버 안에서 터미널을 띄워둘 때 사용한다. 예를 들어 매번 jupyter notebook을 터미널에 입력해서 여는 것이 귀찮고 불편하다면, 스크린으로 임의의 터미널을 만들어두고 거기서 jupyter notebook을 영원히(?) 실행하면 되는 것이다.

<br><br>

## screen 생성
터미널에서 대문자 S 옵션으로 이름을 설정하여 입력하면 스크린이 생성된다. 즉, 새로운 터미널 창이 나타난다.

```bash
screen -S {이름}
# screen -S anaconda
``` 

<br><br>

## screen에서 나가기
생성된 screen은 종료하지 않고 다시 터미널로 돌아갈 때는 단축키를 이용하면 편리하다. 단축키는 ctrl + a + d 이다.

<br><br>

## screen으로 다시 들어가기
이미 생성한 screen으로 다시 돌아가기 위해서는 다음과 같이 r 옵션을 주고 실행하면 된다.

```bash
screen -r {이름}
```

<br><br>

## screen 목록 보기
현재 시스템에 띄워진 screen 목록을 볼 수 있다.

```bash
screen -list

# screen -S anaconda 를 수행했을 때,

# 47234.anaconda (Detached)
# 1 Socket in /location/

# 앞의 숫자는 screen number이며 고유한 번호로 생각하면 됨
```  

<br><br>

## screen 삭제
이제 더 이상 screen이 필요하지 않을 때 생성된 screen을 삭제하는 방법이다. screen 목록 보기를 통해 screen number를 확인하고, 이를 입력하여 삭제하면 된다.

```bash
screen -X -S {스크린넘버} kill
```  

<br><br>

처음 리눅스를 접할 때는 어렵게 느껴지지만 알면 매우 간단하고 편리한 기능인 것 같다.