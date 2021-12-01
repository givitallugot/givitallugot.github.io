---
layout: post
title:  "[Linux] 쉘 스크립트(Shell Script) 3 / 인자(args), 반복문(for), 조건문(if), 함수(function)"
date:   2021-11-29 11:46:27
categories: [Engineering]
comments: true
---
<br>
리눅스 쉘 스크립팅 시 필요할 수 있는 반복문, 조건문 등을 추가로 정리해보았다.


## 인자(args)
다음과 깉이 run.sh 스크립크를 실행할 때, 뒤에는 순서대로 인자를 받을 수 있다.

```bash
sh run.sh '2021-10-30' '2021-11-30'
``` 

run.sh 내부에서는 두 인자를 순서대로 $1, $2로 표현한다. 즉, 읹의 순서대로 $# 형태로 사용하면 된다.
```sh
DATE = $1
``` 

## 변수(variables)
스크립트 내에서 여러번 사용할 변수를 지정하거나, path를 매번 사용하지 않고 위치를 저장하여 사용할 수 있다. 특정 이름으로 path를 지정할 때 export 명령어를 사용하면 된다. 이렇게 사용하면 재사용이 간편해진다.

```sh
export HOME=/user/desktop/model

export DATA_DIR=$HOME/data
export CODE_DIR=$HOME/code
``` 

위와 같이 model 폴더에서 작업할 때 model/run.sh에서 code 폴더와 data 폴더에 접근할 필요가 있을 때, 다음과 같이 path를 설정하면 훨씬 가독성이 좋다.


## 반복문(for)
반복문의 기본 형식은 다음과 같다.

```sh
for i in '1' '2' '3' '4'; do
    # action

    # 하둡 상의 test-# tsv 파일을 로컬 /user/desktop/model/data 위치로 저장
    hdfs dfs -cat hdfs://user/test/test-$i.tsv > /user/desktop/model/data 
done
``` 

action의 위치에서 반복할 내용을 입력하면 된다. 


## 조건문(if)
조건문의 기본 형식은 다음과 같다.

```sh
if [ $1 = '2021-11-30']; then
    # action

    hdfs dfs -put -f /user/desktop/model/data/$1.tsv \
                    hdfs://user/test/$1.tsv
fi
``` 

만약 첫 번째 인자로 받은 값이 '2021-11-30' 이라면 로컬의 2021-11-30.tsv 파일을 하둡 상으로 저장하는 명령어이다. 참고로 \ 는 스크립트 파일에서 줄을 나눠서 사용할 때 추가해야 한 줄로 인식이 가능하다.

참고로 [ ] 안에는 여러 조건식을 사용할 수 있으며 다음을 참고하면 된다.


> [ -eq ]: 값이 같으면 참
> [ -ne ]: 값이 다르면 참

> [ -gt ]: 값1 > 값2
> [ -ge ]: 값1 >= 값2
> [ -lt ]: 값1 < 값2
> [ -le ]: 값1 <= 값2

> [ -a ]: &&, and 연산
> [ -o ]: ||, or 연산

> [ -z ]: 문자열의 길이가 0이면 참
> [ -n ]: 문자열의 길이가 0이 아니면 참

> [ -d ]: 파일이 디렉토리면 참
> [ -e ]: 파일이 있으면 참
> [ -w ]: 파일이 쓰기 가능하면 참
> [ -x ]: 파일이 실행 가능하면 참

> [ 파일1 -nt 파일2 ]: 파일1이 파일2보다 최신 파일이면 참
> [ 파일1 -ot 파일2 ]: 파일1이 파일2보다 과거 파일이면 참
> [ 파일1 -ef 파일2 ]: 파일1이 파일2랑 같은 파일이면 참


## 함수(function)
스크립트 내에서도 function 등을 사용하여 일종의 모듈화를 진행할 수 있다.

```sh
function make_dataset(){
    python3 $CODE_DIR/make_df.py $1
}

make_dataset '2021-11-30'
```

다음 코드는 make_dataset 함수를 생성하여 호출하고, make_df.py 파이썬 파일을 수행하도록 하는 스크립트이다. 이런 형식으로 수행할 작업을 나눌 수 있다는 점에서 더욱 가독성 높은 스크립트를 작성할 수 있다.