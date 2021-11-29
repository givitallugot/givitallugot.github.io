---
layout: post
title:  "[Linux] 쉘 스크립트(Shell Script) 작성을 위한 기본 명령어 2 / sh, cp, mv, rm, tar, echo"
date:   2021-11-25 13:24:10
categories: [Engineering]
comments: true
---
<br>

리눅스 쉘 스크립팅을 위한 기본 명령어를 이어서 정리해보았다. 두서 없지만 그래도 정리하는 데에 의의를!


## sh
sh 파일을 실행할 때 사용하는 명령어. 마치 python3 test.py 와 비슷한 형태로, 뒤에 인자들을 받아서 스크립트 내에서 이용할 수 있다. 그 때의 인자들은 $1 $2 이런 방식으로 받은 순서대로 나타낸다.

```bash
sh test.sh
``` 

<br><br>

## cp
파일을 복사하는 명령어. 앞에 명시된 파일을 뒤에 명시한 파일 이름으로 복사한다. 디렉토리도 함께 지정하면 된다. 현재 디렉토리의 test.py를 script 폴더 내의 test2.py로 저장하라는 의미이다.

```bash
cp test.py ./script/test2.py
```

<br><br>

## mv
파일을 이동하는 명령어로 마찬가지로 디렉토리를 함께 명시할 수 있다.

```bash

```

<br><br>

## df/du
시스템 상태 확인 시 사용하는 명령어다. 

```bash
df -h # 서버 메모리 확인
du -sh
du --max-depth=1 -h # 파일시스템 사용량 depth 1에 대해서 확인
```  

<br><br>

## cat
모든 파일 내용을 보여주는 명령어로 단독으로 쓸 때도 있지만 .sh 파일 내에서 > 와 함께 사용하여 현재 출력된 내용을 어떤 파일로 저장할 때도 많이 사용된다.

```bash
cat test.csv # test.csv 내용을 출력
cat test.csv > test.tsv # 출력 내용을 tsv 파일로 저장
```  

<br><br>

## 추가적인 기초 내용
알면 쉽지만 처음 접할 때는 생소한 것들이다.
1. **#!/bin/bash** 또는 **#!/usr/bin/python3** 이런 식으로 .sh 파일 맨 위에 쓰는 것은 `shebang`(셔뱅)으로 어떤 인터프리터에서 동작되는지 알려주는 것이다.

2. **#!/usr/bin/env bash #!/usr/bin/env python #!/usr/bin/env python3** 형태로 쓰기도 하는데, 이는 로그램의 경로가 시스템 환경에 따라 달라질 수 있어서 사용하는 것이다.

