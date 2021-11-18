---
layout: post
title:  "[Linux] 쉘 스크립트(Shell Script) 작성을 위한 기본 명령어 / ls, chmod, vi, cat, df du, env shebang"
date:   2021-08-28 16:29:26
categories: [Engineering]
comments: true
---
<br>

하둡을 공부하며 알게된 파일 시스템 (HDFS) 쉘 명령어를 정리해보았다. 두서 없지만 그래도 정리하는 데에 의의를!


## ls
디렉토리 확인을 위한 명령어로 -al 옵션을 추가하면 권한을 비롯한 자세한 정보를 볼 수 있다. 비슷한 명령어로 ll도 있다.
```bash
ls -al
``` 

<br><br>

## chmod
권한을 바꾸는 명령어다. 스크립트 등의 파일을 터미널에서 실행할 때 권한이 있어야 실행 가능하다. 권한은 크게 세 개로 나뉘고 이를 각각 설정할 수 있는데, 모든 유저에게 실행 권한을 주는 777 명령어를 실행한다.

```bash
chmod 777 test.py # 777: 소유자 권한 / 그룹 사용자 권한 / 기타 사용자 권한
chmod +x test.py # 소유자만 모든 것(쓰기, 읽기, 실행)이 가능하고 그 외 사용자의 경우는 읽기, 실행은 가능하나 쓰기는 불가능
``` 

<br><br>

## vi
ls, pwd 다음으로 가장 많이 접해봤을 명령어다. 파일을 확인하고 수정할 수 있도록 열 때 사용하고, 주로 소프트웨어 설치 후 bash 파일 바꾸는 용도로 많이 열어봤을 것이다.

```bash
vi test.sh
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

