---
layout: post
title:  "[Hadoop] 하둡 파일 시스템 (HDFS) 쉘 명령어 정리 / ls, mkdir, put, get, cat, mv, cp, rm, chmod, chown"
date:   2021-08-28 16:29:26
categories: [Engineering]
comments: true
---
<br>

리눅스 쉘 스크립팅의 기본적인 내용들을 정리해보았다.

기본적인 HDFS의 파일을 조작하기 위해서 **hadoop fs** 또는 **hdfs dfs** 커맨드를 사용한다.
```bash
bin/hdfs dfs
hadoop fs
``` 

## ls
디렉토리 확인을 위한 명령어로 현재 디렉토리 내 모든 파일과 하위 디렉토리를 출력한다.

```bash
hadoop fs -ls /
``` 

* / : 하둡의 홈디렉토리 위치
* lsr은 ls -R과 같은 명령어

<br><br>

## mkdir
새로운 디렉토리를 생성하는 명령어. 당연하지만 이렇게 만든 폴더가 실제 로컬에 생성되는 것이 아니라 HDFS에 생성된다.

```bash
hadoop fs -mkdir /hdfs-user # hdfs-user 폴더 생성
hadoop fs -ls / # 로 확인
``` 

<br><br>

## touchz
빈 파일을 생성하는 명령어.

```bash
bin/hdfs dfs -touchz /hdfs-user/test.txt # test.txt 파일 생성
```

<br><br>

## put
로컬 시스템에 있는 특정 파일이나 디렉토리 내 모든 파일들을 하둡 파일 시스템으로 복사하는 명령어.

```bash
hadoop fs -put ./etc/xml /user/etc/ # 앞은 로컬 주소, 뒤는 HDFS 폴더 위치
```  
<br>

* **copyFromLocal** 명령어와 비슷하나 이 명령어가 조금 더 제한적.

```bash
bin/hdfs dfs -copyFromLocal
```

<br><br>

## get
put과 의미상 반대로 하둡 파일 시스템으로부터 로컬 시스템으로 파일을 복사하는 명령어.

```bash
hadoop fs -get /user/etc/apt/sources.list /tmp # 앞은 HDFS 폴더 위치, 뒤는 로컬 주소
```  
<br>

* **copyToLocal** 명령어와 비슷하나 이 명령어가 조금 더 제한적.

```bash
bin/hdfs dfs -copyToLocal /user/hadoop-test.txt /tmp
```

<br><br>

## cat
파일의 내용을 stdout(화면 출력)으로 보여주는 명령어.

```bash
bin/hdfs dfs -cat /user/hadoop-test.txt
```

<br><br>

## mv
HDFS 내에서 파일을 이동하는 명령어.

```bash
hadoop fs -mv /user/etc/apt /user/etc/xml # 앞은 이동 전 주소, 뒤에는 이동 후 주소
```

<br><br>

## cp
HDFS 내에서 파일을 복사하는 명령어.

```bash
bin/hdfs dfs -cp /user/etc/xml/apt /user/etc
```

<br><br>

## rm
파일 또는 디렉토리 삭제를 위한 명령어.

```bash
hadoop fs -rm -r /user/etc/xml/apt # -r 옵션 추가하면 디렉토리 삭제 가능
```

<br><br>

## chmod
읽기,삭제,수정에 대한 권한을 수정할 수 있는 명령어.

```bash
hadoop fs -chmod -r /user/etc/xml # read 접근 불가
hadoop fs -chmod -R 755 /user/etc/xml # read 접근 가능
```

<br><br>

## chown
owner를 변경할 때 사용하는 명령어.

```bash
hadoop fs -chown root /user/etc
```

<br><br>


