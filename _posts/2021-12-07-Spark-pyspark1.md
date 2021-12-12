---
layout: post
title:  "[Spark/pyspark] pyspark dataframe 읽기 및 저장 / spark.read.option.csv, wrtie.option.csv"
date:   2021-12-07 12:36:08
categories: [Engineering]
comments: true
---

빅데이터 처리를 위해 하둡과 함께 많이 사용하는 Spark를 정리해보았다. Spark는 주로 Scala 또는 Python 으로 사용하는데, Python에서 사용할 때 pyspark을 이용하게 된다. 이때 이름은 마찬가지로 dataframe이나 pandas dataframe과는 조금 상이하고 사용하는 함수도 다르다. 이번 포스팅에서는 pyspark dataframe을 읽고 저장하는 코드를 정리해보았다.

<br>

## 데이터 로드

pyspark를 사용하기 전에 먼저 pyspark에서 session을 연결할 수 있도록 설정해야 한다.

```pyspark
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession

sc = SparkContext.getOrCreate()
spark = SparkSession(sc)
```
<br>

이렇게 세션을 하나 생성한 후 하둡에 저장된 csv 파일에 대해서 다음과 같이 spark.read.csv으로 데이터를 로드한다. 

```python
df = spark.read.csv("hdfs://user/test/content.csv")
```
<br>

데이터를 로드할 때 추가적으로 옵션들을 지정할 수 있다. 예를 들어 csv가 아니라 tsv 형태이거나 parquet 형태라면 다음과 같이 읽을 수 있다.

```python
# To read TSV
df = spark.read.option("delimiter", "\t").csv("hdfs://user/test/content.tsv")

# To read parquet
df = spark.read.parquet("hdfs://user/test/content.parquet")
```
<br>

또한 header가 있는 데이터를 읽을 때는 다음과 같이 설정할 수 있다.

```python
df = spark.read.option("header", "true").csv("hdfs://user/test/content.csv")
```
<br>

기본적으로 데이터를 로드할 때 모두 컬럼을 string type으로 읽는다. 정수형이나 실수형 컬럼이 포함되어 있고 이를 자동으로 설정하고 싶을 때는 다음과 같이 inferSchema 옵션을 설정할 수 있다.

```python
df = spark.read.option("inferSchema", "true").csv("hdfs://user/test/content.csv")
```
<br>

추가적으로 만약 폴더 내의 모든 파일을 읽어야할 때는 다음과 같이 읽을 수 있다. 예를 들어 배치 파일을 읽거나 분산되어 저장된 파일을 읽을 때 자주 사용하게 된다.

```python
df = spark.read.option("inferSchema", "true").csv("hdfs://user/test/*")
```

<br>
<br>

## 데이터 저장

데이터를 저장할 때는 write.csv를 이용한다. 저장은 result 폴더 내에 여러 압축된 형태의 분산된 파일들이 저장된다. 이는 앞에서 /* 형식으로 다시 읽을 수 있다.

```python
df.write.csv("hdfs://user/test/result")
```
<br>

데이터 로드와 마찬가지로 여러가지 옵션을 줄 수 있다. tsv를 저장할 때는 다음과 같이 설정하면 된다.

```python
df.write.option("delimiter", "\t").csv("hdfs://user/test/result")
```
<br>

또한 header를 함께 저장할 때는 다음과 같이 설정할 수 있다.

```python
df.write.option("header", "true").csv("hdfs://user/test/result")
```
<br>

마지막으로 이미 데이터가 있더라도 덮어씌우고 싶을 때, 다음과 같이 mode를 설정하면 된다.

```python
df.write.mode("overwrite").csv("hdfs://user/test/result")
```

<br>