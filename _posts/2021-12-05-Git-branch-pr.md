---
layout: post
title:  "[Linux] 협업을 위한 Git branch 및 Pull Request 생성 / Branch, PR"
date:   2021-12-05 10:26:02
categories: [Other topics]
comments: true
---

추가적인 Git 사용법을 정리했다.

<br><br>

## 기본적인 사용 방법

가장 처음 git을 접할 때, 먼저 원격 저장소를 clone한다. 기본적으로 master branch를 clone하게 된다.

```bash
git clone {repository 주소}
```

다음으로 코드가 수정되면, 이를 반영하기 위해서 다음과 같이 add, commit, push를 진행한다. 이때 push는 master에 하게 된다.

```bash
git add {. or 특정 파일}

git commit -m 'message'

git push origin master
```

<br><br>

## 협업을 위한 Git, Branch

협업용으로 git을 사용할 때는 주로 branch를 생성하게 된다. 먼저, git에서 branch를 만들고 clone한 후 다음과 같은 명령어로 branch를 만든다. -b 옵션은 branch를 만든 후 새로운 branch로 이동한다는 의미이다.

```bash
git checkout -b {branch_name}
```

<br><br>

## Push 및 PR

branch를 생성한 후에는 이제 branch에 push를 진행하면 된다. add, commit은 동일하나, push를 할 때 branch 명을 입력한다는 점이 다르다.

```bash
git add .
git commit -m 'message'
git push origin {branch_name}
```