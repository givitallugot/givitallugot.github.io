---
layout: post
title:  "[Hadoop] MAC OS M1(Big Sur)에서 하둡(Hadoop) 설치하기 / namenode 에러 java.net.ConnectException 및 .zshrc HADOOP PATH 설정"
date:   2021-08-16 14:15:00
categories: [Engineering]
comments: true
---
<br>

일단 사용한 버전은 다음과 같다.
- Mac OS Big Sur 11.2.2
- M1 칩, Mac Pro, 메모리 16GB

대용량 데이터 처리 공부를 하려고 하둡 (Hadoop) 설치하는데 설치에만 무려 4일이 걸렸다. 발생했던 오류들과 알게된 주의사항들을 정리해보았다.

**참고 링크**  
[https://techblost.com/how-to-install-hadoop-on-mac-with-homebrew/][link-1]  
[https://key4920.github.io/p/mac-os에-하둡hadoop-설치/][link-2]

일단 Homebrew를 다운받고 **brew install hadoop**으로 설치했으며 위와 같은 좋은 게시글을 참고했다. 다만 해당 글들을 제대로 따라가더라도 M1 이후에 라이브러리 위치 등 바뀐 점이 많아서 그런지 오류를 정말 샅샅히 찾아야했던.. Mac OS를 M1으로 처음 시작한 사람으로써 겪었던 고난들을 적어봤다. 혹시 맥잘알이 이 게시글을 보시고 설명 중에 틀린 부분이 있다면 댓글 주시면 좋을 것 같다.
<br><br><br><br>

## 1. VirtualBox를 사용
일단 결론부터 말하면 VirtualBox 위에 Ubuntu를 설치하고 하둡을 설치하는 가장 일반적인 방식은 Mac OS M1에서는 어려울 것으로 보인다. M1에서는 VirtualBox 대신 Parallels와 같은 가상환경을 이용할 수 있다고 하는데, 이렇게 사용하는 경우가 많지 않은 것 같아서 포기했다. 즉, 위에 두 링크 모두 local에 직접 Hadoop을 설치하는 방식이며, 만약 M1에 램 8GB 사용자라면 설치 및 테스트에 문제가 없을지 잘 모르겠다.
<br>

참고로 VirtualBox 설치까지는 문제가 없으나 다음과 같이 Ubuntu를 설치할 때 문제가 발생한다. '잘못된 설정 감지됨'을 확인해보면 '호스트 시스템에서 하드웨어 가속화를 지원하지 않지만, 시스템 페이지의 가속 부분에서 활성화되어 있습니다. 가상 시스템을 시작하려면 해당 설정을 비활성화하세요' 라는 문구와 함께 다음 단계로 넘어갈 수가 없다. 
<br>

![1-error](/!contents_plot/2021-08-16-ERROR-1.jpg){: width="50%"}
<br>

그래서 VirtualBox로 설치하는 방법은 자연스럽게 포기하고 로컬에 설치했다.
<br><br><br><br>

## 2. zsh 사용
에러 발생 시 구글링을 통해 찾는 대부분의 솔루션은 `~/.bashrc` 또는 `~/.bash_profile`에 Path를 추가하는 방법이다. 그러나 M1에서 기본 터미널은 더 이상 bash가 아닌 zsh를 이용하고 있으며 대부분의 Path 파일을 `~/.zshrc` 또는 `~/.zshenv`를 바꿔야 한다. 이미 MAC OS에 적응되신 분들이라면 zsh를 bash로 변경 후 사용하는 경우도 있는 것 같지만, 맥 초보라면 여러가지 파일을 만들기보다는 그냥 `~/.zshrc` 또는 `~/.zshenv`를 변경하는 것이 편한 것 같다.
<br><br><br><br>

## 3. JAVA_HOME Path 설정
JAVA_HOME Path를 제대로 설정했다면 패스해도 되는 부분이다. zsh 사용자는 `~/.zshrc`에 다음을 추가하면 된다. 물론 해당 위치는 사용자의 jdk 위치에 따라 적절히 변경해야 한다. 
<br>

```bash
## JAVA env variables
export JAVA_HOME="/Library/Java/JavaVirtualMachines/jdk1.8.0_301.jdk/Contents/Home"
export PATH=$PATH:$JAVA_HOME/bin
```  
<br>

참고로 사용자가 다운받은 jdk 버전을 이용하기 위해서 기본 Java 설정과 `etc/hadoop/hadoop-env.sh`에 JAVA_HOME이 동일한 jdk를 가리켜야 한다. 저는 jdk 1.8.0_301 버전을 다운받았고, `~/.zshrc`에 JAVA_HOME Path를 설정한 후 터미널에 **java -version**을 입력하면 java version "1.8.0_301"이 출력되어 제대로 설정된 줄 알았었다. 
<br>

하지만 제대로 확인하기 위해서는 터미널에 **/usr/libexec/java_home**를 입력하여 확인하는 것이 가장 좋다. 해당 명령어는 기본 JAVA 위치를 알려주는데, 만약 `/Library/Internet Plug-Ins/JavaAppletPlugin.plugin/Contents/Home`이 출력된다면 M1에서 기본적으로 깔려있는 Java를 가리키고 있는 것이라고 보면 될 것 같다. Path 설정이 잘못된 것을 더욱 정확히 알려면 본인이 다운받은 버전을 추가해서 터미널에 **/usr/libexec/java_home -v "1.8.0"**를 입력했을 때 이전과 다른, 새롭게 다운받은 위치가 나온다면 Path 설정이 다시 필요함을 알 수 있다. 저는 JAVA_HOME 설정 시 오타가 발생해서 제대로 Path 설정이 되지 않았었다.  
<br>

![2-error](/!contents_plot/2021-08-16-ERROR-2.jpg){: width="60%"}
<br><br><br><br>

## 4. Hadoop Path 설정

구글링을 통해 대부분 HADOOP_HOME의 위치로 `/usr/local/Celler/hadoop/3.x.x/libexec`를 가리킨다. 하지만 해당 위치가 맞는지 brew install hadoop으로 하둡 설치 시 출력되는 위치를 반드시 확인해야 한다. 제 경우에는 hadoop 3.3.1 버전이 다운받아졌고, `/opt/homebrew/Cellar/hadoop/3.3.1/libexec/` 위치를 HADOOP_HOME으로 설정해야 했다. 어차피 /etc/hadoop 내 파일 5개를 변경해야 되기도 하므로 실제 Finder에서 command+shift+g로 해당 위치에 직접 들어가서 확인해보는 것이 가장 좋다. 
<br><br><br><br>

## 5. no namenode & Error java.net.ConnectException 

마지막까지 씨름했던 에러이다. start-dfs.sh 후에 JPS를 확인해보면 namenode가 뜨지 않았다. 그리고 **hdfs dfs -ls**를 입력하면 `localhost:9000 failed on connection exception: java.net.ConnectException: Connection refused; For more details see:  http://wiki.apache.org/hadoop/ConnectionRefused` 에러가 출력되었다.  
<br>

![3-error](/!contents_plot/2021-08-16-ERROR-3.jpg){: width="80%"} 
<br>

logs 폴더에서 log를 확인해본 결과 다음과 같은 에러가 출력되었다. `/usr/local/Cellar/hadoop/hdfs/tmp/dfs/name is in an inconsistent state: storage directory does not exist or is not accessible.` 앞의 4번에서도 나왔던 문제로, M1에서는 /usr/local/Celler ~ 위치가 아닌 /opt/~ 위치가 기본으로 어딘가에 Path가 잘못 설정되어있다는 문제로 보였다. 즉, 이는 namenode에 대한 데이터를 저장할 공간을 제대로 가리켜주지 않아서 생기는 에러인 것 같다.  
<br>

![4-error](/!contents_plot/2021-08-16-ERROR-4.jpg){: width="50%"}  
<br>

따라서 구글링을 통해 `etc/hadoop/hdfs-site.xml` 파일에 다음과 같이 property를 추가해서 directory를 설정해주면 문제가 해결된다. 여기서 폴더는 그냥 편한 위치로 설정하기 위해 desktop에 hadoop/hdfs/namenode, datanode 폴더를 만들었고, 이렇게 각각 namenode와 datanode 실행 시 관련 정보를 저장할 공간을 만든 후에 그 위치로 <value> 부분을 지정해주면 된다.  
<br>

```bash
<property>
    <name>dfs.namenode.name.dir</name>
    <value>file:/Users/clue7/desktop/hadoop/hdfs/namenode</value>
</property>
<property>
    <name>dfs.datanode.data.dir</name>
    <value>file:/Users/clue7/desktop/hadoop/hdfs/datanode</value>
</property>
```
<br>

**참고 링크**  
[https://stackoverflow.com/questions/27271970/hadoop-hdfs-name-is-in-an-inconsistent-state-storage-directoryhadoop-hdfs-data][link-3]
[https://stackoverflow.com/questions/22565200/where-hdfs-stores-data][link-4]  
<br>

이전 내용들을 포맷하기 위해서 **libexec/hadoop namenode -format**를 입력하고 **hdfs --daemon start**를 입력하여 namenode만 먼저 실행해보았다. 이렇게 설정하니 더 이상 에러가 출력되지 않고, namenode가 제대로 실행되었다. 또한 생성했던 namenode 폴더에 파일이 생성되는 것을 확인할 수 있었다.  
<br>

![5-error](/!contents_plot/2021-08-16-ERROR-5.jpg){: width="80%"}
<br><br><br><br>

이제 진짜 하둡 명령어를 실행해볼 수 있게 되었다. `WARN util.NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable` 경고를 해결하고 싶었으나 구글링을 통해 찾은 방법들 모두 제대로 작동하지 않았다. 뭔가 놓친 것 같지만 아직은 문제가 없는 것 같아서 일단은 미지로 남겼다 ㅎ

[link-1]: https://techblost.com/how-to-install-hadoop-on-mac-with-homebrew/
[link-2]: https://key4920.github.io/p/mac-os%EC%97%90-%ED%95%98%EB%91%A1hadoop-%EC%84%A4%EC%B9%98/
[link-3]: https://stackoverflow.com/questions/27271970/hadoop-hdfs-name-is-in-an-inconsistent-state-storage-directoryhadoop-hdfs-data
[link-4]: https://stackoverflow.com/questions/22565200/where-hdfs-stores-data