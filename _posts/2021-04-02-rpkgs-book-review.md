---
layout: post
title:  "[Review/Book] R Packages (2nd Edition) R 패키지 만들기의 기본서 리뷰 및 정리 1 with devtools, roxygen2, create_package, 5 package states"
date:   2021-04-02 12:23:40
categories: [Other topics]
comments: true
---
<br>
Hadely Wickham의 책 R packages의 Second Edition을 읽으며 리뷰했다. 사실 리뷰라고 보기엔 정리에 가깝지만.. 아직 한국어 번역본은 따로 없으며 1판의 한국어 번역본은 존재하지만 굉장히 많은 부분이 업데이트 되어 2판으로 출시되었다고 한다. 책 전문은 [해당 사이트][book-site]에서 확인할 수 있다 :D
<br><br><br><br>

## [R Packages, Hadley Wickham and Jenny Bryan][book-site]
### Part I. Getting started

#### chapter 1: Introduction
- 이 책은 꼭 패키지를 만들기 위한 목적이 아니더라도 R 코드를 Organize 하기 위한 Conventions를 배우기 좋은 책이다. 패키지를 만들 때 가장 중요한 철학은 automation, 최대한 직접 할 일을 줄이는 것이라고 말한다.

- R 사용자라면 어디선가 봤을 conventions의 의미는 다음과 같다. `foo() = functions, bar = variables and function, baz/ = paths`

- 가장 핵심이 되는 세 패키지: `devtools`(개발에서 많은 부분을 자동화시켜주는 일종의 메타 패키지), `roxygen2`(패키지 내 함수를 문서화하기 위함), `testhat`(테스트를 위함)
<br><br><br><br>

### chapter 2: The whole game

- 패키지 개발을 위한 root directory 설정 시 주의해야 한다. 이미 git과 연결된 폴더거나, R package가 깔려있거나, R project가 있는 폴더는 적절하지 않다.

- **create_package()**를 통해 새로운 폴더에 initialize를 수행하는데, R studio에서 실행하면 새로운 R studio 창이 생기고 해당 폴더에 기본 파일들이 생성된다. `(ex .Rbuildignore, .gitignore, DESCRIPTION, NAMESPACE, R directory, dirname.Rproj)` 특히 R 폴더 밑에 **use_r()** 함수를 이용하여 사용자 정의 함수를 작성하면 편하다.

- R/ 밑에 함수를 정의할 때 파일명은 해당 함수 이름이며, 함수 정의에 필요한 arguments에 실제 상수 정의나 패키지 로드 코드까지 작성할 필요는 없다.

- **load_all()** 또는 **source('R/fbind.R')** 코드를 통해 정의한 함수를 불러오면 된다.

- `DESCRIPTION`에는 패키지에 대한 metadata가 들어간다.

- RStudio에서 Ctrl + . 단축키는 현재 디렉토리의 다른 파일을 검색하여 내용 수정 시 용이하다.

- **?**을 통해 볼 수 있는 help 파일은 `roxygen2` 라이브러리로 Code 탭에 **Insert Roxygen2 Skeleton**을 통하여 편리하게 작성할 수 있고, man 폴더 또한 생성된다. help를 작성 후 **document()**를 통해 man/ 밑에 함수명으로 .Rd 파일을 생성하면 된다.
<br>

![image-1](/!contents_plot/2021-04-02-rpkgs-fbind.jpg){: width="50%"}
<br>

- **install()**을 통해 만든 패키지를 설치하고 library(패키지명)으로 불러올 수 있다. 이때 패키지명은 당연하지만 **create_package()**를 수행한 위치의 폴더명과 동일하다.

- 다양한 input 값에 따른 올바른 결과가 도출되는지 테스트하기 위해 **use_testthat()**으로 testing machinery를 initialize 한다. 그리고 **use_test()** 함수로 생성된 tests 폴더와 이 폴더 내 코드에 테스트 예시를 작성한다. 작성이 끝나면 **library(testthat)**을 로드하고 **test()** 함수를 통해 테스트를 진행한다. 
<br>

![image-2](/!contents_plot/2021-04-02-rpkgs-test.jpg){: width="45%"}
<br>

- 패키지 생성 시 다른 패키지의 함수를 import하여 이용할 수도 있는데, 이 경우 **use_package()** 함수를 이용하여 명시해주면 `DESCRIPTION` 파일에 해당 내용이 기록된다. 그리고 함수에서 **패키지명::함수** 형식으로 다른 패키지 함수를 이용하면 된다.

- github에 commit할 때 주로 프로젝트 설명을 하는 README.md 파일을 **use_readme_rmd()**로 자동 생성할 수 있다.

- 항상 **check()** 함수를 통해 패키지를 검증하고 테스트한다. ~~일종의 테스트 및 ctrl+s로 생각하면 되나보다.~~
<br>

![image-3](/!contents_plot/2021-04-02-rpkgs-check.jpg){: width="70%"}
<br><br><br><br>

### Chapter 3: System setup

- 패키지를 만들기 위해서는 최신의 R(4.0.5 이상)과 다음과 같은 패키지`(devtools, roxygen2, testthat, knitr)`를 준비해야 한다.

- 점점 무거워지는 `devtools` 패키지의 denpendency 이슈를 줄이고 더욱 편리하게 만들기 위해 2.0.0 이후 기능별로 sub-package로 나눠졌다. 더 자세한 내용은 [conscious uncoupling][link-devtool]에서 확인할 수 있다.

- 특히 개발하는 입장에서, `devtools` 패키지에 **load_all()** 함수를 사용할 때, `devtools` 밑의 함수로 불러오기 보다는 sub-package인 **pkgload()**를 명시하여 사용하는 것을 추천한다.

{% highlight R %}
# 방법1: 비추천
library(devtools)
load_all()

# 방법2: 추천
pkgload::load_all()
{% endhighlight %}
<br>

- .Rrofile 내에 `usethis`를 이용하여 패키지를 만든 본인에 대한 personal defaults를 추가할 수 있다. 예시는 다음과 같다.

{% highlight R %}
options(
  usethis.full_name = "Jane Doe",
  usethis.description = list(
    `Authors@R` = 'person("Jane", "Doe", email = "jane@example.com", role = c("aut", "cre"), 
    comment = c(ORCID = "YOUR-ORCID-ID"))',
    License = "MIT + file LICENSE",
    Version = "0.0.0.9000"
  ),
  usethis.protocol  = "ssh"  
)
{% endhighlight %}
<br>

- 패키지를 만들 때 C 또는 C++을 함께 사용하고 싶은 경우, 각 OS마다 다른 방법으로 준비할 수 있도록 한다. macOSC의 경우 **xcode-select --install**을 통해 `Xcode`를 설치한다. ~~github 사용할 때와 마찬가지.~~ 
<br><br><br><br>

### Chapter 4. Package structure and state

- 패키지 개발 상태는 크게 5가지 상태 `source, bundled, binary, installed, in-memory`로 나뉜다. 이미 잘 알고 있듯이 **install.packages()** 또는 **devtools::install_github()**는 `source, bundled` 또는 `binary` 상태의 패키지를 `installed` 상태로 변환하는 것이다. 또한 **library()**로 패키지가 `in-memory` 상태가 된다.
<br><br>

> `Source package`: **just a directory of files with a specific stucture**, 즉 .R 파일들이 포함된 R/ 폴더, DESCRIPTION 파일 등을 말한다. 이 책의 남은 대부분의 Chapter에서 이 Source의 디테일을 다룬다. 
<br>

> `Bundled package`: **been compressed into a single file**, 즉 .tar.gz로 하나의 파일로 압축된 Source를 의미한다. ~~이는 여러 파일들이(.gz) gzip을 이용하여 압축되었음(.gz)을 의미하는 Linux의 convention을 따르는 이름일 뿐이다.~~ CRAN에서 Package source: packagename_version.tar.gz 형식으로 한 파일로 다운로드 가능한 것을 확인할 수 있다.  
압축을 푼 Bundle package는 Source package와 거의 비슷한데, Bundle은 vignettes가 구현되어 있고, src/ 폴더의 local source package가 없고, .Ruildignore이 포함되어 있지 않다는 차이점이 존재한다.
<br>

> `Binary package`: **패키지 개발 툴이 없는 R 사용자를 위한 하나의 압축된 파일**이다. Windows와 macOS 버전으로 나뉜다. CRAN에서 Windows binaries: 또는 macOS binaries: 로 업로드되어있어 다운로드 가능한 것을 확인할 수 있다.  
압축을 푼 Binary package는 R/ 폴더 밑에 .R 파일이 없으며, Meta/ 폴더에는 여러 개의 .rds 파일이 포함되어있고 이는 DESCRIPTION의 parsed version으로 생각할 수 있다. 기존의 help 내용과 src/ 밑의 코드, inst/ 폴더 등의 위치가 변경되고, 그 외에도 사라지거나 추가된 파일들이 있다.
<br>

> `Installed package`: **a binary package that's been decompressed into a package library**, 아래 책의 도식화를 참고하면 어떤 방식으로 패키지를 다운받을 수 있는지 참고하기 좋다.

![image-4](/!contents_plot/2021-04-02-rpkgs-plot1.png){: width="55%"}
*<center> 출처: FIGURE 4.2/ https://r-pkgs.org/package-structure-state.html#structure-binary </center>*

<br>

> `In-memory package`: **library()를 통해 R 상에서 사용 가능한 패키지**를 의미한다. 일반적으로는 **library()**를 이용하나, 패키지를 만드는 입장이라면 **devtools::load_all()**을 사용하여 `source → in-memory`로 바로 올려서 효율을 높이는 것이 좋다.
<br><br>

- 이 Chapter의 마지막 절에서는 **library**와 **package**의 차이를 설명한다. ~~library() 함수로 인해 이 둘을 혼용해서 사용하는 경우가 많으나~~ **정확히는 library() 함수를 통해 package를 로드하는 것**이며, 이 둘의 차이를 아는 것이 패키지 개발자의 입장에서는 중요하다.

- (MacOS 기준) library는 일종의 폴더을 의미하며 **user library**와 **system-level(or global library)**가 존재한다. 다운받은 `package`들은 `library`에 존재하며, 한 가지 주의해야 하는 것은 패키지 내부에는 절대 **library()**를 사용해서는 안된다.
<br><br>

[출처] https://r-pkgs.org/ 
<br><br><br><br>

아직 책을 정독한게 아니라서 잘못 해석한 부분이 있을 수 있다는 점.. 참고 바랍니다. 다음 포스팅에서 이어서 리뷰할 예정입니다.



[link-devtool]: https://www.tidyverse.org/blog/2018/10/devtools-2-0-0/#conscious-uncoupling
[book-site]: https://r-pkgs.org/ 