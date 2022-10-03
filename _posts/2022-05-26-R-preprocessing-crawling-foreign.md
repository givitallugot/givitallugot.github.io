---
layout: post
title:  "[R] 외국인 채권 일별 거래량 거래대금 크롤링 한국거래소(KRX) 투자자별 거래실적 / httr, rvest, readr"
date:   2022-05-26 11:50:42
categories: [Preprocessing]
comments: true
---
<br>

한국거래소(KRX)에서 일별 외국인 채권 거래량 또는 거래대금을 크롤링하는 내용을 정리해보았습니다.

(* 아래 잘 정리된 블로그 참고해서 다른 데이터 크롤링에 활용했습니다.)

https://hyunyulhenry.github.io/quant_cookbook/%EA%B8%88%EC%9C%B5-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%88%98%EC%A7%91%ED%95%98%EA%B8%B0-%EA%B8%B0%EB%B3%B8.html


## 사이트 접속

한국거래소(KRX)에서 기본 통계 > 채권 > 거래실적 > 투자자별 거래실적 으로 들어갑니다. 다음과 같은 화면을 확인할 수 있고, 조회기간 6개월을 누른다면 일별 데이터가 아니라 6개월 누적 데이터가 출력됩니다. 따라서 일별 거래실적을 구하기 위해서는 크롤링이 필요합니다.


![1](/!contents_plot/2022-05-22-crawling1-1.jpg){: width="65%"}

<br>
<br>

## 크롤링 정보 확인

다음으로 가장 중요한 크롤링 정보를 확인해보겠습니다. 먼저 개발자 도구를 열겠습니다. 맥에서 단축키는 command + option + i 입니다. 그리고 우측 상단에 다운로드 아이콘을 눌러서 데이터를 다운로드 받습니다. (저는 CSV로 다운받았습니다.) 이때 개발자 도구에 정보를 잘 확인해야 합니다.

![2](/!contents_plot/2022-05-22-crawling1-2.jpg){: width="65%"}

generate.cmd와 download.cmd가 Network 탭에 보입니다. 먼저 generate.cmd를 클릭하고 Headers에 들어가서 Request URL 정보를 기록해둡니다. 그리고 마찬가지로 download.cmd를 클릭하고 Header에 들어가서 Request URL 정보를 기록해둡니다.

<br>

generate.cmd Request URL: http://data.krx.co.kr/comm/fileDn/GenerateOTP/generate.cmd

download.cmd Request URL: http://data.krx.co.kr/comm/fileDn/download_csv/download.cmd

<br>
<br>

![3](/!contents_plot/2022-05-22-crawling1-3.jpg){: width="65%"}

다음은 Payload에 들어가서 정보를 기록해둡니다. 해당 정보들은 크롤링의 OTP 정보로 필요합니다.

locale: ko_KR

bndMktTpCd: TOT

strtDd: 20211125 # 원하는 날짜료 변경

endDd: 20220525 # 원하는 날짜로 변경

money: 3

csvxls_isNo: false

name: fileDown

url: dbms/MDC/STAT/standard/MDCSTAT10301

<br>
<br>

## R 코드

이제 크롤링을 위한 R 코드를 생성합니다. date를 인자로 넘겨 받아서 해당 일자의 거래실적을 크롤링하는 함수를 만들었습니다. 아까 기록해둔 파라미터들을 list 형태로 잘 담아주어야 합니다. 변경하고 싶은 옵션이 있다면 파라미터를 변경해서 사용하면 됩니다.

```R
library(httr)
library(rvest)
library(readr)

get_data <- function(date){
  gen_otp_url = 'http://data.krx.co.kr/comm/fileDn/GenerateOTP/generate.cmd'
  gen_otp_data = list(
    bndMktTpCd = 'TOT',
    strtDd = date,
    endDd = date,
    money = '3',
    csvxls_isNo = 'false',
    name = 'fileDown',
    url = 'dbms/MDC/STAT/standard/MDCSTAT10301'
  )
  otp = POST(gen_otp_url, query = gen_otp_data) %>%
    read_html() %>%
    html_text()
  
  down_url = 'http://data.krx.co.kr/comm/fileDn/download_csv/download.cmd'
  down_csv = POST(down_url, query = list(code = otp),
                  add_headers(referer = gen_otp_url)) %>%
                  read_html(encoding = 'EUC-KR') %>%
                  html_text() %>%
                  read_csv()

  return(down_csv)
}

# 데이터 확인
get_data("20210503")
```

![4](/!contents_plot/2022-05-22-crawling1-4.jpg){: width="50%"}

<br>

데이터가 잘 들어온 것을 확인할 수 있습니다. 이제 일자를 변경해가며 for문을 이용해서 해당 함수를 여러번 수행하여 데이터셋을 만들어봅니다. 저는 외국인, 외국인기타 만 추출하고 그리고 외국인 + 외국인기타 인 합계를 추가했습니다. 그리고 데이터를 일자별로 축적할 BondByForeigner 테이블을 생성하여 각 행이 일자를 나타내도록 변경했습니다. 밑에 코드는 참고용으로 확인하시고, 필요에 맞게 변경해서 사용하시면 됩니다.

<br>

```R
start_date <- as.Date("2021-05-01") # 시작일
end_date <- as.Date("2021-06-30") # 종료일
date_set <- format(seq(as.Date(start_date), as.Date(end_date), by = "day"), format="%Y%m%d") # 데이터 가져올 일련의 날짜 생성

# 외국인 채권 거래실적 축적할 데이터 프레임
BondByForeigner <- data.frame(날짜=c(), 투자자구분=c(), 거래량_매도=c(), 거래량_매수=c(), 거래량_순매수=c(), 
                                거래대금_매도=c(), 거래대금_매수=c(), 거래대금_순매수=c())

for(i in c(1:length(date_set))){
  tryCatch({
    # 30일 단위로 20초 멈추기 (임의 설정)
    if(i%%30 == 0){
      Sys.sleep(20)
    }
      
    down_sector_KS <- get_data(date_set[i])
    down_sector_KS_df <- data.frame(down_sector_KS)
    
    # 날짜 열 추가
    date_df <- data.frame(날짜 = rep(date_set[i], 13))
    down_sector_KS_df <- cbind(date_df, down_sector_KS_df)
    
    # 외국인 + 기타외국인 => '외국인합계' 행 추가
    down_sector_KS_df <- rbind(down_sector_KS_df, cbind(down_sector_KS_df[11,1:2], (down_sector_KS_df[11,3:8] + down_sector_KS_df[12,3:8])))
    down_sector_KS_df[14,2] <- '외국인합계'
    
    # 외국인, 기타외국인, 외국인합계 행만 최종 테이블에 붙이기
    BondByForeigner <- rbind(BondByForeigner,down_sector_KS_df[c(11,12,14),])
  }, error = function(e){cat("ERROR: ", i, "th ", conditionMessage(e), "\n")})
}

print(BondByForeigner)

# 데이터 index 재설정
rownames(BondByForeigner) <- NULL

# 거래량/거래대금 순매도 계산
BondByForeigner$거래량_순매도 <- BondByForeigner$거래량_매도 - BondByForeigner$거래량_매수
BondByForeigner$거래대금_순매도 <- BondByForeigner$거래대금_매도 - BondByForeigner$거래대금_매수

# 컬럼 순서 변경
BondByForeigner <- BondByForeigner[,c(1,2,3,4,5,9,6,7,8,10)]

# 외국인 채권 거래실적 축적할 최종 데이터 프레임 (30일씩만 가져오기 가능)
BondByForeignerTotal <- data.frame(날짜=c(), 투자자구분=c(), 거래량_매도=c(), 거래량_매수=c(), 거래량_순매수=c(),
                                     거래대금_매도=c(), 거래대금_매수=c(), 거래대금_순매수=c())
BondByForeignerTotal <- rbind(BondByForeignerTotal, BondByForeigner)
```

<br>

![5](/!contents_plot/2022-05-22-crawling1-5.jpg){: width="40%"}

<br>

결과 테이블은 위와 같습니다. for문에서 30일 반복 수행했을 때 매크로로 인식하는 것인지 더 이상 크롤링이 되지 않습니다. 그래서 tryCatch문으로 30번 반복마다 임의로 20초 정도 멈추도록 코드에 작성했습니다.

<br><br>
