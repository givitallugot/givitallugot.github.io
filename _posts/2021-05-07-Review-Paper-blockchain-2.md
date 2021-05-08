---
layout: post
title:  "[Review/Paper] 블록체인 논문 리뷰 2 with 스마트 컨트랙트 보안 이슈 및 해결 방안"
date:   2021-05-08 02:00:31
categories: [Other topics]
comments: true
---
<br>
(스마트 컨트랙트 보안 이슈 관련 논문에 대한 짧은 리뷰입니다.)

#### Security, Performance, and Applications of Smart Contracts: A Systematic Survey, S Rouhani, R Deters
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>[click to browse the paper][paper-1] 
<br><br>

발제 준비를 하며 읽게된 논문인데, 스마트 컨트랙트의 잘 알려진 보안 이슈들(Acess Control, Reentrancy, Arithmetic Issues, DoS, Bad Randomness, Front Running, Time Manipulation, Short Address Attack)을 범주화하고 해결 방안에 관한 연구들을 정리한 논문이다.
<br><br>

--- 
<br>

##### Abstract & Introduction (연구 배경 및 목적)
<br>

기본적으로 스마트 컨트랙트 제 3자 없이 신뢰성 있는 트랜잭션 이행한다. 그런데, 이러한 속성으로 인해 오히려 많은 보안 이슈에 노출되어있고 이로 인한 막대한 손실 발생 가능성 존재하기도 한다고 지적하며 다음과 같은 사건들을 나열한다.
<br>

> 2016.06 DAO 해킹 사례 (이더리움 370만 개 도난)  
> 2017.11 Parity Wallet 해킹 사례 (이더리움 15만 개 도난)  
> 2018.04 BEC(Business Email Compromise) attack (약 9억 달러 손해)

<br>

스마트 컨트랙트에서 강조하는 보안`(security)`과 신뢰`(reliability)`에는 사실 이면이 존재하는데,
1. 정적인 프로그램이므로 반드시 정확성이 선행되어야 한다. 스마트 컨트랙트에서 조건에 부합하면 계약이 자동으로 이행되기 때문에 중간에 멈추고 수정하기는 어렵다.
2. 컨트랙트 작성 시에는 몰랐던 보안 이슈가 컨트랙트 수행 도중 드러날 수 있다.
<br><br>

따라서 해당 논문은 최신 스마트 컨트랙트의 보안 이슈 및 해결방법 관련 논문 53개를 분석한다. 크게 **보안성 확보**`(security assurance)`과 **정확성 검증**`(correctness verification)` 두 가지 범주로 보인 이슈를 나눠서 분석하게 된다.
<br><br><br><br>

---
<br>

##### A Synoptic Overview (연구 개요)
<br>

1. `'security verification'` 키워드를 통해 다양한 전자저널에서 논문 발췌 - 총 258개 (2018.12 기준)
2. 신생 주제로 논문의 수가 적지만, 매년 증가하고 있어 필수적이고 유망한 주제
<br><br>

이 논문의 뼈대가 되는 두 범주에 대해 각각 20개, 33개 논문을 분석한다.
<br>

**[ 보안성 확보`(security assurance)` ]**  

총 20개 논문 취합, 3개 주제로 분류 - 보안 환경(environment security), 취약성 탐색(vulnerability scanning), 성능 효과(performance impacts)

**[ 정확성 검증`(correctness verification)` ]**  

총 33개 논문 취합, 2개 주제로 분류 - 프로그래밍 정확도(programming correctness), 정형 검증(formal verification)

* formal method = formal specification(정형 명세) + formal verification(정형 검증)
<br><br><br><br>

---
<br>

##### Security Assurance of Smart Contract (스마트 컨트랙트에서 보안성 확보)
<br>

**A. 보안 환경(Environment Security)**
<br>

: 1) 블록체인 보안

> Alharby and Moorsel [10]: 블록체인의 공공연한 위험은 computing power, private key protection, criminal activities, double payment, and transaction information disclosure 등이라고 지적  
> Li et al. [11]: 크게 트랜잭션 순서 종속성(Transaction Order Dependency)과 타임스탬프 종속성(Timestamp dependency) 두 가지 문제를 지적, OYENTE를 해결방안으로 제시  

<br>

: 2) 데이터 소스: 스마트 컨트랙트는 일반적으로 HTTPS로 외부 데이터를 가져오기 위한 연결이 어려움

> Zhang et al. [12]: Town Crier(TC)를 도입하여 HTTPS 데이터 소스를 스마트 컨트랙트와 연결, 신뢰할 수 있는 데이터 피드 부족 문제를 해결

<br><br>

**B. 취약성 탐색(vulnerability scanning)**  
취약성을 탐색하기 위함이며, 특히 Reentrancy, Access control, DoS(Denial of Service), bad randomness 취약성으로 큰 손실이 발생 가능하다.
<br>

: 1) 특정 취약성을 위한 해결방안

> Bissias et al. [14]: (DAO 사건처럼) DAPP에 의한 도난당한 자산을 회수하는 매커니즘 제안  
> Marcus et al. [15]: Kademlia 프로토콜의 artifact를 제거함으로써 공격의 문턱을 높일 것, eclipse 공격에 대비  
> Liu et al. [16]: Reentrancy 취약성 탐지를 위한 ReGuard 제시, reentrant 오류에 자동으로 플래그 지정을 통해 동적으로 재진입 식별  
> Torres et al. [18]: 이더리움에서 정수 버그를 정확히 찾아내는 Osiris를 제안  

<br>

: 2) 일반적인 취약성을 위한 해결방안

> Bragagnolo et al. [20]: SmartInspect 아키텍처 제안, 배포된 스마트 컨트랙트의 역컴파일을 통해 비구조화 정보를 얻어 상태를 파악  
> SolMet [22], MAIAN [24], ZEUS [25], Securify [26], Mythril [50], SmartDec [51], and Solgraph [52]: 전자 5개는 실행 중 취약성 탐색을 위해 EVM 바이트 코드 분석 도구, 후자 2개는 코드를 분석하는 도구  

<br><br>

**C. 성능 효과(performance impacts)**  
1초에 최대 10~20개 트랜잭션이 처리되는데, Abdellatif and Brousmiche [67]에 의하면 miner's pool에 오래 머무를수록 공격에 당할 가능성이 높다. 따라서 속도를 높임으로서 스마트 컨트랙트의 효율성을 향상시킬 방법을 강구한다.

<br>

> Vukolic [28]: 성능을 높이기 위해 독립적인 스마트 컨트랙트가 병렬적으로 수행되는 방법 제시  
> Dickerson et al. [29]: 광부들이 트랜잭션을 스케줄링하고 서로 충돌하지 않는 스마트 컨트랙트는 병렬적으로 수행하는 방법 제안  

<br><br>

**Discussion of Security Assurance(보안성 확보에 대한 정리)**  
<br>

A. 보안 환경(Environment Security)
- **성과**: 가능한 위험성 탐색, Town Crier(TC)로 데이터 소스와의 안전한 연결
- **한계**: 여전히 연구가 미숙하고 효과는 아직 미흡
<br>

B. 취약성 탐색(Vulnerability Scanning)
- **성과**: 같은 실수를 피할 수 있음, 가능한 다양한 해결방법이 제안됨
- **한계**: 이미 알려진 취약점만 분석이 가능함, 제시된 해결방법이 큰 규모의 스마트 컨트랙트에서는 비효율적
<br>

C. 성능 효과(Performance Impacts)
- **성과**: 과거의 병렬 실행 기법들을 바탕으로 스마트 컨트랙트에서도 제시
- **한계**: 이더리움이 여전히 가장 인기있으며 병렬 실행이 아직은 불완전함
<br><br><br><br>

---
<br>

##### Correctness Verification of Smart Contrats (스마트 컨트랙트에서 정확성 검증)  
프로그래밍에 단일화된 방법이 존재하지 않고 언어의 제한이 없기 때문에 프로그래밍 시 발생하는 다양한 취약점 존재하는데, 프로그래밍 중 발생 가능한 실수를 최소하고 개발 진행 효율성을 높이는 방안의 연구들이다.  

<br>

**A. 프로그래밍 정확성(Programming Correctness)**
<br>

: 1) 표준 설정(Setting standards)

> Delmolino et al. [30]: 코드를 작성하는 과정에서 빈번히 일어나는 실수를 찾기 위해 대량의 코드를 분석  
> [32], [33]: 시맨틱 프레임워크를 설계하기 위해 이론과 계산법을 결합  
> Marino and Juels [34]: 변경 및 삭제를 위한 표준을 제시  

<br>

: 2) 새로운 컨트랙트 언어 개발(Developing New Contract Language)

> 정확한 코드를 효율적으로 작성할 수 있도록 하는 것이 목적  
> Idris [35], Simplicity [36], Obsidian [39], and Flint [37]: 다양한 Type-based functional language가 제시되어 안전한 개발을 도모  
> Idelberger et al. [40]: logic language라는 새로운 아이디어 제시  
> Regnath and Steinhorst [38]: SmaCoNat이라는 사람이 읽을 수 있고 안전한 언어를 제시, 이 언어는 프로그래밍 문법을 자연어 문장으로 변환하여 주소와 직접 연결하기보다 변수명을 전달  

<br>

: 3) 의미론적 분석(Semantic Analysis)

> Hildenbrandt et al. [41]: EVM의 Bytecode stak-based 언어인 KEVM으로 실행 가능한 정형 명세 방법을 제시, 스마트 컨트랙트에서 정형 분석을 더욱 쉽게  
> Zhou et al. [43]: SASC 정적 분석 도구를 제시하여 어느 위치에서 취약성이 존재하는지 topology map 생성을 통해 마킹  
> Liu et al. [45]: 의미론적 S-gram을 통해 이더리움을 위한 보안성 검사 기법을 제시, N-gram 언어를 의미론적 모델과 결합 후 불규칙한 토큰 시퀀스에 대해 잠재적인 취약성으로 판단

<br>

: 4) 소프트웨어 엔지니어링 도구(Software Engineering tools): 개발을 위해 일반적으로 받아들여진 표준을 바탕으로 컨트랙트 프로그래밍 절차를 제공하는 도구

> Porru et al. [47]: 블록체인 기반의 소프트웨어를 개발하고 테스트하는 도구 제시  
> Marchesi [49]: 새로운 분야에서 활용되는 블록체인 기반의 소프트웨어 개발 도구 제시 

<br><br>


**B. 정형 검증(Formal Verification)**  
정형 기법은 스마트 컨트랙트 검증에도 많이 사용한다.
<br>

: 1) 프로그램 기반 정형 검증(Program-Based Formal Verification)

> The 3rd Global Blockchain Summit: 정형 검증 플랫폼인 VaaS 등장, 이는 Solidity 스크립트 언어를 Coq 코드로 변환 후 정확성을 검증
> Bhargavan et al. [55]: 비슷하게, Solidity 언어를 F* 언어로 변환 후 정확성을 검증
> Yang and Lei [60], [61]: FSPVM(은 Coq 언어의 Hoare logic을 이용하여 스마트 컨트랙트의 신뢰성과 보안성을 검증, Lolisa 언어와 그 인터프리터 이용
> Hirai [56]: 이더리움 Bytecode를 바탕으로 정형 기법 수행을 제시, Isabelle/HOL을 이용해 바이너리 이더리움 코드를 검증

<br>

: 2) 행동 기반 정형 검증(Behavior-Based Formal Verification)

> Ellul and Pace[64]: 런타임 검증 방법을 제시, 위반 당사자가 정확한 행동에 대한 보험을 제공
> Bigi et al. [65]: 기존의 정형 기법에 게임 이론을 결합하여 확률적 정경 모델로 스마트 컨트랙트 검증을 제안, 모델은 PRISM 이라는 도구를 이용해서 검증
> Abdellatif and Brousmiche [67]: 프로그램과 환경 간의 상호작용 사이의 검증을 제시, BIP 프레임워크로 모델의 구성요소를 나눈 후, SMC 도구로 모델이 특정 속성을 만족하는지 검증

<br><br>

**Discussion of Correctness Verification(정확성 검증에 대한 정리)**  
<br>

A. 프로그래밍 정확성(Programming Correctness)
- **성과**: SASC와 같은 개발 툴로 효율성 증대, 스마트 컨트랙트의 정확도 높임
- **한계**: 통합된 하나의 프로그래밍 프레임워크는 존재하지 않음, 새롭게 개발된 언어의 취약성은 발견이 어려움, 언어 변환 시 변환 전 의미와 달라질 수도 있음
<br>

B. 정형 검증(Formal Verification)
- **성과**: 수학적 방법을 이용하여 엄격하게 검증, 논리 구조에 대한 정적인 분석과 실행 시 발생하는 동적 검증이 모두 이루어짐
- **한계**: 모델의 정확성을 검증하기 어려움, 모델 점검과 이론 증명은 신생 단계
<br><br><br><br>

---
<br>

##### Conclusion (결론)
<br>

**이 논문의 중요한 세 가지 의의**
<br>

1. 53개 보안성 검증 관련 최신 논문에 대한 깊이있는 분석  

2. 보안 이슈에 따른 분류 - Security Assurance(보안성 확보), Correctness Verification(정확성 검증)  

3. 현주소와 연구가 진행되어야 할 방향 제시  
    1) 스마트 컨트랙트의 보안성과 정확성에 관심이 증대  
    2) 보안성 측면에서 취약성 탐색(vulnerability scanning)이 많이 사용되며 상당한 성과를 거둠  
    3) 정확성 측면에서 프로그래밍 정확성에 많은 연구의 초점이 있고, 앞으로 프로그래머의 보안 문제 인식에도 강조가 필요  
    4) 가장 트랜드한 주제는 정형 기법이며 수학적 모델을 바탕으로 엄밀하게 검증, 앞으로 완성형 정형 검증 도구 설계에 관한 연구가 필요  
<br><br><br><br>

---
<br>

스마트 컨트랙트 보안 이슈에 대한 해결책을 찾고 있다면 정말 많은 도움이 될 논문이라고 생각한다. 정리된 번호는 원래 논문의 Reference 이며 필요한 내용을 구체적으로 얻기 위해 해당 번호의 논문을 따라가서 읽으면 정말 많은 도움이 될 것 같다.

<br><br>

[출처] Rouhani, S., & Deters, R. (2019). Security, performance, and applications of smart contracts: A systematic survey. IEEE Access, 7, 50759-50779.

[paper-1]: https://ieeexplore.ieee.org/abstract/document/8689026/