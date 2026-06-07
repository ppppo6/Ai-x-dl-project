# 태양광 발전량 예측
AI+X 딥러닝 프로젝트 (G28)

---

Members:

한민채 (경영학과) gksalsc00@hanyang.ac.kr

양희지 (전기공학전공) yangheeji0920@gmail.com

전석호 (정치외교학과) jseokho31@naver.com

김도윤 (데이터사이언스학부) doyun0337@naver.com

---

### 목차 

- [I. Proposal](#i-proposal-option-a)
- [II. Datasets](#ii-datasets)
- [III. Methodology](#iii-methodology)
- [IV. Evaluation & Analysis](#iv-evaluation--analysis)
- [V. Related Work](#v-related-work-eg-existing-studies)
- [VI. Conclusion](#vi-conclusion-discussion)

---

## I. Proposal (Option A)

### 이 프로젝트를 하는 이유  

 태양광 에너지는 탄소 배출 없이 전력을 생산할 수 있는 핵심적인 재생에너지로, 전 세계적으로 빠르게 보급되고 있습니다. 그러나 태양광 발전량은 일사량, 기온, 습도 등 기상 상황에 따라 변동이 크기 때문에 정확한 예측이 어렵습니다.
태양 일사량을 미리 예측할 수 있다면 다양한 분야에서 활용이 가능할 것입니다.

### 최종적으로 어떤것을 얻으려 하나?  

 다양한 시퀀스 모델들을 통해 태양광 발전량을 예측해보면서 어떤 모델이 시계열 예측에 가장 적합한지 분석하는 것을 목표로 하려 합니다.
또한 이를 통해 다양한 모델들의 특징과 장단점에 대해서 알아보려고 합니다.

---

## II. Datasets

 이 프로젝트를 위해서 저희가 사용한 데이터셋은 2개입니다.

태양광 발전량을 예측할려면 **실제 발전소**에서의 발전량이랑 그 발전소 주위의 **기상 상황**에 대한 정보가 필요합니다. 

하지만 '단일 발전소에서의 발전량 / 그 주위의 기상 상황 / 최소 1년치 이상' 을 한번에 만족시키는 데이터셋을 찾기 어려웠습니다.

따라서 동일한 조건 아래 입력 변수와 예측 대상에 대한 데이터셋을 따로 구해서 합치는 작업을 진행했습니다.

---

 ## 사용한 데이터셋 1) NIST - Photovoltaic Data ##

### 데이터셋 설명 ###

 이 데이터셋은 미국 메릴랜드주 게이더스버그에 위치한 미국 국립표준기술연구소(NIST)에 설치된 태양광 어레이에서 수집된 데이터입니다. 여기서 태양광 어레이란 태양광 패널 여러개를 직렬 또는 병렬로 연결한 것입니다.

이 데이터셋의 측정 간격은 1분이며, NIST 사이트에는 6곳에서 측정한 데이터가 존재해서 저희가 원하는 1곳을 골라서 따로 데이터를 얻을 수 있습니다. (Roof로 선택함)

또한 이 데이터셋은 무료로 오픈되어있어 누구나 사용할 수 있습니다.

<img width="200" height="200" alt="image" src="https://github.com/user-attachments/assets/22f36a87-af28-40e2-94a9-c47ef0cd93c4" />

  (태양광 발전소 사진)

  --

### 데이터셋 특징 ###

원래 이 데이터셋에는 발전량 이외에도 많은 변수들(ex. 장비 상태)이 있습니다.

하지만 저희가 여기서 필요했던 것은 발전량이였기 때문에 1개의 변수만 사용했습니다.

| 변수 | 변수명 | 설명 |
| --- | --- | --- |
| 발전량 | InvPAC_kW_Avg | 인버터를 통해 변환된 실제 교류(AC) 출력 전력의 평균값이다. 1분마다 측정되었으며 단위는 kW이다. |

여기서 인버터란 태양광 패널이 생산한 직류(DC) 전기를 실생활에서 사용 가능한 교류(AC)로 변환하는 장치입니다.

---

 ## 사용한 데이터셋 2) NSRDB - Weather Data ##

### 데이터셋(사이트) 설명 ###

 이 데이터셋은 미국 국립재생에너지연구소(NREL)에서 공개한 태양 복사량 및 기상 상황 데이터베이스입니다. (NSRDB: National Solar Radiation Database)

NSRDB 사이트의 특징 중 하나는 자신의 원하는 지역의 데이터를 자기가 원하는 변수만 골라서 데이터를 얻을 수 있다는 점입니다.

 이를 이용해 저희는 태양광 발전량이 측정된 장소 (미국 게이더스버그)에서의 데이터를 얻을 수 있었습니다. 또한 여러 변수들 중에서 태양광 발전량에 크게 영향을 미친다고 판단한 변수 6개를 골라서 데이터셋을 최종적으로 얻었습니다.

 ### 데이터셋 특징 ###

| 변수 | 변수명 | 설명 |
| --- | --- | --- |
| 기온 | Temperature | 그 지역의 기온을 나타내며, 기온이 높으면 패널 효율이 떨어져 발전량이 낮아질 수도 있다. <br> (단위 : °C) |
| 맑은 하늘 기준 일사량 | Clearsky GHI | 실제 일사량(GHI)과 비교해 구름 영향 파악하기 위해 쓰였다. <br> (단위 : W/m²) |
| 실제 일사량 | GHI | 실제 지면에 닿는 태양 복사량을 뜻하며 발전량과 가장 직접적으로 비례한다. <br> (단위 : W/m²) |
| 상대 습도 | Relative Humidity | 상대 습도가 높을수록 구름or안개 가능성이 높아 발전량 감소 가능성도 높아진다. <br> (단위 : %) |
| 기압 | Pressure | 기압 패턴으로 날씨 변화 예측이 가능하다. <br> (단위 : mbar) |
| 풍속 | Wind Speed | 패널 냉각 효과로 인해 발전 효율에 영향을 끼칠거라고 생각하였다. <br> (단위 : m/s) |

---

## III. Methodology

 저희가 사용할 모델은 RNN, LSTM, Transformer 입니다.  
태양 일사량은 시간 순서에 따라 변화하는 시계열 데이터이므로, 순서가 있는 데이터 처리에 특화된 시퀀스 모델 3가지를 선택하여 비교 분석할 예정입니다.

### **RNN** (순환 신경망)  

 RNN은 일반 신경망(MLP)과 달리 이전 출력을 다시 입력에 넣는 순환 구조를 가집니다.  
RNN은 순서가 있는 데이터를 처리하는데 강점을 보입니다. (ex. 시계열 데이터) 또한 RNN의 학습속도가 빠르다는 장점도 있습니다.
하지만 RNN은 치명적인 문제들이 있는데 그 중에 하나는 순서가 긴 데이터에선 앞부분 기억을 까먹는 문제가 있다는 것입니다.  
이것을 기울기 소실 문제 (Vanishing Gradient) 라고 부릅니다.

                                                                                          
### **LSTM** (Long short-term memory)  

 LSTM은 발전된 RNN 기법중 하나에 속합니다.  
기존의 RNN에 Cell State와 Forget Gate, Input Gate, Output Gate를 추가하여 기존 RNN의 기울기 소실 문제를 해결하였습니다.
따라서 LSTM은 장기적인 패턴을 학습할 수 있다는 장점이 존재합니다.

### **Transformer**  

 Transformer는 앞의 2개의 모델에 비해 비교적 최근에 나왔습니다.  
Transformer는 RNN 계열 모델과는 다른 방식을 이용합니다. -> Self Attention
따라서 Transformer는 병렬 처리가 가능하고 데이터가 많을수록 효과적입니다.

{공정한 비교를 하기 위해서 딥러닝 모델을 제외하고 Epoch나 Batch Size, Train/Test 비율 같은 것들은 모두 동일하게 진행할 예정입니다}

{더욱 자세한 설명을 추가할 예정입니다}

---

### 사용할 DL 라이브러리 : PyTorch

{PyTorch에 관한 설명 추가할 예정입니다}

---

## IV. Evaluation & Analysis

```python
print('Work-In-Process')
```
### !!임시 - Blog_In_Progress 제출용!! LLM을 이용해 간단히 코드를 짜서 Transformer를 이용해 그래프 도출해봤습니다 -> 수정 예정

<img width="1500" height="1100" alt="image" src="https://github.com/user-attachments/assets/814eed5a-ff2f-4bc5-bfdf-1e6111902568" />
(걸린시간 : 약 20분)

{평가 지표 : MAE, RMSE, R^2 ??}

---

| 이름 | 나이 | 직업 |
| --- | --- | --- |
| 철수 | 25 | 개발자 |
| 영희 | 30 | 디자이너 |

## V. Related Work (e.g., existing studies)

사용한 데이터셋 :  

https://pvdata.nist.gov/

Transformer 관련 내용 :  강의자료 07-beyond

---

## VI. Conclusion: Discussion

{3개 모델 중 어느 것이 가장 정확했는지 확인할 예정입니다}
{또한 각 모델마다 두드러지게 나타나는 특징들을 확인할 예정입니다}
