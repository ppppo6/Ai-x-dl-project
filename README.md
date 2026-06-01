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

태양

---

## II. Datasets

사용한 데이터셋) **Solar Radiation Prediction**  (Kaggle에서 찾음)
이 데이터셋은 NASA의 해커톤에서 제공되었으며, 미국 하와이의 HI-SEAS 기상 관측소에 수집된 기상 데이터입니다.  
또한 이 데이터셋은 단일 지점에서 시간 순서대로 기록된 시계열 데이터셋입니다.  


컬럼에는 UNIXTime, Data, Time, Radiation, Temperature, Pressure, Humidity, WindDirection(Degrees), Speed, TimeSunrise, TimeSunset 이 있으며  
이 중에 저희가 사용할 컬럼들은 Temperature (기온), Humidity (습도), Pressure (기압), Speed (풍속) 입니다. 



!!위의 데이터셋은 Blog 진행과정을 위해 임시로 넣은 데이터셋입니다. 우선 이 데이터셋은 크게 2개의 문제점을 지닙니다. 첫 번째로 관측기간이 4개월로 매우 짧다는 것입니다. 관측 기간이 짧으면 RNN, LSTM, Transformer 모델들 사이의 차이가 두드러지게 안 날 것입니다. 두 번째로 





---

## III. Methodology

### **RNN** (순환 신경망)  

 RNN은 일반 신경망(MLP)과 달리 이전 출력을 다시 입력에 넣는 순환 구조를 가집니다.  
RNN은 순서가 있는 데이터를 처리하는데 강점을 보입니다. (ex. 시계열 데이터) 또한 RNN의 학습속도가 빠르다는 장점도 있습니다.
하지만 RNN은 치명적인 문제들이 있는데 그 중에 하나는 순서가 긴 데이터에선 앞부분 기억을 까먹는 문제가 있다는 것입니다.  
이것을 기울기 소실 문제 (Vanishing Gradient) 라고 부릅니다.

                                                                                          
### **LSTM** (Long short-term memory)  

 LSTM은 발전된 RNN 기법중 하나에 속합니다.  
기존의 RNN에 Cell State와 Forget Gate, Input Gate, Output Gate를 추가하여 기존 RNN의 기울기 소실 문제를 해결하였습니다.

### **Transformer**  

 Transformer는 앞의 2개의 모델에 비해 비교적 최근에 나왔습니다.  
Transformer는 RNN 계열 모델과는 다른 방식을 이용합니다. -> Self Attention

 
{더욱 자세한 설명을 추가할 예정입니다}
---

## IV. Evaluation & Analysis

```python
print('Work-In-Process')
```

---

## V. Related Work (e.g., existing studies)

사용한 데이터셋 :  

(https://www.kaggle.com/datasets/dronio/SolarEnergy)

---

## VI. Conclusion: Discussion

{3개 모델 중 어느 것이 가장 정확했는지 확인할 예정입니다}
{또한 각 모델마다 두드러지게 나타나는 특징들을 확인할 예정입니다}
