# Stock-AI 신규주섹터 분석 및 모델 활용
국내 주식시장은 항상 새로운 종목들이 상장을 하게 됩니다. 신규주들은 시장에서 항상 많은 주목을 받았고, 이를 통해 많은 거래가 활발히 이루어 지게 됩니다. 
이런 국내 신규상장 섹터 분야의 시계열 패턴을 분석하고 유의미한 특성들을 추출하여 AI를 활용한 모델을 개발 계획했습니다.

## 신규 상장 시계열 데이터 특성
 주가 데이터를 분석 함에 있어서 기본이 되는 특성은 아무래도 가격이다. 그리고 이것들의 추세를 받쳐주는 거래량과 거래대금은 세트로 따라온다고 생각한다.
그리고 주식의 흐름에서 시장의 시가와 고가, 저가는 주식의 흐름에 영향을 주는 특성이므로 데이터에 추가하였다. 신규주에서 의미가 있는 특성을 찾아보면,
처음 상장된 날의 등락률이 중요하다고 할 수 있다. 그 이유는 처음으로 상장되는 날은 최대 -100% ~ 300% 까지의 등락을 할 수 있는 특성이 있다. 이러한 큰
등락률은 많은 투자자들이 투자를 하는 이유라고 할 수 있다. 등락률이 너무 높게 형성된 신규주식들은 높은 확률로 가격의 조정을 받을 확률이 높아진다. 
그런 이유로 공모가 대비 현재가 (현재가/공모가) 특성을 따로 추가했.

##  신규 상장 데이터 수집
 데이터는 키움 증권에서 제공하는 API를 통해서 수집하였다. 2020년 1월부터 2024년 4월까지 상장된 종목들의 주가 데이터를 수집하였다.



# Time Series Classification 과 Model  구현


## Learning Representations for Complete/Incomplete Time Series Clustering

시계열 데이터를 딥러닝 모델을 통해 클러스터링하는 방법을 소개하는 논문이다.

[1] Learning Representations for Time Series Clustering

본 논문에서는 time series clustering을 위한 representation learning 방법론인 Deep Temporal Clustering Representation(DTCR)을 제안하였다. 본 방법론은 temporal reconstruction, K-means objective, real/fake sample의 auxiliary classification을 통합하여 cluster-specific time series representation을 학습한다.

논문에서는 Encoder Decoder구조로 시계열 데이터를 재구성하는것을 목적으로 한다. 
![image](https://github.com/sangse/Stock-AI/assets/145996429/57389885-95e7-4e3f-a809-76d5290eb102)

 Encoder block의  output값을 k-means clustering 해준다. Encoder Decoder 구조를 통해 시계열 데이터를 재구성하여 Encoder에서 나오는 적은 차원의 벡터가 시계열 데이터의 정보를 최대한 많이 담을수
있게 학습하는것이 목적이다. 그 과정에서 Hidden layer와 clustering 결과값을 같이 활용해 이 데이터가 진짜인지 아닌지 분류해주는 모델도 활용한다. 그래서 모델의 Total Loss는 Reconsturction Loss와 Classification Loss의 합이 된다.

## CNN models's Encoder Decoder
 논문에서 Encoder Decoder 모델을 구성할때 LSTM, RNN, CNN 등을 활용해서 모델을 구성하는 예시를 제시했다. 여기서 모델에 사용될 데이터 구조가 Timestep x features 형태이기 때문에 이 데이터를 하나의 이미지 셋이라고 생각하고 CNN모델을 적용했다. 시계열 데이터를 CNN 모델을 활용할떄 많이 사용하는 방법 중 하나이다. 논문에서는 classification loss를 활용했지만 본 모델에서는 활용하지 않았다. 의미가 있는 방법이긴 하지만 수집한 데이터셋에서는 그렇게 많은 차이가 발생하지 않을 것이라고 판단했다. 학습 후 종목의 현재가를 재구현한 것은 다음과 같다. 재구현 된 데이터가 추세와 가격을 잘 재구현한것을 확인 할수 있다.
 
<p align="center">
  <img src="https://github.com/sangse/Stock-AI/assets/145996429/d92b2029-2a76-4437-943d-1421e5dcfc69">
</p>



## Encoder output's K-Means Clustering
### 3 dimension clustering view
CNN 기반의 Encoder Decoder 모델을 학습 하고 Encoder Block을 통해 3차원 형태의 vector를 추출한다. 여기서 3차원 형태로 추출한 이유는 데이터를 시각화 하여서 군집을 이루는 것을 눈으로 확인 하기 위함이다. 생성된 군집의 시각화 아래 그림과 같다. 군집의 개수는 10으로 설정 하였고 3차원에서의 데이터는 산개정도가 있긴하지만 군집을 이루고 있는 것을 확인 할수 있다. 


<p align="center">
  <img src="https://github.com/sangse/Stock-AI/assets/145996429/e08f2e09-ea29-4e34-aa32-ea5b933d7e82">
</p>

### clustering samples
위의 군집들이 실제 가격측면 변동성을 잘 표현하는 가에대한 평가가 필요하다. 이것은 각 군집들의 표본들을 통해 확인할수 있다. 각 군집들의 5개의 표본을 시각화 한것은 다음 그림과 같다.


