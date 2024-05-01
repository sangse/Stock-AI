# Stock-AI 신규주섹터 분석 및 모델 활용
국내 주식시장은 항상 새로운 종목들이 상장을 하게 됩니다. 신규주들은 시장에서 항상 많은 주목을 받았고, 이를 통해 많은 거래가 활발히 이루어 지게 됩니다. 
주식시장에는 차트의 패턴을 투자에 아주 중요한 지표로 생각합니다. 시계열 분석을 통해 이러한 패턴을 파악하고 모델을 통해 분류하는 것을 목적으로 합니다.
## 신규 상장 시계열 데이터 특성
주가 데이터 분석에 있어 핵심적인 특성은 가격입니다. 이 가격 추세를 지지하는 거래량과 거래대금 역시 중요한 데이터로 함께 고려됩니다. 또한, 주식 시장의 동향을 파악하기 위해 시가, 고가, 저가와 같은 정보도 데이터에 포함시켰습니다.

특히, 신규 상장 주식의 경우, 상장 첫 날의 등락률은 매우 중요한 지표가 됩니다. 상장 첫 날은 등락률이 -100%에서 300%에 이르는 폭넓은 변동성을 보일 수 있으며, 이러한 변동성은 많은 투자자들이 투자 결정을 내리는 중요한 요인입니다. 실제로, 등락률이 지나치게 높은 신규 주식은 가격 조정을 받을 확률이 높다고 볼 수 있습니다. 이런 이유로, 공모가 대비 현재가(현재가/공모가)라는 특성을 데이터에 추가하여, 주식의 성과를 더욱 정밀하게 분석할 수 있도록 하였습니다.

##  신규 상장 데이터 수집
 데이터는 키움 증권에서 제공하는 API를 통해서 수집하였다. 2020년 1월부터 2024년 4월까지 상장된 종목들의 주가 데이터를 수집하였다.



# Time Series Classification 과 Model  구현


## Learning Representations for Complete/Incomplete Time Series Clustering

시계열 데이터를 딥러닝 모델을 통해 군집화하는 방법을 소개하는 논문입니다.

[1] Learning Representations for Time Series Clustering

본 논문에서는 time series clustering을 위한 representation learning 방법론인 Deep Temporal Clustering Representation(DTCR)을 제안하였다. 본 방법론은 temporal reconstruction, K-means objective, real/fake sample의 auxiliary classification을 통합하여 cluster-specific time series representation을 학습한다.

논문에서는 Encoder Decoder구조로 시계열 데이터를 재구성하는것을 목적으로 한다. 
![image](https://github.com/sangse/Stock-AI/assets/145996429/57389885-95e7-4e3f-a809-76d5290eb102)


논문의 목적은 시계열 데이터를 Encoder-Decoder 구조를 통해 재구성하는 것입니다. 이 과정에서 Encoder block은 데이터에서 얻은 정보를 최대한 보존하면서 차원을 축소한 벡터를 출력합니다. 이 출력된 벡터들은 k-means 클러스터링을 통해 분류되어, 각 시계열 데이터가 어떤 특성을 가지고 있는지 군집화합니다.

더불어, 모델은 재구성된 데이터의 정확성을 평가하기 위해 Hidden Layer와 클러스터링 결과를 활용하여 데이터가 진짜인지를 분류하는 추가적인 작업도 수행합니다. 이로 인해 모델의 전체 손실(Total Loss)은 재구성 손실(Reconstruction Loss)과 분류 손실(Classification Loss)의 합으로 계산됩니다. 이러한 접근은 모델이 시계열 데이터를 효과적으로 학습하고 재구성하는 능력을 향상시키는 데 기여합니다.

## CNN models's Encoder Decoder
논문에서는 시계열 데이터의 구조를 Timestep x Features 형태로 활용하여 LSTM, RNN, CNN 등 다양한 모델 구조를 설계하는 방법을 제시했습니다. 이러한 데이터 구조를 하나의 이미지 세트로 간주하고 CNN 모델을 적용하는 것은 시계열 데이터를 분석할 때 흔히 사용되는 접근 방식 중 하나입니다. 논문에서는 일반적으로 classification loss를 활용하지만, 본 연구에서는 이를 사용하지 않았습니다. 이유는 수집한 데이터셋에서 classification loss가 크게 차이를 만들어내지 않을 것으로 예상되었기 때문입니다. 모델 학습 후, 재구현된 종목의 현재가 데이터는 추세와 가격을 잘 재현하는 것으로 나타났습니다. 이 결과는 모델이 시계열 데이터의 특성을 효과적으로 캡처하고 재현할 수 있음을 보여줍니다.
 
<p align="center">
  <img src="https://github.com/sangse/Stock-AI/assets/145996429/d92b2029-2a76-4437-943d-1421e5dcfc69">
</p>



## Encoder output's K-Means Clustering
### 1) 3 dimension clustering view
CNN 기반의 Encoder-Decoder 모델을 학습시킨 후, Encoder Block을 통해 데이터를 3차원 벡터로 변환했습니다. 이러한 3차원 형태로 데이터를 변환한 주된 목적은 시각화를 통해 데이터 간의 군집 형성을 직관적으로 확인하기 위함입니다. 생성된 군집은 총 10개로 설정되었으며, 3차원 공간에서의 데이터 포인트들은 다소 흩어져 있지만, 명확하게 군집을 형성하고 있음을 아래 그림에서 확인할 수 있습니다.

<p align="center">
  <img src="https://github.com/sangse/Stock-AI/assets/145996429/e08f2e09-ea29-4e34-aa32-ea5b933d7e82">
</p>

### 2) clustering samples
군집이 실제 가격 변동성을 얼마나 잘 반영하는지 평가하기 위해 각 군집의 5개 샘플을 시각화한 결과를 분석할 수 있습니다. 이러한 분석을 통해 각 군집이 변동 패턴, 상승 추세, 하락 추세 등 특정 특성을 가지고 있음을 확인할 수 있습니다. 또한, 이는 단순한 추세뿐만 아니라 가격 측면에서도 군집 형성의 근거가 될 수 있음을 보여줍니다. 일부 데이터는 샘플과 다르게 표현될 수 있으며, 이는 군집 수가 부족하여 발생하는 문제일 가능성이 있습니다. 앞으로의 연구에서는 시계열 예측 분야에서 최신 성과를 이룬 DLinear 모델을 이용해 군집 데이터에 적용하여 주가 예측을 시도해 보겠습니다.

<p align = "center">
<img src="https://github.com/sangse/Stock-AI/assets/145996429/38e69c31-e945-4790-b955-f08b5b0d22ca">
</p>


# 주가예측을 위한 DLinear 모델 
 최근 시계열 데이터 분석에서는 트랜스포머 모델을 중심으로 한 연구가 많이 진행되고 있습니다. 그러나 Dlinear 논문은 트랜스포머 모델이 장기적인 추세와 잔차 예측에서 한계를 보이며, 모델 구현에 상당한 비용이 든다고 지적합니다. 반면, Dlinear 모델은 적은 비용으로 더 우수한 성능을 제공하는 간단한 방법을 제안합니다. 이 방법은 시계열 데이터의 추세와 잔차를 각각 선형 모델로 학습한 후 이를 결합하여 더 정확한 예측 결과를 도출합니다.

특히, 추세를 정확하게 파악하기 위해서는 충분한 길이의 데이터가 필요하다는 점이 중요합니다. 데이터 길이가 짧을 경우 추세는 단순히 평균값과 크게 다르지 않게 됩니다.

현재 진행 중인 프로젝트는 단기적인 신규 상장주의 예측에 초점을 맞추고 있어, Dlinear 모델의 접근법이 적합하지 않을 수 있습니다. 그럼에도 불구하고, 현재 복잡한 모델들이 주목받는 연구 환경 속에서 Dlinear 모델이 기존 모델들의 성능을 능가한다는 점에서 이 모델을 실험해보았습니다.
![image](https://github.com/sangse/Stock-AI/assets/145996429/c4e22464-b468-403d-aab0-b86ad54e9e5b)

## 











