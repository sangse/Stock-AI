# Stock-AI) 신규주섹터 분석 및 모델 활용
국내 주식시장은 항상 새로운 종목들이 상장을 하게 됩니다. 신규주들은 시장에서 항상 많은 주목을 받았고, 이를 통해 많은 거래가 활발히 이루어 지게 됩니다. 
주식시장에는 차트의 패턴을 투자에 아주 중요한 지표로 생각합니다. 시계열 분석을 통해 이러한 패턴을 파악하고 모델을 통해 분류하는 것을 목적으로 합니다.

![image](https://github.com/sangse/Stock-AI/assets/145996429/15575d20-7594-4d74-a2d3-27cafdb190ea)


## 신규 상장 시계열 데이터 특성
주가 데이터 분석에 있어 핵심적인 특성은 가격입니다. 이 가격 추세를 지지하는 거래량과 거래대금 역시 중요한 데이터로 함께 고려됩니다. 또한, 주식 시장의 동향을 파악하기 위해 시가, 고가, 저가와 같은 정보도 데이터에 포함시켰습니다.

특히, 신규 상장 주식의 경우, 상장 첫 날의 등락률은 매우 중요한 지표가 됩니다. 상장 첫 날은 등락률이 -100%에서 300%에 이르는 폭넓은 변동성을 보일 수 있으며, 이러한 변동성은 많은 투자자들이 투자 결정을 내리는 중요한 요인입니다. 실제로, 등락률이 지나치게 높은 신규 주식은 가격 조정을 받을 확률이 높다고 볼 수 있습니다. 이런 이유로, 공모가 대비 현재가(현재가/공모가)라는 특성을 데이터에 추가하여, 주식의 성과를 더욱 정밀하게 분석할 수 있도록 하였습니다.

##  신규 상장 데이터 수집
 데이터는 키움 증권에서 제공하는 API를 통해서 수집하였다. 2020년 1월부터 2024년 4월까지 상장된 종목들의 주가 데이터를 수집하였다.

# Data Extraction & Preprocessing
데이터는 상장 후 4개월간의 데이터를 수집했습니다. 이는 신규주가 상장 후 4개월 동안의 데이터 흐름을 집중적으로 분석하기 위해 진행한 것으로, 상장 후 4개월간은 거래대금이 비교적 유지되며 주가의 흐름 패턴이 존재한다고 판단했기 때문입니다.

수집한 데이터에는 현재가, 시가, 저가, 고가, 거래량, 거래대금을 포함했으며, 이를 기반으로 몇 가지 특성을 추가했습니다. 추가된 특성은 공모가 대비 가격(현재가, 시가, 저가, 고가)의 4가지 특성을 포함합니다. 상장 초기 주식의 가장 중요한 기준은 공모가이므로, 공모가는 회사의 가치를 일정 부분 반영하며 이 가격을 기준으로 주식의 상승 및 하락을 판단할 수 있습니다.

각 특성마다 가격의 범위가 다르기 때문에 이러한 차이는 모델 학습에 영향을 줄 수 있습니다. 따라서 모든 특성을 0~1 사이의 값으로 조정하기 위해 MinMaxScaler를 사용했습니다.

# Time Series Classification Model Define


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


# 주가예측을 위한 DLinear Model 
 최근 시계열 데이터 분석에서는 트랜스포머 모델을 중심으로 한 연구가 많이 진행되고 있습니다. 그러나 Dlinear 논문은 트랜스포머 모델이 장기적인 추세와 잔차 예측에서 한계를 보이며, 모델 구현에 상당한 비용이 든다고 지적합니다. 반면, Dlinear 모델은 적은 비용으로 더 우수한 성능을 제공하는 간단한 방법을 제안합니다. 이 방법은 시계열 데이터의 추세와 잔차를 각각 선형 모델로 학습한 후 이를 결합하여 더 정확한 예측 결과를 도출합니다.

특히, 추세를 정확하게 파악하기 위해서는 충분한 길이의 데이터가 필요하다는 점이 중요합니다. 데이터 길이가 짧을 경우 추세는 단순히 평균값과 크게 다르지 않게 됩니다.

현재 진행 중인 프로젝트는 단기적인 신규 상장주의 예측에 초점을 맞추고 있어, Dlinear 모델의 접근법이 적합하지 않을 수 있습니다. 그럼에도 불구하고, 현재 복잡한 모델들이 주목받는 연구 환경 속에서 Dlinear 모델이 기존 모델들의 성능을 능가한다는 점에서 이 모델을 실험해보았습니다.
<p align = "center">
 <img src="https://github.com/sangse/Stock-AI/assets/145996429/c4e22464-b468-403d-aab0-b86ad54e9e5b">
</p>

## 1) 입력데이터를 Seasonaliy와 Trend로 바꿔주는 작업
데이터의 길이는 총 24일 x feature 8 데이터로 구성했습니다.
```python
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
    	# [BATCH SIZE, SEQ_LEN, CHANNEL]
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x # [BATCH SIZE, SEQ_LEN, CHANNEL]

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
```
## 2) Model 정의 분해된 입력을 각각 1-Layer Linear Network에 통과시켜 예측 결과를 얻는 모델을 구현합니다.
```python
class Model(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Decompsition Kernel Size
        kernel_size = 3
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)

            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        # seasonal_init: [Batch, Input length, Channel]
        # trend_init: [Batch, Input length, Channel]
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        # seasonal_init: [Batch, Channel, Input length]
        # trend_init: [Batch, Channel, Input length]
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        x = x.permute(0,2,1)
        x = x[:,:,:] # 현재가만 뽑아오기

        return  x# to [Batch, Output length, Channel]
```

## 예측결과 및 결론
예측 결과를 살펴보면, 전반적인 추세에 대한 예측은 어느 정도 정확하지만, 구체적인 가격대 예측에서는 상당한 오류가 발생하는 것을 확인할 수 있습니다. 이는 Dlinear 모델이 단기간 데이터에 대한 예측력이 다소 떨어질 수 있음을 시사합니다.

그럼에도 불구하고, 이 모델이 가진 잠재력은 무시할 수 없습니다. 특히, 데이터의 장기적인 추세를 파악하는 데는 효과적일 수 있어, 반도체와 비트코인 같은 변동성이 큰 섹터에서 유용하게 사용될 수 있을 것으로 보입니다. 추후 이 두 섹터에서의 구체적인 적용 사례를 통해 모델의 성능을 다시 평가할 계획입니다. 이 과정에서 모델의 정밀도를 개선하기 위한 추가적인 조정이 이루어질 예정이며, 이는 더 정확한 가격 예측과 장기적인 투자 전략 수립에 도움을 줄 것입니다.
![image](https://github.com/sangse/Stock-AI/assets/145996429/5fa43547-58d0-44ce-92fe-358c0ba12572)




