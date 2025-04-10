import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class MarketDataLoader:
    def __init__(self):
        self.data_dir = "sample_data"
        
        # 데이터 폴더가 없으면 생성
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def load_market_data(self, symbol, days=365):
        """지정한 종목과 VIX의 시장 데이터를 생성"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        try:
            # 주식 데이터 생성
            print(f"{symbol} 데이터 생성 중...")
            stock_data = self._generate_stock_data(symbol, start_date, end_date)
            
            # VIX 데이터 생성
            print("VIX 데이터 생성 중...")
            vix_data = self._generate_vix_data(start_date, end_date)
            
            # 종가, 거래량, VIX 종가로 데이터프레임 구성
            data = pd.DataFrame()
            data['Close'] = stock_data['Close']
            data['Volume'] = stock_data['Volume']
            data['VIX'] = vix_data['Close']
            
            # 수익률 계산
            data['Returns'] = data['Close'].pct_change()
            
            # 20일간 수익률의 표준편차(변동성) 계산
            data['Volatility'] = data['Returns'].rolling(window=20).std()
            
            # 결측치 제거
            data = data.dropna()
            
            if data.empty:
                raise ValueError("유효한 데이터가 없습니다.")
            
            print(f"{len(data)}일치 데이터 생성 완료")
            return data
            
        except Exception as e:
            print(f"시장 데이터 생성 오류: {str(e)}")
            raise

    def _generate_stock_data(self, symbol, start_date, end_date):
        """가상의 주식 데이터를 생성"""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        data = pd.DataFrame(index=dates)
        
        # 종목별 기준 가격 설정
        if symbol == "AAPL":
            base_value = 150
        elif symbol == "MSFT":
            base_value = 250
        elif symbol == "GOOGL":
            base_value = 2000
        elif symbol == "AMZN":
            base_value = 3000
        else:
            base_value = 100
        
        # 가격 생성 (정규분포 + 일정한 노이즈)
        data['Open'] = np.random.normal(base_value, base_value * 0.05, len(dates))
        data['High'] = data['Open'] + np.random.uniform(0, base_value * 0.02, len(dates))
        data['Low'] = data['Open'] - np.random.uniform(0, base_value * 0.02, len(dates))
        data['Close'] = np.random.normal(data['Open'], base_value * 0.01, len(dates))
        data['Volume'] = np.random.randint(1000000, 10000000, len(dates))
        
        # 추세(trend)와 계절성(seasonality) 추가
        trend = np.linspace(0, base_value * 0.2, len(dates))
        seasonality = base_value * 0.05 * np.sin(np.arange(len(dates)) * 0.1)
        data['Close'] = data['Close'] + trend + seasonality
        
        # 고가/저가 정리
        data['High'] = data[['Open', 'Close', 'High']].max(axis=1)
        data['Low'] = data[['Open', 'Close', 'Low']].min(axis=1)
        
        return data

    def _generate_vix_data(self, start_date, end_date):
        """가상의 VIX 데이터를 생성"""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        data = pd.DataFrame(index=dates)
        
        base_value = 20  # VIX 기준값
        
        data['Open'] = np.random.normal(base_value, 5, len(dates))
        data['High'] = data['Open'] + np.random.uniform(0, 3, len(dates))
        data['Low'] = data['Open'] - np.random.uniform(0, 3, len(dates))
        data['Close'] = np.random.normal(data['Open'], 1, len(dates))
        data['Volume'] = np.random.randint(100000, 1000000, len(dates))
        
        # 추세 및 계절성 반영
        trend = np.linspace(0, 10, len(dates))
        seasonality = 5 * np.sin(np.arange(len(dates)) * 0.05)
        data['Close'] = data['Close'] + trend + seasonality
        
        # 고가/저가 정리
        data['High'] = data[['Open', 'Close', 'High']].max(axis=1)
        data['Low'] = data[['Open', 'Close', 'Low']].min(axis=1)
        
        return data

    def prepare_features(self, data):
        """모델 학습을 위한 피처 생성"""
        features = pd.DataFrame()
        
        # 가격 기반 피처
        features['Returns'] = data['Returns']
        features['Volatility'] = data['Volatility']
        
        # 거래량 기반 피처
        features['Volume_Change'] = data['Volume'].pct_change()
        features['Volume_MA'] = data['Volume'].rolling(window=20).mean()
        
        # VIX 기반 피처
        features['VIX_Change'] = data['VIX'].pct_change()
        features['VIX_MA'] = data['VIX'].rolling(window=20).mean()
        
        # 타겟 생성: 다음 날 변동성이 커졌는가? → 1이면 증가, 0이면 감소 또는 유지
        features['target'] = (data['Volatility'].shift(-1) > data['Volatility']).astype(int)
        
        # 결측치 제거
        features = features.dropna()
        
        if features.empty:
            raise ValueError("유효한 피처 데이터가 없습니다.")
        
        return features

if __name__ == "__main__":
    # 테스트 코드 실행
    loader = MarketDataLoader()
    features = loader.prepare_features(loader.load_market_data("AAPL"))
    # features.to_csv('./sample_data/AAPL.csv')
    print("피처 데이터 형태:", features.shape)
    print("\n피처 일부 미리보기:")
    print(features.head())