import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import streamlit as st

class MarketDataLoader:
    def __init__(self):
        """마켓 데이터 로더 초기화"""
        self.data_dir = "sample_data"
        
        # 샘플 데이터 디렉토리 생성
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def load_market_data(self, symbol, days=365):
        """
        주식 데이터 로드 함수
        
        Args:
            symbol (str): 주식 심볼 (예: 'AAPL')
            days (int): 로드할 데이터의 일수
            
        Returns:
            pd.DataFrame: 주식 데이터와 VIX 데이터가 포함된 데이터프레임
        """
        try:
            # 현재 날짜 기준 종료일
            end_date = datetime.now()
            # 시작일 계산
            start_date = end_date - timedelta(days=days)
            
            st.info(f"{symbol} 데이터 생성 중... (기간: {days}일)")
            
            # 데이터 생성
            data = self._generate_data(symbol, start_date, end_date)
            
            # 데이터 유효성 검사
            if data is None or data.empty:
                st.error(f"{symbol} 데이터를 생성할 수 없습니다.")
                return None
            
            # 샘플 데이터 저장 (나중에 참조용)
            data.to_csv(f"{self.data_dir}/{symbol}_data.csv")
            st.success(f"{len(data)}일 데이터가 준비되었습니다.")
            
            return data
            
        except Exception as e:
            st.error(f"데이터 로드 중 오류 발생: {str(e)}")
            return None

    def _generate_data(self, symbol, start_date, end_date):
        """
        합성 데이터 생성 함수
        
        Args:
            symbol (str): 주식 심볼
            start_date (datetime): 시작일
            end_date (datetime): 종료일
            
        Returns:
            pd.DataFrame: 생성된 데이터
        """
        try:
            # 날짜 범위 생성
            dates = pd.date_range(start=start_date, end=end_date, freq='B')  # 영업일 기준
            data = pd.DataFrame(index=dates)
            
            # 초기 가격 설정 (심볼에 따라 다른 기준가)
            base_price = self._get_base_price(symbol)
            
            # 주가 데이터 생성
            data['Open'] = np.random.normal(base_price, base_price * 0.02, len(dates))
            data['High'] = data['Open'] * (1 + np.random.uniform(0, 0.05, len(dates)))
            data['Low'] = data['Open'] * (1 - np.random.uniform(0, 0.05, len(dates)))
            data['Close'] = np.random.normal(data['Open'], base_price * 0.01, len(dates))
            data['Volume'] = np.random.randint(1000000, 10000000, len(dates))
            
            # 추세 추가
            trend = np.linspace(0, base_price * 0.2, len(dates))
            data['Close'] += trend
            
            # VIX 데이터 생성
            data['VIX'] = np.random.normal(20, 5, len(dates))
            data['VIX'] += np.sin(np.arange(len(dates)) * 0.1) * 10  # 시간에 따른 변동 추가
            data['VIX'] = data['VIX'].clip(lower=10, upper=50)  # 현실적인 범위로 조정
            
            # 최대/최소 조정 (High는 가장 높은 값, Low는 가장 낮은 값)
            data['High'] = data[['Open', 'Close', 'High']].max(axis=1)
            data['Low'] = data[['Open', 'Close', 'Low']].min(axis=1)
            
            # 수익률 계산
            data['Returns'] = data['Close'].pct_change()
            
            # 변동성 계산 (20일 이동 표준편차)
            data['Volatility'] = data['Returns'].rolling(window=20).std() 
            
            # NaN 값 제거
            data = data.dropna()
            
            return data
            
        except Exception as e:
            st.error(f"데이터 생성 중 오류 발생: {str(e)}")
            return None

    def _get_base_price(self, symbol):
        """심볼에 따른 기준 가격 반환"""
        # 일반적인 주식 가격 범위 설정
        price_ranges = {
            'AAPL': 150,
            'MSFT': 300,
            'GOOGL': 130,
            'AMZN': 140,
            'TSLA': 250,
            'META': 300,
            'NVDA': 500,
            'BRK.A': 550000,  # 극단적인 예
            'BRK.B': 350
        }
        
        # 심볼에 맞는 기준가 반환, 없으면 기본값 100
        return price_ranges.get(symbol, 100)

    def prepare_features(self, data):
        """
        예측 모델을 위한 특성 준비
        
        Args:
            data (pd.DataFrame): 원시 데이터
            
        Returns:
            pd.DataFrame: 모델 입력용 특성 데이터프레임
        """
        try:
            if data is None or data.empty:
                st.error("유효한 데이터가 없어 특성을 준비할 수 없습니다.")
                return None
                
            # 특성 생성
            features = pd.DataFrame(index=data.index)
            
            # 수익률 관련 특성
            features['Returns'] = data['Returns']
            features['Returns_MA5'] = data['Returns'].rolling(window=5).mean()
            features['Returns_MA10'] = data['Returns'].rolling(window=10).mean()
            features['Returns_Std5'] = data['Returns'].rolling(window=5).std()
            
            # 변동성 특성
            features['Volatility'] = data['Volatility']
            features['Volatility_Change'] = features['Volatility'].pct_change()
            
            # 거래량 특성
            features['Volume_Change'] = data['Volume'].pct_change()
            features['Volume_MA10'] = data['Volume'].rolling(window=10).mean()
            features['Volume_MA20'] = data['Volume'].rolling(window=20).mean()
            features['Volume_Ratio'] = data['Volume'] / features['Volume_MA10']
            
            # VIX 관련 특성
            features['VIX'] = data['VIX']
            features['VIX_Change'] = data['VIX'].pct_change()
            features['VIX_MA5'] = data['VIX'].rolling(window=5).mean()
            
            # 가격 관련 특성
            features['Price_Change'] = data['Close'].pct_change()
            features['Price_MA5'] = data['Close'].rolling(window=5).mean()
            features['Price_MA10'] = data['Close'].rolling(window=10).mean()
            features['Price_Ratio'] = data['Close'] / features['Price_MA5']
            
            # 고가-저가 스프레드
            features['HL_Spread'] = (data['High'] - data['Low']) / data['Close']
            
            # 타겟 변수 생성: 다음 날의 변동성이 증가하면 1, 아니면 0
            # 5일 후 변동성 증가 예측
            features['Target'] = (data['Volatility'].shift(-5) > data['Volatility']).astype(int)
            
            # NaN 제거
            features = features.dropna()
            
            return features
            
        except Exception as e:
            st.error(f"특성 준비 중 오류 발생: {str(e)}")
            return None

if __name__ == "__main__":
    # 테스트 코드
    loader = MarketDataLoader()
    features = loader.prepare_features(loader.load_market_data("AAPL"))
    print("Features shape:", features.shape)
    print("\nFeatures head:")
    print(features.head()) 