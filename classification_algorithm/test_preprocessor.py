from data_loader import MarketDataLoader
from preprocessor import DataPreprocessor

def test_preprocessor():
    # 데이터 로드
    loader = MarketDataLoader()
    symbol = 'AAPL'
    days = 365
    
    print(f"\n{symbol} 주식 데이터 로드 중...")
    data = loader.load_market_data(symbol, days)
    
    if data is not None:
        # 전처리기 인스턴스 생성
        preprocessor = DataPreprocessor()
        
        # 특성 준비
        print("\n특성 준비 중...")
        features = preprocessor.prepare_features(data)
        
        print("\n특성 미리보기:")
        print(features.head())
        print("\n특성 정보:")
        print(features.info())
        
        # 데이터 분할
        print("\n데이터 분할 중...")
        X_train, X_test, y_train, y_test = preprocessor.split_data(features)
        
        print(f"\n학습 데이터 크기: {X_train.shape}")
        print(f"테스트 데이터 크기: {X_test.shape}")
    else:
        print("데이터를 로드할 수 없습니다.")

if __name__ == "__main__":
    test_preprocessor() 