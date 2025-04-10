from data_loader import MarketDataLoader

def test_data_loader():
    # 데이터 로더 인스턴스 생성
    loader = MarketDataLoader()
    
    # AAPL 주식 데이터 로드
    symbol = 'AAPL'
    days = 365
    
    print(f"\n{symbol} 주식 데이터 로드 중...")
    data = loader.load_market_data(symbol, days)
    
    if data is not None:
        print("\n데이터 미리보기:")
        print(data.head())
        print("\n데이터 정보:")
        print(data.info())
    else:
        print("데이터를 로드할 수 없습니다.")

if __name__ == "__main__":
    test_data_loader() 