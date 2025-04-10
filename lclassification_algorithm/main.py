import argparse
from data_loader import MarketDataLoader
from preprocessor import DataPreprocessor
from model import VolatilityPredictor
import streamlit as st
import subprocess
import sys

def train_model(symbol: str, days: int, model_type: str = 'ensemble'):
    """모델을 학습시키고 결과를 반환합니다."""
    # 데이터 로드
    loader = MarketDataLoader()
    data = loader.load_market_data(symbol, days)
    
    if data is not None:
        # 전처리
        preprocessor = DataPreprocessor()
        features = preprocessor.prepare_features(data)
        X_train, X_test, y_train, y_test = preprocessor.split_data(features)
        
        # 모델 학습
        model = VolatilityPredictor(model_type=model_type)
        model.train(X_train, y_train)
        
        # 모델 평가
        eval_results = model.evaluate(X_test, y_test)
        
        # 특성 중요도
        importance = model.get_feature_importance(features.columns)
        
        print(f"\n{model_type.upper()} 모델 학습 결과:")
        print(f"정확도: {eval_results['accuracy']:.3f}")
        print("\n분류 보고서:")
        print(eval_results['classification_report'])
        
        if not importance.empty:
            print("\n상위 5개 중요 특성:")
            print(importance.head())
        
        # 모델 저장
        model.save_model(f'models/{symbol}_{model_type}_model.joblib')
        print(f"\n모델이 저장되었습니다: models/{symbol}_{model_type}_model.joblib")
        
        return model, eval_results
    else:
        print("데이터를 로드할 수 없습니다.")
        return None, None

def run_streamlit():
    """Streamlit 앱을 실행합니다."""
    try:
        print("Streamlit 앱을 시작합니다...")
        print("앱에 접속하려면 다음 URL을 브라우저에서 열어주세요: http://localhost:8501")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except Exception as e:
        print(f"Streamlit 앱 실행 중 오류 발생: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="주식 시장 변동성 예측 시스템")
    parser.add_argument("--mode", choices=['train', 'streamlit'], default='streamlit',
                      help="실행 모드 선택 (train: 모델 학습, streamlit: 대시보드 실행)")
    parser.add_argument("--symbol", default="AAPL",
                      help="주식 심볼 (예: AAPL)")
    parser.add_argument("--days", type=int, default=365,
                      help="분석할 기간(일)")
    parser.add_argument("--model", choices=['dt', 'rf', 'gb', 'ensemble'], default='ensemble',
                      help="사용할 모델 유형 (dt: Decision Tree, rf: Random Forest, gb: Gradient Boosting, ensemble: Voting Ensemble)")
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_model(args.symbol, args.days, args.model)
    else:
        run_streamlit()

if __name__ == "__main__":
    main() 