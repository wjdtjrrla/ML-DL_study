from data_loader import MarketDataLoader
from preprocessor import DataPreprocessor
from model import VolatilityPredictor
import pandas as pd

def test_models():
    # 데이터 로드
    loader = MarketDataLoader()
    symbol = 'AAPL'
    days = 365
    
    print(f"\n{symbol} 주식 데이터 로드 중...")
    data = loader.load_market_data(symbol, days)
    
    if data is not None:
        # 전처리
        preprocessor = DataPreprocessor()
        features = preprocessor.prepare_features(data)
        X_train, X_test, y_train, y_test = preprocessor.split_data(features)
        
        # 모델 유형별 테스트
        model_types = ['dt', 'rf', 'gb', 'ensemble']
        results = {}
        
        for model_type in model_types:
            print(f"\n{model_type.upper()} 모델 테스트 중...")
            
            # 모델 학습
            model = VolatilityPredictor(model_type=model_type)
            print("모델 학습 중...")
            model.train(X_train, y_train)
            
            # 예측 및 평가
            print("모델 평가 중...")
            eval_results = model.evaluate(X_test, y_test)
            results[model_type] = eval_results
            
            # 특성 중요도 계산
            importance = model.get_feature_importance(features.columns)
            
            # 결과 출력
            print(f"\n{model_type.upper()} 모델 결과:")
            print(f"정확도: {eval_results['accuracy']:.3f}")
            print("\n분류 보고서:")
            print(eval_results['classification_report'])
            
            if not importance.empty:
                print("\n상위 5개 중요 특성:")
                print(importance.head())
            
            # 앙상블 모델의 경우 개별 모델 성능도 출력
            if model_type == 'ensemble' and 'model_scores' in eval_results:
                print("\n개별 모델 성능:")
                for name, scores in eval_results['model_scores'].items():
                    print(f"\n{name.upper()} 모델:")
                    print(f"정확도: {scores['accuracy']:.3f}")
        
        # 모델 비교 결과 저장
        comparison = pd.DataFrame({
            model_type: {'Accuracy': results[model_type]['accuracy']}
            for model_type in model_types
        }).T
        
        print("\n모델 비교 결과:")
        print(comparison)
        
        # 최고 성능 모델 확인
        best_model = comparison['Accuracy'].idxmax()
        print(f"\n최고 성능 모델: {best_model.upper()} (정확도: {comparison.loc[best_model, 'Accuracy']:.3f})")
    
    else:
        print("데이터를 로드할 수 없습니다.")

if __name__ == "__main__":
    test_models() 