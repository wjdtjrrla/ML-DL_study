from data_loader import MarketDataLoader
from preprocessor import DataPreprocessor
from model import VolatilityPredictor
from report_generator import ReportGenerator
import os
import pandas as pd

def test_report_generator():
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
        
        # 모든 모델 학습 및 평가
        model_types = ['dt', 'rf', 'gb', 'ensemble']
        model_results = {}
        
        for model_type in model_types:
            print(f"\n{model_type.upper()} 모델 학습 및 평가 중...")
            model = VolatilityPredictor(model_type=model_type)
            model.train(X_train, y_train)
            
            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)
            eval_results = model.evaluate(X_test, y_test)
            importance = model.get_feature_importance(features.columns)
            
            model_results[model_type] = {
                'predictions': predictions,
                'probabilities': probabilities,
                'eval_results': eval_results,
                'importance': importance
            }
        
        # 모델 비교 결과 생성
        comparison = pd.DataFrame({
            model_type: {'Accuracy': results['eval_results']['accuracy']}
            for model_type, results in model_results.items()
        }).T
        
        # 최고 성능 모델 선택
        best_model_type = comparison['Accuracy'].idxmax()
        best_results = model_results[best_model_type]
        
        # 보고서 생성
        print("\n보고서 생성 중...")
        generator = ReportGenerator(
            symbol=symbol,
            data=data,
            features=features,
            predictions=best_results['predictions'],
            probabilities=best_results['probabilities'],
            importance=best_results['importance'],
            eval_results=best_results['eval_results'],
            model_comparison=comparison,
            best_model_type=best_model_type
        )
        
        # reports 디렉토리 생성
        os.makedirs('reports', exist_ok=True)
        output_path = f'reports/{symbol}_model_comparison_report.pdf'
        
        # 보고서 생성
        report_path = generator.generate_report(output_path)
        print(f"\n보고서가 생성되었습니다: {report_path}")
        
        # 모델 비교 결과 출력
        print("\n모델 비교 결과:")
        print(comparison)
        print(f"\n최고 성능 모델: {best_model_type.upper()} (정확도: {comparison.loc[best_model_type, 'Accuracy']:.3f})")
    else:
        print("데이터를 로드할 수 없습니다.")

if __name__ == "__main__":
    test_report_generator() 