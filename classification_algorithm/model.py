import numpy as np
import pandas as pd
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from typing import Tuple, Dict

class VolatilityPredictor:
    """변동성 예측 모델 클래스"""
    
    def __init__(self, model_type='logistic'):
        """
        모델 초기화
        
        Args:
            model_type (str): 사용할 모델 타입
                - 'logistic': 로지스틱 회귀
                - 'dt': 의사결정 트리
                - 'rf': 랜덤 포레스트
                - 'gb': 그래디언트 부스팅
                - 'ensemble': 앙상블 (투표 기반)
        """
        self.model_type = model_type
        self.models = self._initialize_models()
        self.best_model = None
        self.feature_importance_ = None
    
    def _initialize_models(self):
        """모델 초기화 및 사전 반환"""
        models = {}
        
        # 로지스틱 회귀
        models['logistic'] = LogisticRegression(
            C=1.0, 
            max_iter=1000, 
            random_state=42
        )
        
        # 의사결정 트리
        models['dt'] = DecisionTreeClassifier(
            max_depth=5, 
            min_samples_split=10,
            random_state=42
        )
        
        # 랜덤 포레스트
        models['rf'] = RandomForestClassifier(
            n_estimators=100, 
            max_depth=5,
            min_samples_split=10,
            random_state=42
        )
        
        # 그래디언트 부스팅
        models['gb'] = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        # 앙상블 모델은 다른 모델이 학습된 후에 초기화
        
        return models
        
    def train(self, X_train, y_train):
        """
        모델 학습
        
        Args:
            X_train (numpy.ndarray): 학습 데이터
            y_train (numpy.ndarray): 학습 타깃
        """
        try:
            if X_train is None or y_train is None:
                st.error("학습할 데이터가 없습니다.")
                return
            
            if self.model_type == 'ensemble':
                # 개별 모델 먼저 학습
                for model_name in ['logistic', 'dt', 'rf', 'gb']:
                    self.models[model_name].fit(X_train, y_train)
                
                # 앙상블 모델 생성 및 학습
                self.models['ensemble'] = VotingClassifier(
                    estimators=[
                        ('logistic', self.models['logistic']),
                        ('dt', self.models['dt']),
                        ('rf', self.models['rf']),
                        ('gb', self.models['gb'])
                    ],
                    voting='soft'
                )
                self.models['ensemble'].fit(X_train, y_train)
                self.best_model = self.models['ensemble']
            else:
                # 선택한 단일 모델 학습
                self.models[self.model_type].fit(X_train, y_train)
                self.best_model = self.models[self.model_type]
            
            # 특성 중요도 계산
            self._calculate_feature_importance()
            
        except Exception as e:
            st.error(f"모델 학습 중 오류 발생: {str(e)}")
    
    def _calculate_feature_importance(self):
        """특성 중요도 계산"""
        if self.best_model is None:
            return
        
        # 모델 타입에 따라 특성 중요도 추출
        if self.model_type in ['dt', 'rf', 'gb']:
            self.feature_importance_ = self.best_model.feature_importances_
        elif self.model_type == 'logistic':
            self.feature_importance_ = np.abs(self.best_model.coef_[0])
        elif self.model_type == 'ensemble':
            # 앙상블 모델은 평균 특성 중요도 계산
            importances = []
            for name, model in self.models.items():
                if name != 'ensemble':
                    if hasattr(model, 'feature_importances_'):
                        importances.append(model.feature_importances_)
                    elif hasattr(model, 'coef_'):
                        importances.append(np.abs(model.coef_[0]))
            
            if importances:
                self.feature_importance_ = np.mean(importances, axis=0)
    
    def predict(self, X_test):
        """
        예측 수행
        
        Args:
            X_test (numpy.ndarray): 테스트 데이터
            
        Returns:
            numpy.ndarray: 예측 결과 (0 또는 1)
        """
        if self.best_model is None:
            st.error("모델이 학습되지 않았습니다.")
            return np.array([])
        
        try:
            return self.best_model.predict(X_test)
        except Exception as e:
            st.error(f"예측 중 오류 발생: {str(e)}")
            return np.array([])
    
    def predict_proba(self, X_test):
        """
        예측 확률 반환
        
        Args:
            X_test (numpy.ndarray): 테스트 데이터
            
        Returns:
            numpy.ndarray: 양성 클래스(1)의 예측 확률
        """
        if self.best_model is None:
            st.error("모델이 학습되지 않았습니다.")
            return np.array([])
        
        try:
            # predict_proba는 [확률(0), 확률(1)] 형태의 2차원 배열 반환
            probas = self.best_model.predict_proba(X_test)
            
            # 클래스 1(변동성 급증)의 확률만 반환 (1차원 배열)
            if probas.shape[1] > 1:
                return probas[:, 1]
            else:
                return probas.flatten()
                
        except Exception as e:
            st.error(f"확률 예측 중 오류 발생: {str(e)}")
            return np.array([])
    
    def evaluate(self, X_test, y_test):
        """
        모델 평가
        
        Args:
            X_test (numpy.ndarray): 테스트 데이터
            y_test (numpy.ndarray): 테스트 타깃
            
        Returns:
            dict: 평가 결과 (정확도, 분류 보고서 등)
        """
        if self.best_model is None:
            st.error("모델이 학습되지 않았습니다.")
            return {}
        
        try:
            predictions = self.predict(X_test)
            
            # 예측 결과가 비어있으면 빈 결과 반환
            if len(predictions) == 0:
                return {}
            
            # 기본 평가 지표
            results = {
                'accuracy': accuracy_score(y_test, predictions),
                'classification_report': classification_report(y_test, predictions)
            }
            
            # 혼동 행렬 추가
            results['confusion_matrix'] = confusion_matrix(y_test, predictions).tolist()
            
            # 앙상블 모델의 경우 개별 모델 성능도 평가
            if self.model_type == 'ensemble':
                model_results = {}
                for name, model in self.models.items():
                    if name != 'ensemble':
                        model_preds = model.predict(X_test)
                        model_results[name] = {
                            'accuracy': accuracy_score(y_test, model_preds),
                            'classification_report': classification_report(y_test, model_preds)
                        }
                results['model_results'] = model_results
            
            return results
            
        except Exception as e:
            st.error(f"모델 평가 중 오류 발생: {str(e)}")
            return {}
    
    def get_feature_importance(self, feature_names):
        """
        특성 중요도 반환
        
        Args:
            feature_names (list): 특성 이름 목록
            
        Returns:
            pd.DataFrame: 특성 중요도 데이터프레임
        """
        if self.feature_importance_ is None or len(feature_names) == 0:
            return pd.DataFrame()
        
        try:
            # feature_names 길이와 중요도 길이가 다른 경우 처리
            if len(feature_names) != len(self.feature_importance_):
                st.warning(f"특성 이름 개수({len(feature_names)})와 중요도 개수({len(self.feature_importance_)})가 일치하지 않습니다.")
                # 더 짧은 길이 사용
                min_len = min(len(feature_names), len(self.feature_importance_))
                feature_names = feature_names[:min_len]
                importance = self.feature_importance_[:min_len]
            else:
                importance = self.feature_importance_
            
            # 데이터프레임 생성 및 내림차순 정렬
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            })
            
            return importance_df.sort_values('importance', ascending=False)
            
        except Exception as e:
            st.error(f"특성 중요도 계산 중 오류 발생: {str(e)}")
            return pd.DataFrame()
    
    def save_model(self, filepath):
        """
        모델 저장
        
        Args:
            filepath (str): 저장 경로
        """
        if self.best_model is None:
            st.error("저장할 모델이 없습니다.")
            return
        
        try:
            joblib.dump(self, filepath)
            st.success(f"모델이 {filepath}에 저장되었습니다.")
        except Exception as e:
            st.error(f"모델 저장 중 오류 발생: {str(e)}")
    
    @classmethod
    def load_model(cls, filepath):
        """
        모델 로드
        
        Args:
            filepath (str): 로드할 모델 경로
            
        Returns:
            VolatilityPredictor: 로드된 모델 인스턴스
        """
        try:
            return joblib.load(filepath)
        except Exception as e:
            st.error(f"모델 로드 중 오류 발생: {str(e)}")
            return None

if __name__ == "__main__":
    # 테스트 코드
    predictor = VolatilityPredictor(model_type='rf')
    print(f"모델 타입: {predictor.model_type}")
    print(f"사용 가능한 모델: {list(predictor.models.keys())}") 