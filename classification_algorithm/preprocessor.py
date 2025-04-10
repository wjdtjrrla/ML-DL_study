import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple
import urllib.parse
import traceback

class DataPreprocessor:
    """데이터 전처리 클래스"""
    
    def __init__(self):
        """데이터 전처리기 초기화"""
        self.scaler = StandardScaler()
        self.test_size = 0.2
        self.random_state = 42
        
    def prepare_data(self, features):
        """
        데이터 전처리 및 학습/테스트 분할
        
        Args:
            features (pd.DataFrame): 특성이 포함된 데이터프레임
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test) - 전처리된 데이터
        """
        try:
            if features is None or features.empty:
                st.error("전처리할 데이터가 없습니다.")
                return None, None, None, None
            
            # 타겟 변수와 특성 분리
            if 'Target' in features.columns:
                X = features.drop('Target', axis=1)
                y = features['Target']
            else:
                st.error("'Target' 열이 데이터에 없습니다.")
                return None, None, None, None
            
            # 특성 스케일링
            X_scaled = self.scaler.fit_transform(X)
            
            # 학습/테스트 분할
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, 
                test_size=self.test_size, 
                random_state=self.random_state,
                shuffle=True
            )
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            st.error(f"데이터 전처리 중 오류 발생: {str(e)}")
            return None, None, None, None
    
    def prepare_data_with_indices(self, features):
        """
        데이터 전처리 및 학습/테스트 분할 (테스트 인덱스 함께 반환)
        
        Args:
            features (pd.DataFrame): 특성이 포함된 데이터프레임
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test, test_indices) - 전처리된 데이터 및 테스트 인덱스
        """
        try:
            if features is None or features.empty:
                st.error("전처리할 데이터가 없습니다.")
                return None, None, None, None, None
            
            # 타겟 변수와 특성 분리
            if 'Target' in features.columns:
                X = features.drop('Target', axis=1)
                y = features['Target']
            else:
                st.error("'Target' 열이 데이터에 없습니다.")
                return None, None, None, None, None
            
            # 특성 스케일링
            X_scaled = self.scaler.fit_transform(X)
            
            # 인덱스 추적을 위한 배열 생성
            indices = np.arange(len(features))
            
            # 학습/테스트 분할 (인덱스 함께 분할)
            X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
                X_scaled, y, indices,
                test_size=self.test_size, 
                random_state=self.random_state,
                shuffle=True
            )
            
            # 테스트 데이터의 원래 인덱스 가져오기
            test_indices = features.index[test_idx]
            
            # 배열 크기 확인 출력
            st.write("Debug - 배열 길이 확인:")
            st.write(f"X_train: {X_train.shape}, y_train: {len(y_train)}")
            st.write(f"X_test: {X_test.shape}, y_test: {len(y_test)}")
            st.write(f"test_indices: {len(test_indices)}")
            
            return X_train, X_test, y_train, y_test, test_indices
            
        except Exception as e:
            st.error(f"데이터 전처리 중 오류 발생: {str(e)}")
            return None, None, None, None, None
    
    def get_feature_names(self, features):
        """
        특성 이름 반환
        
        Args:
            features (pd.DataFrame): 특성 데이터프레임
            
        Returns:
            list: 특성 이름 리스트
        """
        if features is None or features.empty:
            return []
        
        # Target 열 제외하고 특성 이름 반환
        return [col for col in features.columns if col != 'Target']

if __name__ == "__main__":
    # 테스트 코드
    from data_loader import MarketDataLoader
    from datetime import datetime, timedelta
    
    loader = MarketDataLoader()
    features, _ = loader.prepare_features(
        "AAPL",
        (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
        datetime.now().strftime('%Y-%m-%d')
    )
    
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(features)
    
    print("Training set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)

    try:
        # 분석 시작 버튼을 클릭했을 때의 로직
        pass
    except Exception as e:
        error_msg = str(e)
        st.error(f"오류가 발생했습니다: {error_msg}")
        
        # ChatGPT 버튼 생성
        error_text = f"Streamlit 앱에서 다음 오류가 발생했습니다: {error_msg}" + "\n\n스택 트레이스:\n" + traceback.format_exc()
        chatgpt_url = f"https://chat.openai.com/chat?message={urllib.parse.quote(error_text)}"
        st.markdown(f'<a href="{chatgpt_url}" target="_blank"><button style="background-color:#4CAF50;color:white;padding:10px 20px;border:none;border-radius:4px;cursor:pointer;">ChatGPT에게 이 오류에 대해 물어보기</button></a>', unsafe_allow_html=True) 