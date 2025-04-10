import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import os

from data_loader import MarketDataLoader
from preprocessor import DataPreprocessor
from model import VolatilityPredictor

# 페이지 설정
st.set_page_config(
    page_title="주식 시장 변동성 예측 시스템",
    page_icon="📈",
    layout="wide"
)

# CSS 스타일 추가
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .footer {
        text-align: center;
        color: #757575;
        font-size: 0.8rem;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# 제목
st.markdown('<h1 class="main-header">주식 시장 변동성 예측 시스템 🚀</h1>', unsafe_allow_html=True)

# 사이드바 설정
st.sidebar.title("분석 설정")
symbol = st.sidebar.text_input("주식 심볼", value="AAPL")
days = st.sidebar.number_input("분석 기간(일)", min_value=30, max_value=3650, value=365)
model_type = st.sidebar.selectbox(
    "모델 선택",
    options=['logistic', 'dt', 'rf', 'gb', 'ensemble'],
    format_func=lambda x: {
        'logistic': '로지스틱 회귀',
        'dt': '의사결정 트리',
        'rf': '랜덤 포레스트',
        'gb': '그래디언트 부스팅',
        'ensemble': '앙상블 (Voting)'
    }[x]
)

# 분석 버튼
analyze_button = st.sidebar.button("분석 시작")

# 사이드바에 정보 추가
st.sidebar.markdown("---")
st.sidebar.markdown("### 정보")
st.sidebar.info(
    "이 대시보드는 머신러닝 모델을 사용하여 주식 시장의 변동성 급증을 예측합니다. "
    "다양한 특성과 지표를 활용하여 미래 변동성 패턴을 예측합니다."
)

# 메인 콘텐츠
if analyze_button:
    with st.spinner("데이터를 로드하고 분석 중입니다..."):
        # 진행 상태 표시
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        
        try:
            # 1. 데이터 로드
            loader = MarketDataLoader()
            data = loader.load_market_data(symbol, days)
            
            if data is not None:
                # 2. 특성 준비
                features = loader.prepare_features(data)
                
                if features is not None:
                    # 3. 데이터 전처리
                    preprocessor = DataPreprocessor()
                    X_train, X_test, y_train, y_test, test_indices = preprocessor.prepare_data_with_indices(features)
                    
                    if X_train is not None and X_test is not None:
                        # 4. 모델 학습
                        with st.spinner(f"{model_type} 모델 학습 중..."):
                            model = VolatilityPredictor(model_type=model_type)
                            model.train(X_train, y_train)
                            
                            # 5. 예측 및 평가
                            predictions = model.predict(X_test)
                            probabilities = model.predict_proba(X_test)
                            eval_results = model.evaluate(X_test, y_test)
                            
                            # 특성 중요도
                            feature_names = preprocessor.get_feature_names(features)
                            importance = model.get_feature_importance(feature_names)
                            
                            # 6. 결과 데이터프레임 생성
                            if len(test_indices) > 0 and len(predictions) > 0 and len(probabilities) > 0:
                                # 최소 길이 계산
                                min_length = min(len(test_indices), len(predictions), len(probabilities))
                                
                                # 안전한 방식으로 결과 데이터 생성
                                result_data = []
                                for i in range(min_length):
                                    # 각 행 데이터 생성
                                    row = {
                                        'Date': test_indices[i],
                                        'Prediction': int(predictions[i]) if i < len(predictions) else None,
                                        'Probability': float(probabilities[i]) if i < len(probabilities) else None
                                    }
                                    
                                    # 실제 변동성 값 추가
                                    if test_indices[i] in data.index:
                                        row['Actual_Volatility'] = float(data.loc[test_indices[i], 'Volatility'])
                                    
                                    # 행 추가
                                    result_data.append(row)
                                
                                # 데이터프레임 생성
                                results_df = pd.DataFrame(result_data)
                                
                                # 7. 결과 표시
                                st.markdown('<h2 class="sub-header">분석 결과</h2>', unsafe_allow_html=True)
                                
                                # 7.1 주요 지표 표시
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                                    accuracy = eval_results.get('accuracy', 0)
                                    st.metric("정확도", f"{accuracy:.3f}")
                                    st.markdown('</div>', unsafe_allow_html=True)
                                    
                                with col2:
                                    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                                    st.metric("데이터 기간", f"{days}일")
                                    st.markdown('</div>', unsafe_allow_html=True)
                                    
                                with col3:
                                    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                                    st.metric("데이터 포인트", f"{len(features)}")
                                    st.markdown('</div>', unsafe_allow_html=True)
                                
                                # 7.2 차트 표시
                                st.markdown('<h2 class="sub-header">변동성 분석</h2>', unsafe_allow_html=True)
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # 변동성 차트
                                    st.markdown('<h3>주가 변동성 추이</h3>', unsafe_allow_html=True)
                                    fig_volatility = go.Figure()
                                    fig_volatility.add_trace(go.Scatter(
                                        x=data.index,
                                        y=data['Volatility'],
                                        name='변동성',
                                        line=dict(color='blue')
                                    ))
                                    fig_volatility.update_layout(
                                        title='주가 변동성 추이',
                                        xaxis_title='날짜',
                                        yaxis_title='변동성',
                                        height=400,
                                        template='plotly_white'
                                    )
                                    st.plotly_chart(fig_volatility, use_container_width=True)
                                    
                                    # 주가 데이터 표시
                                    st.markdown('<h3>주가 데이터</h3>', unsafe_allow_html=True)
                                    st.dataframe(data[['Open', 'High', 'Low', 'Close', 'Volume']].head(10), use_container_width=True)
                                
                                with col2:
                                    # 변동성 예측 차트
                                    st.markdown('<h3>변동성 급증 예측 확률</h3>', unsafe_allow_html=True)
                                    fig_prediction = go.Figure()
                                    fig_prediction.add_trace(go.Scatter(
                                        x=results_df['Date'],
                                        y=results_df['Probability'],
                                        name='변동성 급증 확률',
                                        line=dict(color='red')
                                    ))
                                    fig_prediction.update_layout(
                                        title='변동성 급증 예측 확률',
                                        xaxis_title='날짜',
                                        yaxis_title='확률',
                                        height=400,
                                        template='plotly_white'
                                    )
                                    st.plotly_chart(fig_prediction, use_container_width=True)
                                    
                                    # 특징 데이터 표시
                                    st.markdown('<h3>특징 데이터</h3>', unsafe_allow_html=True)
                                    st.dataframe(features.head(10), use_container_width=True)
                                
                                # 7.3 특성 중요도 및 모델 성능
                                st.markdown('<h2 class="sub-header">모델 분석</h2>', unsafe_allow_html=True)
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # 특성 중요도 차트
                                    st.markdown('<h3>특성 중요도</h3>', unsafe_allow_html=True)
                                    if not importance.empty:
                                        fig_importance = px.bar(
                                            importance.head(10),
                                            x='importance',
                                            y='feature',
                                            orientation='h',
                                            title="상위 10개 중요 특성"
                                        )
                                        fig_importance.update_layout(
                                            height=400,
                                            template='plotly_white'
                                        )
                                        st.plotly_chart(fig_importance, use_container_width=True)
                                    else:
                                        st.warning("특성 중요도를 계산할 수 없습니다.")
                                
                                with col2:
                                    # 모델 성능 지표
                                    st.markdown('<h3>모델 성능 지표</h3>', unsafe_allow_html=True)
                                    st.markdown('<div class="info-box">', unsafe_allow_html=True)
                                    if 'classification_report' in eval_results:
                                        st.markdown("**분류 보고서:**")
                                        st.text(eval_results['classification_report'])
                                    else:
                                        st.warning("분류 보고서를 생성할 수 없습니다.")
                                    st.markdown('</div>', unsafe_allow_html=True)
                                
                                # 7.4 데이터 다운로드
                                st.markdown('<h2 class="sub-header">데이터 다운로드</h2>', unsafe_allow_html=True)
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # 특징 데이터 다운로드
                                    features_csv = features.to_csv(index=True).encode('utf-8')
                                    st.download_button(
                                        label="특징 데이터 다운로드 (CSV)",
                                        data=features_csv,
                                        file_name=f"{symbol}_features.csv",
                                        mime="text/csv"
                                    )
                                
                                with col2:
                                    # 예측 결과 다운로드
                                    results_csv = results_df.to_csv(index=False).encode('utf-8')
                                    st.download_button(
                                        label="예측 결과 다운로드 (CSV)",
                                        data=results_csv,
                                        file_name=f"{symbol}_predictions.csv",
                                        mime="text/csv"
                                    )
                                
                                # 모델 저장
                                model_path = f'model_{symbol}.joblib'
                                model.save_model(model_path)
                                
                            else:
                                st.error("예측 결과를 생성할 수 없습니다.")
                        
                    else:
                        st.error("데이터 전처리에 실패했습니다.")
                else:
                    st.error("특성 데이터를 생성할 수 없습니다.")
            else:
                st.error("데이터를 로드할 수 없습니다.")
        
        except Exception as e:
            st.error(f"분석 중 오류가 발생했습니다: {str(e)}")
            st.info("다른 주식 심볼이나 기간을 선택해보세요.")

else:
    # 초기 화면
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("""
    ## 환영합니다!
    
    이 대시보드는 주식 시장의 변동성 급증을 예측하는 머신러닝 모델을 시각화합니다.
    
    ### 사용 방법:
    1. 왼쪽 사이드바에서 주식 심볼과 분석 기간을 설정하세요.
    2. 사용할 모델을 선택하세요 (로지스틱 회귀, 의사결정 트리, 랜덤 포레스트, 그래디언트 부스팅, 앙상블).
    3. '분석 시작' 버튼을 클릭하여 분석을 시작하세요.
    
    ### 주요 기능:
    - 변동성 추이 분석 및 시각화
    - 변동성 급증 예측 모델 구축
    - 예측 결과 시각화
    - 특성 중요도 분석
    - 모델 성능 평가
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 예시 차트
    st.markdown('<h2 class="sub-header">예시 차트</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 변동성 예시 차트
        st.markdown('<h3>주가 변동성 추이 (예시)</h3>', unsafe_allow_html=True)
        dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='B')
        volatility = np.random.normal(0.02, 0.005, len(dates))
        volatility = volatility + np.sin(np.arange(len(dates)) * 0.1) * 0.01
        
        fig_example = go.Figure()
        fig_example.add_trace(go.Scatter(
            x=dates,
            y=volatility,
            name='변동성',
            line=dict(color='blue')
        ))
        fig_example.update_layout(
            title='주가 변동성 추이 (예시)',
            xaxis_title='날짜',
            yaxis_title='변동성',
            height=400,
            template='plotly_white'
        )
        st.plotly_chart(fig_example, use_container_width=True)
    
    with col2:
        # 예측 확률 예시 차트
        st.markdown('<h3>변동성 급증 예측 확률 (예시)</h3>', unsafe_allow_html=True)
        proba_example = np.random.uniform(0, 1, len(dates))
        proba_example = proba_example + np.sin(np.arange(len(dates)) * 0.05) * 0.3
        proba_example = np.clip(proba_example, 0, 1)
        
        fig_proba_example = go.Figure()
        fig_proba_example.add_trace(go.Scatter(
            x=dates,
            y=proba_example,
            name='변동성 급증 확률',
            line=dict(color='red')
        ))
        fig_proba_example.update_layout(
            title='변동성 급증 예측 확률 (예시)',
            xaxis_title='날짜',
            yaxis_title='확률',
            height=400,
            template='plotly_white'
        )
        st.plotly_chart(fig_proba_example, use_container_width=True)

# 푸터
st.markdown('<div class="footer">© 2023 주식 시장 변동성 예측 시스템</div>', unsafe_allow_html=True) 