import os
import sys
import subprocess
import pkg_resources

def check_streamlit_installed():
    """Streamlit이 설치되어 있는지 확인합니다."""
    try:
        pkg_resources.get_distribution('streamlit')
        return True
    except pkg_resources.DistributionNotFound:
        return False

def install_requirements():
    """필요한 패키지를 설치합니다."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        return True
    except subprocess.CalledProcessError:
        print("패키지 설치 중 오류가 발생했습니다.")
        return False

def run_streamlit():
    """Streamlit 앱을 실행합니다."""
    try:
        # Streamlit 설치 확인
        if not check_streamlit_installed():
            print("Streamlit이 설치되어 있지 않습니다. 설치를 시도합니다...")
            if not install_requirements():
                print("Streamlit 설치에 실패했습니다.")
                print("다음 명령어로 수동 설치를 시도해보세요:")
                print("pip install -r requirements.txt")
                return
        
        print("\n주식 시장 변동성 예측 시스템을 시작합니다...")
        print("\n사용 가능한 모델:")
        print("1. 의사결정 트리 (Decision Tree)")
        print("2. 랜덤 포레스트 (Random Forest)")
        print("3. 그래디언트 부스팅 (Gradient Boosting)")
        print("4. 앙상블 (Voting Ensemble) - 기본값")
        print("\n모델은 웹 인터페이스에서 선택할 수 있습니다.")
        
        print("\n웹 인터페이스에 접속하려면 다음 URL을 브라우저에서 열어주세요:")
        print("http://localhost:8501")
        
        # Streamlit 앱 실행
        os_command = [sys.executable, "-m", "streamlit", "run", "app.py"]
        subprocess.run(os_command)
        
    except Exception as e:
        print(f"\n오류 발생: {str(e)}")
        print("\n문제 해결 방법:")
        print("1. 가상환경이 활성화되어 있는지 확인")
        print("2. 필요한 패키지가 설치되어 있는지 확인")
        print("3. Python 버전 확인 (python --version)")
        print("4. 설치된 패키지 목록 확인 (pip list)")
        
        # 상세 오류 정보 출력
        print("\n상세 오류 정보:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_streamlit() 