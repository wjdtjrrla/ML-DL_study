# GitHub 프로젝트 설정 가이드

## 1. Git 초기화 및 설정

### 1.1 Git 초기화
```bash
# 프로젝트 디렉토리에서 Git 초기화
git init

# Git 사용자 정보 설정 (처음 사용하는 경우)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### 1.2 .gitignore 파일 생성
```bash
# .gitignore 파일 생성
echo "# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Project specific
reports/
*.log
*.csv
*.pkl
*.joblib
.DS_Store" > .gitignore
```

## 2. GitHub 저장소 생성

### 2.1 GitHub에서 새 저장소 생성
1. GitHub.com에 로그인
2. 우측 상단의 '+' 버튼 클릭 → 'New repository' 선택
3. 저장소 정보 입력:
   - Repository name: `logistic_regression`
   - Description: "주식 시장 변동성 예측 시스템"
   - Visibility: Public 또는 Private 선택
   - Initialize with README: 체크 해제
4. 'Create repository' 클릭

### 2.2 로컬 저장소와 GitHub 연결
```bash
# GitHub 저장소를 원격 저장소로 추가
git remote add origin https://github.com/your-username/logistic_regression.git
```

## 3. 파일 추가 및 커밋

### 3.1 초기 파일 추가
```bash
# 모든 파일을 스테이징 영역에 추가
git add .

# 첫 번째 커밋 생성
git commit -m "Initial commit: 주식 시장 변동성 예측 시스템 기본 구조"
```

### 3.2 GitHub에 푸시
```bash
# main 브랜치로 푸시
git push -u origin main
```

## 4. 프로젝트 문서화

### 4.1 README.md 업데이트
1. 프로젝트 제목 및 설명
2. 설치 방법
3. 사용 방법
4. 주요 기능
5. 라이선스 정보

### 4.2 추가 문서 업데이트
1. `IMPLEMENTATION_GUIDE.md`
2. `PROJECT_SETUP.md`
3. `GITHUB_SETUP.md`

## 5. 브랜치 관리

### 5.1 개발 브랜치 생성
```bash
# 개발 브랜치 생성 및 전환
git checkout -b develop

# GitHub에 푸시
git push -u origin develop
```

### 5.2 기능 브랜치 생성
```bash
# 새로운 기능 개발을 위한 브랜치 생성
git checkout -b feature/new-feature develop

# 작업 완료 후 develop 브랜치로 병합
git checkout develop
git merge feature/new-feature
```

## 6. 협업 설정

### 6.1 이슈 템플릿 설정
1. `.github/ISSUE_TEMPLATE` 디렉토리 생성
2. 이슈 템플릿 파일 추가:
   - `bug_report.md`
   - `feature_request.md`

### 6.2 Pull Request 템플릿 설정
1. `.github/PULL_REQUEST_TEMPLATE.md` 파일 생성
2. PR 템플릿 내용 작성

## 7. GitHub Actions 설정 (선택사항)

### 7.1 CI/CD 파이프라인 설정
1. `.github/workflows` 디렉토리 생성
2. `python-app.yml` 파일 생성:
   - Python 버전 테스트
   - 의존성 설치
   - 테스트 실행

## 8. 보안 설정

### 8.1 환경 변수 설정
1. GitHub 저장소의 Settings → Secrets and variables → Actions
2. 필요한 시크릿 추가:
   - `ALPHA_VANTAGE_API_KEY`
   - 기타 API 키

### 8.2 보안 취약점 스캔
1. GitHub의 Security 탭 활성화
2. Dependabot 설정

## 9. 유지보수

### 9.1 정기적인 업데이트
```bash
# 최신 변경사항 가져오기
git pull origin main

# 변경사항 확인
git status

# 변경사항 커밋 및 푸시
git add .
git commit -m "Update: 변경사항 설명"
git push origin main
```

### 9.2 릴리즈 관리
1. GitHub의 Releases 섹션에서 새 릴리즈 생성
2. 버전 태그 생성
3. 릴리즈 노트 작성 