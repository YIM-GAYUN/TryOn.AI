# Virtual Try-On Web Application

cv_final.ipynb 파일을 기반으로 제작된 가상 피팅 웹 애플리케이션입니다. </br>
전체적인 구조는 demo.ipynb와 데모 웹페이지 부분인 나머지 폴더 및 파일들로 구성되어 있습니다. </br>
즉, colab 환경에서 실행 가능한 최종 코드는 demo.ipynb입니다.

## 기능
1. 전신 사진 업로드
2. 상의/하의/원피스 옷 선택 (원피스 선택시 다른 옷 선택 불가)
3. Pose estimation을 통한 관절점 감지
4. 옷 이미지 워핑을 통한 가상 피팅

## 기술 스택
- Computer Vision: Pose Detection + Image Warping
- Backend: FastAPI + OpenCV + MediaPipe
- Frontend: React
- 데이터의 경우 kaggle의 clothes dataset을 활용하였습니다: https://www.kaggle.com/datasets/ryanbadai/clothes-dataset/data

## 실행 방법

### 자동 실행 (Windows)
```bash
start_app.bat
```

### 수동 실행

#### Backend 실행
```bash
cd backend
pip install -r requirements.txt
python main.py
```

#### Frontend 실행
```bash
cd frontend
npm install
npm start
```

## 접속 주소
- Backend API: http://localhost:8000
- Frontend: http://localhost:3000

## API 엔드포인트
- `GET /`: 서버 상태 확인
- `GET /clothing-items`: 사용 가능한 옷 목록 조회
- `POST /detect-pose`: 업로드된 이미지에서 포즈 감지
- `POST /try-on`: 가상 피팅 수행

## 프로젝트 구조
```
cv/
├── backend/
│   ├── main.py              # FastAPI 메인 애플리케이션
│   ├── pose_detection.py    # 포즈 감지 및 옷 워핑 모듈
│   ├── clothes_data.json    # 의류 데이터 및 앵커 포인트
│   ├── requirements.txt     # Python 의존성
│   └── static/
│       └── clothes/         # 카테고리별 옷 이미지 파일들
│           ├── top/         # 상의 (블레이저, 후디, 재킷, 셔츠 등)
│           │   ├── Blazer/
│           │   ├── Hoodie/
│           │   ├── Jaket/
│           │   ├── Jaket_Denim/
│           │   ├── Kaos/
│           │   ├── Kemeja/
│           │   ├── Polo/
│           │   └── Sweter/
│           ├── bottom/      # 하의 (바지, 치마 등)
│           │   ├── Celena_Panjang/  # 긴 바지
│           │   ├── Celena_Pendek/   # 짧은 바지
│           │   ├── Jeans/           # 청바지
│           │   └── Rok/             # 치마
│           └── dress/       # 원피스 및 코트
│               ├── Gaun/            # 원피스
│               └── Mantel/          # 코트/외투
├── frontend/
│   ├── src/
│   │   ├── App.js           # 메인 React 컴포넌트
│   │   ├── components/
│   │   │   ├── ImageUpload.js      # 이미지 업로드 컴포넌트
│   │   │   ├── ClothingSelection.js # 옷 선택 컴포넌트 (카테고리 탭 기능)
│   │   │   └── ResultDisplay.js    # 결과 표시 컴포넌트
│   │   └── index.css        # 스타일시트
│   └── package.json         # React 의존성
├── cv_final.ipynb          # 원본 Jupyter 노트북
└── start_app.bat           # 자동 실행 스크립트
```

## 의류 카테고리 구조
### 상의 (Top)
- 블레이저 (Blazer)
- 후디 (Hoodie)
- 재킷 (Jaket)
- 데님 재킷 (Jaket_Denim)
- 티셔츠 (Kaos)
- 셔츠 (Kemeja)
- 폴로셔츠 (Polo)
- 스웨터 (Sweter)

### 하의 (Bottom)
- 긴 바지 (Celena_Panjang)
- 짧은 바지 (Celena_Pendek)
- 청바지 (Jeans)
- 치마 (Rok)

### 원피스/외투 (Dress)
- 원피스 (Gaun)
- 코트/외투 (Mantel)

## 주의사항
- 전신이 잘 보이는 사진을 업로드해주세요
- 포즈 감지가 실패할 경우 다른 각도의 사진을 시도해보세요
- 의류는 카테고리별로 구분되어 있으며, 각 카테고리 탭에서 선택할 수 있습니다
- 원피스 선택 시 상의/하의는 자동으로 선택 해제됩니다
- 새로운 의류 추가 시 해당 카테고리 폴더에 이미지를 추가하고 clothes_data.json 파일을 업데이트하세요
