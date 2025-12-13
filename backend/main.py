from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import cv2
import numpy as np
import base64
import io
from PIL import Image
from typing import List, Dict, Optional
import os
import json

from pose_detection import PoseDetector, ClothingWarper

app = FastAPI(title="Virtual Try-On API", version="1.0.0")

# 정적 파일 서빙
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React 개발 서버
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 인스턴스
pose_detector = PoseDetector()
clothing_warper = ClothingWarper()

# 옷 데이터 로드
def load_clothing_data():
    try:
        data_path = os.path.join(os.path.dirname(__file__), "clothes_data.json")
        with open(data_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        # 파일이 없으면 기본 템플릿 사용
        return DEFAULT_TEMPLATES

# 옷 이미지 로드
def load_clothing_image(cloth_id):
    """옷 ID에 해당하는 이미지를 로드합니다. 없으면 기본 이미지를 생성합니다."""
    try:
        if cloth_id in CLOTHING_DATA:
            image_path = CLOTHING_DATA[cloth_id].get("image_path")
            if image_path:
                # 상대 경로를 절대 경로로 변환
                if not os.path.isabs(image_path):
                    image_path = os.path.join(os.path.dirname(__file__), image_path)
                
                if os.path.exists(image_path):
                    cloth_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                    if cloth_img is not None:
                        return cloth_img
    except Exception as e:
        print(f"옷 이미지 로드 실패 ({cloth_id}): {e}")
    
    # 기본 이미지 생성
    category = CLOTHING_DATA.get(cloth_id, {}).get("category", "top")
    dummy_cloth_image = np.zeros((600, 400, 3), dtype=np.uint8)
    
    # 카테고리별 기본 색상
    if category == "top":
        dummy_cloth_image[:, :] = [200, 100, 100]  # 빨간색 계열
    elif category == "bottom":
        dummy_cloth_image[:, :] = [100, 100, 200]  # 파란색 계열
    elif category == "dress":
        dummy_cloth_image[:, :] = [150, 100, 200]  # 보라색 계열
    
    return dummy_cloth_image

# 기본 옷 템플릿 (fallback용)
DEFAULT_TEMPLATES = {
    "hoodie_1": {
        "name": "Basic Hoodie",
        "category": "top",
        "anchors": {
            "left_shoulder": [100, 150],
            "right_shoulder": [300, 150],
            "left_wrist": [50, 400],
            "right_wrist": [350, 400],
            "left_hip": [120, 500],
            "right_hip": [280, 500]
        }
    },
    "jeans_1": {
        "name": "Blue Jeans",
        "category": "bottom",
        "anchors": {
            "left_hip": [100, 50],
            "right_hip": [300, 50],
            "left_knee": [120, 250],
            "right_knee": [280, 250],
            "left_ankle": [130, 450],
            "right_ankle": [270, 450]
        }
    },
    "dress_1": {
        "name": "Summer Dress",
        "category": "dress",
        "anchors": {
            "left_shoulder": [100, 100],
            "right_shoulder": [300, 100],
            "left_hip": [120, 300],
            "right_hip": [280, 300],
            "left_knee": [125, 450],
            "right_knee": [275, 450]
        }
    }
}

# 옷 데이터 로드 및 설정
CLOTHING_DATA = load_clothing_data()
clothing_warper.set_cloth_templates(CLOTHING_DATA)


class ClothingItem(BaseModel):
    id: str
    name: str
    category: str  # "top", "bottom", "dress"
    image_url: str


class TryOnRequest(BaseModel):
    selected_clothes: List[str]  # 선택된 옷의 ID 리스트


def decode_base64_image(base64_string: str) -> np.ndarray:
    """Base64 문자열을 OpenCV 이미지로 변환"""
    try:
        # data:image/jpeg;base64, 부분 제거
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
        
        # Base64 디코딩
        image_data = base64.b64decode(base64_string)
        
        # PIL Image로 변환
        image = Image.open(io.BytesIO(image_data))
        
        # RGB -> BGR (OpenCV 형식)
        image = image.convert('RGB')
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        return opencv_image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 디코딩 오류: {str(e)}")


def encode_image_to_base64(image: np.ndarray) -> str:
    """OpenCV 이미지를 Base64 문자열로 변환"""
    try:
        # BGR -> RGB 변환
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # PIL Image로 변환
        pil_image = Image.fromarray(image_rgb)
        
        # BytesIO 버퍼에 저장
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=95)
        
        # Base64 인코딩
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return f"data:image/jpeg;base64,{image_base64}"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 인코딩 오류: {str(e)}")


@app.get("/")
async def root():
    return {"message": "Virtual Try-On API is running!"}


@app.get("/clothing-items", response_model=List[ClothingItem])
async def get_clothing_items():
    """사용 가능한 옷 목록 반환"""
    items = []
    for template_id, template in CLOTHING_DATA.items():
        # 이미지 URL 생성 (실제 파일이 있으면 해당 경로, 없으면 기본 경로)
        image_path = template.get("image_path", f"static/clothes/{template_id}.png")
        if not image_path.startswith("/"):
            image_path = f"/{image_path}"
        
        items.append(ClothingItem(
            id=template_id,
            name=template.get("name", template_id.replace("_", " ").title()),
            category=template["category"],
            image_url=f"http://localhost:8000{image_path}"
        ))
    return items





@app.post("/detect-pose")
async def detect_pose(file: UploadFile = File(...)):
    """업로드된 이미지에서 포즈를 감지하고 결과 이미지 반환"""
    try:
        # 파일 읽기
        contents = await file.read()
        
        # OpenCV 이미지로 변환
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="유효하지 않은 이미지 파일입니다.")
        
        # 포즈 감지
        pose_image, keypoints = pose_detector.detect_pose(image)
        
        if keypoints is None:
            raise HTTPException(status_code=400, detail="포즈를 감지할 수 없습니다.")
        
        # 결과 이미지를 Base64로 인코딩
        pose_image_base64 = encode_image_to_base64(pose_image)
        
        return {
            "success": True,
            "pose_image": pose_image_base64,
            "keypoints": keypoints,
            "message": "포즈가 성공적으로 감지되었습니다."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"포즈 감지 중 오류가 발생했습니다: {str(e)}")


@app.post("/try-on")
async def virtual_try_on(
    file: UploadFile = File(...),
    selected_clothes: str = Form(...)
):
    """가상 피팅 수행"""
    try:
        print(f"=== 가상 피팅 요청 시작 ===")
        print(f"file: {file.filename if file else None}")
        print(f"selected_clothes: {selected_clothes}")
        print(f"selected_clothes type: {type(selected_clothes)}")
        
        if not selected_clothes:
            print("에러: 선택된 옷이 없습니다.")
            raise HTTPException(status_code=400, detail="선택된 옷이 없습니다.")
        
        # JSON 파싱
        print(f"JSON 파싱 시도: {selected_clothes}")
        clothes_list = json.loads(selected_clothes)
        print(f"파싱된 옷 목록: {clothes_list}")
        
        # 파일 읽기
        print(f"파일 읽기 시작, 크기: {file.size if hasattr(file, 'size') else 'unknown'}")
        contents = await file.read()
        print(f"파일 내용 크기: {len(contents)} bytes")
        
        # OpenCV 이미지로 변환
        nparr = np.frombuffer(contents, np.uint8)
        person_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if person_image is None:
            print("에러: 유효하지 않은 이미지 파일")
            raise HTTPException(status_code=400, detail="유효하지 않은 이미지 파일입니다.")
        
        print(f"이미지 로드 성공, 크기: {person_image.shape}")
        
        # 포즈 감지
        print("포즈 감지 시작")
        pose_image, keypoints = pose_detector.detect_pose(person_image)
        
        if keypoints is None:
            print("에러: 포즈를 감지할 수 없음")
            raise HTTPException(status_code=400, detail="포즈를 감지할 수 없습니다.")
        
        print(f"포즈 감지 성공, 키포인트 수: {len(keypoints)}")
        
        # 옷 입히기
        result_image = person_image.copy()
        
        # 카테고리별로 분류
        selected_categories = []
        for cloth_id in clothes_list:
            if cloth_id in CLOTHING_DATA:
                category = CLOTHING_DATA[cloth_id]["category"]
                selected_categories.append(category)
        
        # 드레스가 선택되었는지 확인
        has_dress = "dress" in selected_categories
        
        if has_dress:
            # 드레스만 착용
            for cloth_id in clothes_list:
                if cloth_id in CLOTHING_DATA and CLOTHING_DATA[cloth_id]["category"] == "dress":
                    # 실제 옷 이미지 로드 시도
                    cloth_image = load_clothing_image(cloth_id)
                    
                    result_image = clothing_warper.fit_from_template_id(
                        result_image, keypoints, cloth_id, cloth_image
                    )
                    break
        else:
            # 하의 먼저, 상의 나중에
            for category in ["bottom", "top"]:
                for cloth_id in clothes_list:
                    if cloth_id in CLOTHING_DATA and CLOTHING_DATA[cloth_id]["category"] == category:
                        # 실제 옷 이미지 로드 시도
                        cloth_image = load_clothing_image(cloth_id)
                        
                        result_image = clothing_warper.fit_from_template_id(
                            result_image, keypoints, cloth_id, cloth_image
                        )
        
        # 결과 이미지를 Base64로 인코딩
        pose_image_base64 = encode_image_to_base64(pose_image)
        result_image_base64 = encode_image_to_base64(result_image)
        
        return {
            "success": True,
            "pose_image": pose_image_base64,
            "result_image": result_image_base64,
            "selected_clothes": clothes_list,
            "message": "가상 피팅이 완료되었습니다."
        }
        
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 에러: {str(e)}")
        raise HTTPException(status_code=400, detail=f"잘못된 JSON 형식입니다: {str(e)}")
    except HTTPException as e:
        print(f"HTTP 에러: {e.detail}")
        raise
    except Exception as e:
        print(f"예상치 못한 에러: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"가상 피팅 중 오류가 발생했습니다: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)