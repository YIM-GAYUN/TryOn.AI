import cv2
import numpy as np
import os

def create_sample_clothing_images():
    """샘플 옷 이미지들을 생성합니다."""
    
    # 옷 이미지 저장 디렉토리
    clothes_dir = "static/clothes"
    os.makedirs(clothes_dir, exist_ok=True)
    
    # 옷 이미지 정보
    clothing_info = {
        "hoodie_1": {"color": [100, 150, 200], "name": "후디"},
        "tshirt_1": {"color": [255, 255, 255], "name": "티셔츠"},
        "jeans_1": {"color": [139, 69, 19], "name": "진"},
        "shorts_1": {"color": [200, 200, 100], "name": "반바지"},
        "dress_1": {"color": [255, 192, 203], "name": "원피스"},
        "long_dress_1": {"color": [220, 20, 60], "name": "롱드레스"}
    }
    
    for cloth_id, info in clothing_info.items():
        # 기본 이미지 크기
        height, width = 600, 400
        
        # 빈 이미지 생성 (RGBA)
        img = np.zeros((height, width, 4), dtype=np.uint8)
        
        # 옷 모양에 따른 마스크 생성
        if "hoodie" in cloth_id or "tshirt" in cloth_id:
            # 상의 모양
            cv2.rectangle(img, (50, 100), (350, 500), (*info["color"], 255), -1)
            # 소매
            cv2.rectangle(img, (0, 150), (80, 400), (*info["color"], 255), -1)
            cv2.rectangle(img, (320, 150), (400, 400), (*info["color"], 255), -1)
            
        elif "jeans" in cloth_id or "shorts" in cloth_id:
            # 하의 모양
            end_y = 300 if "shorts" in cloth_id else 550
            cv2.rectangle(img, (80, 50), (320, end_y), (*info["color"], 255), -1)
            # 다리 부분
            cv2.rectangle(img, (80, 200), (180, end_y), (*info["color"], 255), -1)
            cv2.rectangle(img, (220, 200), (320, end_y), (*info["color"], 255), -1)
            
        elif "dress" in cloth_id:
            # 원피스 모양
            end_y = 580 if "long" in cloth_id else 450
            # 상체 부분
            cv2.rectangle(img, (50, 80), (350, 280), (*info["color"], 255), -1)
            # 소매
            cv2.rectangle(img, (0, 100), (80, 250), (*info["color"], 255), -1)
            cv2.rectangle(img, (320, 100), (400, 250), (*info["color"], 255), -1)
            # 스커트 부분 (삼각형 모양으로)
            pts = np.array([[80, 280], [320, 280], [350, end_y], [50, end_y]], np.int32)
            cv2.fillPoly(img, [pts], (*info["color"], 255))
        
        # 테두리 추가
        cv2.rectangle(img, (0, 0), (width-1, height-1), (0, 0, 0, 255), 2)
        
        # 파일 저장
        file_path = os.path.join(clothes_dir, f"{cloth_id}.png")
        cv2.imwrite(file_path, img)
        print(f"✓ {info['name']} 이미지 생성: {file_path}")

if __name__ == "__main__":
    create_sample_clothing_images()
    print("\n샘플 옷 이미지 생성 완료!")
    print("실제 옷 이미지로 교체하려면:")
    print("1. static/clothes/ 폴더에 PNG 파일 업로드")
    print("2. clothes_data.json에서 image_path 경로 확인")
    print("3. 각 옷의 anchors 좌표를 이미지에 맞게 조정")