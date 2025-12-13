import math
import cv2
import numpy as np
import os
from time import time
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing
import json
from typing import Dict, List, Tuple, Optional


class PoseDetector:
    def __init__(self):
        self.pose_model = mp_pose.Pose(
            static_image_mode=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
            model_complexity=2
        )
        
        # 상·하의 피팅 기준 핵심 landmark
        self.key_landmarks = {
            # 얼굴 기준점 (정렬용)
            "nose": 0,
            
            # 상의 (어깨, 팔)
            "left_shoulder": 11,
            "right_shoulder": 12,
            "left_elbow": 13,
            "right_elbow": 14,
            "left_wrist": 15,
            "right_wrist": 16,
            
            # 하의 (골반, 다리)
            "left_hip": 23,
            "right_hip": 24,
            "left_knee": 25,
            "right_knee": 26,
            "left_ankle": 27,
            "right_ankle": 28
        }
    
    def extract_key_landmarks(self, landmarks: List[Tuple[int, int, float]]) -> Dict:
        """핵심 랜드마크만 추출하는 함수"""
        key_points = {}
        
        for name, idx in self.key_landmarks.items():
            if idx < len(landmarks):
                lx, ly, lz = landmarks[idx]
                key_points[name] = (lx, ly, lz)
        
        return key_points
    
    def detect_pose(self, image: np.ndarray) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        의류 피팅용 핵심 landmark만 추출하는 함수.
        - 원본 이미지에서 pose detection 수행
        - 핵심 12개 landmark만 dict로 반환
        """
        output_image = image.copy()
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = self.pose_model.process(img_rgb)
        
        height, width, _ = image.shape
        all_landmarks = []
        
        # 랜드마크 감지되었을 때
        if results.pose_landmarks:
            
            # 그림 그리기 (선과 점을 훨씬 더 굵게)
            mp_drawing.draw_landmarks(
                output_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=20, circle_radius=15),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=12)
            )
            
            # 33개 전체 landmark 추출
            for lm in results.pose_landmarks.landmark:
                x = int(lm.x * width)
                y = int(lm.y * height)
                z = lm.z          # z는 scale하지 않는 것이 좋음
                all_landmarks.append((x, y, z))
            
            # ---- 핵심 12개 landmark만 추출 ----
            keypoints = self.extract_key_landmarks(all_landmarks)
            
        else:
            keypoints = None
        
        return output_image, keypoints


class ClothingWarper:
    def __init__(self, cloth_templates_path: Optional[str] = None):
        self.cloth_templates = {}
        if cloth_templates_path and os.path.exists(cloth_templates_path):
            with open(cloth_templates_path, "r", encoding="utf-8") as f:
                self.cloth_templates = json.load(f)
    
    def ensure_bgra(self, cloth_img: np.ndarray) -> np.ndarray:
        """3채널이면 알파 채널(255)을 추가해서 BGRA로 맞춰줌."""
        if cloth_img is None:
            raise ValueError("cloth_img 가 None 입니다. 경로를 확인하세요.")
        if len(cloth_img.shape) == 3 and cloth_img.shape[2] == 4:
            return cloth_img
        elif len(cloth_img.shape) == 3 and cloth_img.shape[2] == 3:
            b, g, r = cv2.split(cloth_img)
            a = np.full_like(b, 255)
            return cv2.merge([b, g, r, a])
        else:
            raise ValueError(f"잘못된 이미지 형식: {cloth_img.shape}")
    
    def get_src_dst_points_from_template(self, cloth_meta: Dict, keypoints: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        size.json 의 anchors(옷 좌표)와
        pose keypoints(사람 좌표)를 매칭해서
        src_pts(옷), dst_pts(사람) 배열을 만들기
        """
        src_pts, dst_pts = [], []
        
        for name, pos in cloth_meta["anchors"].items():
            if name not in keypoints:
                # 이 키포인트는 pose에서 못 찾았으면 스킵
                continue
            
            cx, cy = pos              # 옷 이미지 안에서의 anchor
            px, py, _ = keypoints[name]  # 사람 이미지 안에서의 keypoint
            
            src_pts.append([cx, cy])
            dst_pts.append([px, py])
        
        src_pts = np.float32(src_pts)
        dst_pts = np.float32(dst_pts)
        
        if len(src_pts) < 3:
            raise ValueError(f"매칭 가능한 포인트가 3개 미만입니다. anchors: {cloth_meta['anchors'].keys()}")
        
        return src_pts, dst_pts
    
    def fit_cloth_with_template(self, person_img_bgr: np.ndarray, keypoints: Dict, 
                              cloth_img: np.ndarray, cloth_meta: Dict) -> np.ndarray:
        """
        - person_img_bgr : 사람 원본 이미지 (BGR)
        - keypoints      : detectClothingPose() 에서 얻은 포즈 키포인트 dict
        - cloth_img      : 옷 PNG (BGR 또는 BGRA)
        - cloth_meta     : size.json 에서 읽은 해당 옷 템플릿(dict)
        """
        ph, pw = person_img_bgr.shape[:2]
        cloth_bgra = self.ensure_bgra(cloth_img)
        
        src_pts, dst_pts = self.get_src_dst_points_from_template(cloth_meta, keypoints)
        
        # Affine 변환 행렬 추정
        M, _ = cv2.estimateAffine2D(src_pts, dst_pts)
        
        # 옷을 사람 이미지 크기에 맞게 warp
        warped = cv2.warpAffine(
            cloth_bgra, M, (pw, ph),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )
        
        # 알파 블렌딩
        wb, wg, wr, wa = cv2.split(warped)
        cloth_rgb = cv2.merge([wb, wg, wr])
        alpha = wa.astype(np.float32) / 255.0
        alpha_3 = cv2.merge([alpha, alpha, alpha])
        
        base = person_img_bgr.astype(np.float32)
        out = cloth_rgb.astype(np.float32) * alpha_3 + base * (1.0 - alpha_3)
        out = np.clip(out, 0, 255).astype(np.uint8)
        return out
    
    def fit_from_template_id(self, person_img_bgr: np.ndarray, keypoints: Dict, 
                           template_id: str, cloth_img: np.ndarray) -> np.ndarray:
        """
        template_id: size.json 에 정의한 키 이름 (예: 'hoodie_1', 'jeans_1', 'gown_1')
        """
        if template_id not in self.cloth_templates:
            raise KeyError(f"템플릿에 '{template_id}'이 없습니다.")
        
        meta = self.cloth_templates[template_id]
        
        if cloth_img is None:
            raise ValueError("옷 이미지가 None입니다.")
        
        result = self.fit_cloth_with_template(person_img_bgr, keypoints, cloth_img, meta)
        return result
    
    def set_cloth_templates(self, templates: Dict):
        """옷 템플릿을 설정합니다."""
        self.cloth_templates = templates