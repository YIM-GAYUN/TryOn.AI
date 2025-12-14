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
    
    def ensure_bgra(self, cloth_img):
        """3채널이면 알파 채널(255)을 추가해서 BGRA로 맞춰줌."""
        if cloth_img is None:
            raise ValueError("cloth_img 가 None 입니다. 경로를 확인하세요.")
        if cloth_img.shape[2] == 4:
            return cloth_img
        b, g, r = cv2.split(cloth_img)
        a = np.full_like(b, 255)
        return cv2.merge([b, g, r, a])

    def make_leg_mask(self, shape, hip, knee, ankle, width):
        """
        단순 사각형 기반 다리 마스크
        """
        mask = np.zeros(shape[:2], dtype=np.uint8)

        pts = np.array([
            [hip[0] - width, hip[1]],
            [hip[0] + width, hip[1]],
            [ankle[0] + width, ankle[1]],
            [ankle[0] - width, ankle[1]],
        ], dtype=np.int32)

        cv2.fillConvexPoly(mask, pts, 255)
        return mask

    def make_leg_mask(self, shape, hip, knee, ankle, width):
        """
        단순 사각형 기반 다리 마스크
        """
        mask = np.zeros(shape[:2], dtype=np.uint8)

        pts = np.array([
            [hip[0] - width, hip[1]],
            [hip[0] + width, hip[1]],
            [ankle[0] + width, ankle[1]],
            [ankle[0] - width, ankle[1]],
        ], dtype=np.int32)

        cv2.fillConvexPoly(mask, pts, 255)
        return mask

    def crop_leg_region(self, cloth_bgra, hip, knee, ankle, padding=30):
        """
        바지 이미지에서 한쪽 다리만 crop
        hip, knee, ankle: cloth 좌표계
        """
        xs = [hip[0], knee[0], ankle[0]]
        ys = [hip[1], knee[1], ankle[1]]

        x_min = max(0, min(xs) - padding)
        x_max = min(cloth_bgra.shape[1], max(xs) + padding)
        y_min = max(0, min(ys) - padding)
        y_max = min(cloth_bgra.shape[0], max(ys) + padding)

        cropped = cloth_bgra[y_min:y_max, x_min:x_max].copy()

        offset = (x_min, y_min)
        return cropped, offset
    
    def get_src_dst_points_from_template(self, cloth_meta, keypoints):
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
    
    ## ----- Homography + RANSAC 방식 적용 -----
    def fit_cloth_with_template(self, person_img_bgr, keypoints, cloth_img, cloth_meta):
        """
        - person_img_bgr : 사람 원본 이미지 (BGR)
        - keypoints      : detectClothingPose() 에서 얻은 포즈 키포인트 dict
        - cloth_img      : 옷 PNG (BGR 또는 BGRA)
        - cloth_meta     : size.json 에서 읽은 해당 옷 템플릿(dict)
        """
        ph, pw = person_img_bgr.shape[:2]
        cloth_bgra = self.ensure_bgra(cloth_img)

        # === (1) src / dst 포인트 ===
        src_pts, dst_pts = self.get_src_dst_points_from_template(cloth_meta, keypoints)

        # 포인트 개수 체크
        if len(src_pts) < 4:
            raise ValueError("Homography는 최소 4개 대응점이 필요합니다.")

        # === (2) Homography (교수님 코드 핵심) ===
        H, inlier_mask = cv2.findHomography(
            src_pts,
            dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0
        )

        if H is None:
            raise RuntimeError("Homography 계산 실패")

        # === (3) warpPerspective ===
        warped = cv2.warpPerspective(
            cloth_bgra,
            H,
            (pw, ph),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )

        # === (4) 알파 블렌딩 ===
        wb, wg, wr, wa = cv2.split(warped)
        cloth_rgb = cv2.merge([wb, wg, wr])

        alpha = wa.astype(np.float32) / 255.0
        alpha = cv2.GaussianBlur(alpha, (0, 0), 1.5)
        alpha_3 = cv2.merge([alpha, alpha, alpha])

        base = person_img_bgr.astype(np.float32)
        out = cloth_rgb.astype(np.float32) * alpha_3 + base * (1.0 - alpha_3)
        out = np.clip(out, 0, 255).astype(np.uint8)

        return out

    def alpha_blend_bgra(self, base_bgra, overlay_bgra, blur_sigma=1.2):
        """
        base_bgra, overlay_bgra: BGRA 이미지
        overlay를 base 위에 알파 블렌딩
        """
        # overlay 분리
        ob, og, or_, oa = cv2.split(overlay_bgra)
        overlay_rgb = cv2.merge([ob, og, or_])

        alpha = oa.astype(np.float32) / 255.0
        alpha = cv2.GaussianBlur(alpha, (0, 0), blur_sigma)
        alpha_3 = cv2.merge([alpha, alpha, alpha])

        base_rgb = base_bgra[:, :, :3].astype(np.float32)

        out_rgb = overlay_rgb.astype(np.float32) * alpha_3 \
                + base_rgb * (1.0 - alpha_3)

        out = base_bgra.copy()
        out[:, :, :3] = np.clip(out_rgb, 0, 255).astype(np.uint8)
        out[:, :, 3] = np.maximum(base_bgra[:, :, 3], oa)

        return out

    def fit_bottom_piecewise(self, person_img_bgr, keypoints, cloth_img, cloth_meta):
        cloth_bgra = self.ensure_bgra(cloth_img)
        ph, pw = person_img_bgr.shape[:2]

        canvas = np.zeros((ph, pw, 4), dtype=np.uint8)

        hip_dist = abs(keypoints["left_hip"][0] - keypoints["right_hip"][0])
        # Colab 코드 기반으로 수정하되 더 넓게
        width_px = max(50, int(0.5 * hip_dist))  # Colab 12%에서 50%로 증가

        for side, sign in [("left", 1), ("right", -1)]:

            # ---------- 사람 좌표 ----------
            dst_hip   = keypoints[f"{side}_hip"][:2]
            dst_knee  = keypoints[f"{side}_knee"][:2]
            dst_ankle = keypoints[f"{side}_ankle"][:2]

            # ---------- 옷 좌표 ----------
            src_hip   = cloth_meta["anchors"][f"{side}_hip"]
            # knee 정보가 없으면 hip과 ankle의 중점 사용
            if f"{side}_knee" in cloth_meta["anchors"]:
                src_knee = cloth_meta["anchors"][f"{side}_knee"]
            else:
                src_ankle_pos = cloth_meta["anchors"][f"{side}_ankle"]
                src_knee = [(src_hip[0] + src_ankle_pos[0]) // 2,
                           (src_hip[1] + src_ankle_pos[1]) // 2]
            src_ankle = cloth_meta["anchors"][f"{side}_ankle"]

            # ---------- 1. 다리 crop ----------
            leg_crop, (ox, oy) = self.crop_leg_region(
                cloth_bgra, src_hip, src_knee, src_ankle
            )

            # crop 좌표계로 변환
            src_hip_c   = (src_hip[0] - ox, src_hip[1] - oy)
            src_ankle_c = (src_ankle[0] - ox, src_ankle[1] - oy)

            # ---------- 2. affine ----------
            src_pts = np.float32([
                src_hip_c,
                src_ankle_c,
                [src_hip_c[0] + sign * width_px, src_hip_c[1]]
            ])
            dst_pts = np.float32([
                dst_hip,
                dst_ankle,
                [dst_hip[0] + sign * width_px, dst_hip[1]]
            ])

            M = cv2.getAffineTransform(src_pts, dst_pts)

            warped = cv2.warpAffine(
                leg_crop, M, (pw, ph),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0,0,0,0)
            )

            # ---------- 3. 합성 ----------
            canvas = self.alpha_blend_bgra(canvas, warped)

        # ---------- 최종 ----------
        final = self.alpha_blend_bgra(
            self.ensure_bgra(person_img_bgr),
            canvas
        )

        return final[:, :, :3]
    
    def fit_from_template_id(self, person_img_bgr, keypoints, template_id, cloth_img):
        """template_id로부터 가상 피팅을 수행합니다."""
        if template_id not in self.cloth_templates:
            raise KeyError(f"템플릿에 '{template_id}'이 없습니다.")
        
        meta = self.cloth_templates[template_id]
        
        if cloth_img is None:
            raise ValueError("옷 이미지가 None입니다.")

        if meta["category"] == "bottom":
            return self.fit_bottom_piecewise(
                person_img_bgr, keypoints, cloth_img, meta
            )

        # 상의 / 원피스
        return self.fit_cloth_with_template(
            person_img_bgr, keypoints, cloth_img, meta
        )
    
    def set_cloth_templates(self, templates: Dict):
        """옷 템플릿을 설정합니다."""
        self.cloth_templates = templates