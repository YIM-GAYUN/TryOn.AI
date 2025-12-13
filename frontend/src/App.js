import React, { useState, useCallback } from 'react';
import ImageUpload from './components/ImageUpload';
import ClothingSelection from './components/ClothingSelection';
import ResultDisplay from './components/ResultDisplay';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

function App() {
  const [uploadedImage, setUploadedImage] = useState(null);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [selectedClothes, setSelectedClothes] = useState([]);
  const [poseImage, setPoseImage] = useState(null);
  const [resultImage, setResultImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  const handleImageUpload = useCallback((file, imageDataUrl) => {
    setUploadedImage(imageDataUrl);
    setUploadedFile(file);
    setPoseImage(null);
    setResultImage(null);
    setError(null);
    setSuccess(null);
  }, []);

  const handleClothingSelection = useCallback((clothingIds) => {
    setSelectedClothes(clothingIds);
    setError(null);
  }, []);

  const clearResults = useCallback(() => {
    setPoseImage(null);
    setResultImage(null);
    setError(null);
    setSuccess(null);
  }, []);

  const detectPose = async () => {
    if (!uploadedFile) {
      setError('먼저 이미지를 업로드해주세요.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', uploadedFile);

      const response = await axios.post(`${API_BASE_URL}/detect-pose`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.data.success) {
        setPoseImage(response.data.pose_image);
        setSuccess('포즈가 성공적으로 감지되었습니다!');
      } else {
        setError('포즈 감지에 실패했습니다.');
      }
    } catch (err) {
      console.error('포즈 감지 에러:', err);
      setError(err.response?.data?.detail || '포즈 감지 중 오류가 발생했습니다.');
    } finally {
      setLoading(false);
    }
  };

  const performTryOn = async () => {
    console.log('performTryOn 시작');
    console.log('uploadedFile:', uploadedFile);
    console.log('selectedClothes:', selectedClothes);
    
    if (!uploadedFile) {
      setError('먼저 이미지를 업로드해주세요.');
      return;
    }

    if (selectedClothes.length === 0) {
      setError('옷을 선택해주세요.');
      return;
    }

    setLoading(true);
    setError(null);
    console.log('API 요청 시작');

    try {
      const formData = new FormData();
      formData.append('file', uploadedFile);
      formData.append('selected_clothes', JSON.stringify(selectedClothes));

      console.log('FormData 준비 완료, API 호출:', `${API_BASE_URL}/try-on`);

      const response = await axios.post(`${API_BASE_URL}/try-on`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      console.log('API 응답:', response.data);

      if (response.data.success) {
        setPoseImage(response.data.pose_image);
        setResultImage(response.data.result_image);
        setSuccess('가상 피팅이 완료되었습니다!');
        console.log('가상 피팅 완료');
      } else {
        setError('가상 피팅에 실패했습니다.');
        console.log('가상 피팅 실패');
      }
    } catch (err) {
      console.error('API 호출 에러:', err);
      console.error('에러 응답:', err.response);
      setError(err.response?.data?.detail || '가상 피팅 중 오류가 발생했습니다.');
    } finally {
      setLoading(false);
      console.log('performTryOn 완료');
    }
  };

  return (
    <div className="app-layout">
      <div className="logo-header">
        <img 
          src="/logo.png" 
          alt="Virtual Try-On Logo" 
          className="logo"
          onError={(e) => {
            console.error('로고 이미지 로드 실패');
            e.target.style.display = 'none';
            e.target.nextSibling.style.display = 'block';
          }}
        />
        <h1 className="logo-text" style={{ display: 'none' }}>👗 Virtual Try-On 👔</h1>
      </div>

      <div className="title-header">
        <h1 className="app-title">TryOn.AI: A Virtual Fitting Room Using Pose Estimation and Garment Warping</h1>
      </div>

      <div className="main-content-tab">
        <div className="tab-header">
          <h2>전신 사진을 업로드하고 원하는 옷을 선택해 가상으로 입어보세요!</h2>
        </div>
        
        <div className="container">
          {error && <div className="error">{error}</div>}
          {success && <div className="success">{success}</div>}

      <div className="step-container">
        <h2 className="step-title">1. 전신 사진 업로드</h2>
        <ImageUpload 
          onImageUpload={handleImageUpload}
          uploadedImage={uploadedImage}
        />
        {uploadedImage && (
          <button 
            className="try-on-button" 
            onClick={detectPose}
            disabled={loading}
          >
            {loading ? '포즈 감지 중...' : '포즈 감지하기'}
          </button>
        )}
      </div>

      <div className="step-container">
        <h2 className="step-title">2. 옷 선택</h2>
        <ClothingSelection 
          onSelectionChange={handleClothingSelection}
          selectedClothes={selectedClothes}
        />
        {selectedClothes.length > 0 && uploadedImage && (
          <button 
            className="try-on-button" 
            onClick={performTryOn}
            disabled={loading}
          >
            {loading ? '가상 피팅 중...' : '가상 피팅하기'}
          </button>
        )}
      </div>

      {(poseImage || resultImage) && (
        <div className="step-container">
          <h2 className="step-title">3. 결과</h2>
          <ResultDisplay 
            poseImage={poseImage}
            resultImage={resultImage}
          />
        </div>
      )}

        {loading && (
          <div className="loading">
            <p>처리 중입니다... 잠시만 기다려주세요.</p>
          </div>
        )}
        </div>
      </div>
    </div>
  );
}

export default App;