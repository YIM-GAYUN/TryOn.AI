import React from 'react';

const ResultDisplay = ({ poseImage, resultImage }) => {
  return (
    <div className="result-display">
      {poseImage && (
        <div className="result-item">
          <h4>포즈 감지 결과</h4>
          <img 
            src={poseImage} 
            alt="Pose Detection Result"
          />
          <p style={{ marginTop: '10px', fontSize: '14px', color: '#666' }}>
            사람의 주요 관절점이 감지되었습니다.
          </p>
        </div>
      )}
      
      {resultImage && (
        <div className="result-item">
          <h4>가상 피팅 결과</h4>
          <img 
            src={resultImage} 
            alt="Virtual Try-On Result"
          />
          <p style={{ marginTop: '10px', fontSize: '14px', color: '#666' }}>
            선택한 옷이 입혀진 모습입니다.
          </p>
        </div>
      )}
    </div>
  );
};

export default ResultDisplay;