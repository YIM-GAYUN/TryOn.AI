import React, { useState, useCallback } from 'react';

const ImageUpload = ({ onImageUpload, uploadedImage }) => {
  const [dragOver, setDragOver] = useState(false);

  const handleFileSelect = useCallback((file) => {
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        onImageUpload(file, e.target.result);
      };
      reader.readAsDataURL(file);
    } else {
      alert('이미지 파일만 업로드할 수 있습니다.');
    }
  }, [onImageUpload]);

  const handleFileInputChange = useCallback((e) => {
    const file = e.target.files[0];
    if (file) {
      handleFileSelect(file);
    }
  }, [handleFileSelect]);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileSelect(files[0]);
    }
  }, [handleFileSelect]);

  const handleUploadAreaClick = useCallback(() => {
    document.getElementById('file-input').click();
  }, []);

  return (
    <div>
      <div 
        className={`upload-area ${dragOver ? 'dragover' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={handleUploadAreaClick}
      >
        {uploadedImage ? (
          <img 
            src={uploadedImage} 
            alt="Uploaded" 
            className="image-preview"
          />
        ) : (
          <div>
            <p>이미지를 드래그하여 놓거나 클릭하여 업로드하세요</p>
            <p style={{ fontSize: '14px', color: '#666' }}>
              전신이 잘 보이는 사진을 선택해주세요
            </p>
          </div>
        )}
      </div>
      
      <input
        id="file-input"
        type="file"
        accept="image/*"
        onChange={handleFileInputChange}
        className="file-input"
      />
      
      {!uploadedImage && (
        <button 
          className="upload-button"
          onClick={handleUploadAreaClick}
        >
          이미지 선택
        </button>
      )}
    </div>
  );
};

export default ImageUpload;