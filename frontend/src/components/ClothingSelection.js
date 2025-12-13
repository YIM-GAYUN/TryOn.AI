import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';

const ClothingSelection = ({ onSelectionChange, selectedClothes }) => {
  const [clothingItems, setClothingItems] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('top');

  // API에서 옷 데이터 가져오기
  const fetchClothingItems = async () => {
    setLoading(true);
    setError(null);
    try {
      console.log('API 요청 시작:', 'http://localhost:8000/clothing-items');
      const response = await axios.get('http://localhost:8000/clothing-items');
      console.log('API 응답:', response.data);
      
      if (response.data && Array.isArray(response.data)) {
        setClothingItems(response.data);
        console.log('옷 데이터 로드 성공:', response.data.length + '개 아이템');
      } else {
        throw new Error('Invalid data format from API');
      }
    } catch (err) {
      console.error('옷 데이터 로드 실패:', err);
      
      if (err.code === 'ECONNREFUSED' || err.message.includes('Network Error')) {
        setError('백엔드 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.');
      } else {
        setError(`옷 데이터를 불러올 수 없습니다: ${err.message}`);
      }
      
      // 백업용 기본 데이터
      setClothingItems([
        {
          id: 'hoodie_1',
          name: '베이직 후디',
          category: 'top',
          image_url: 'http://localhost:8000/static/clothes/top/Hoodie/1.png'
        },
        {
          id: 'panjang_1',
          name: '기본 팬츠',
          category: 'bottom',
          image_url: 'http://localhost:8000/static/clothes/bottom/Celena_Panjang/1.png'
        },
        {
          id: 'gaun_1',
          name: '기본 원피스',
          category: 'dress',
          image_url: 'http://localhost:8000/static/clothes/dress/Gaun/1.png'
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchClothingItems().catch(err => {
      console.error('초기 옷 데이터 로드 에러:', err);
    });
  }, []);

  const handleItemClick = useCallback((itemId, category) => {
    let newSelection = [...selectedClothes];

    // 현재 선택된 아이템인지 확인
    const isAlreadySelected = newSelection.includes(itemId);
    
    if (isAlreadySelected) {
      // 이미 선택된 아이템이면 선택 해제
      newSelection = newSelection.filter(id => id !== itemId);
    } else {
      // 원피스를 선택하는 경우
      if (category === 'dress') {
        // 다른 모든 아이템 선택 해제하고 원피스만 선택
        newSelection = [itemId];
      } else {
        // 상의 또는 하의를 선택하는 경우
        // 먼저 원피스가 선택되어 있는지 확인
        const hasDress = newSelection.some(id => 
          clothingItems.find(item => item.id === id)?.category === 'dress'
        );
        
        if (hasDress) {
          // 원피스가 선택되어 있으면 원피스를 제거하고 새 아이템 추가
          newSelection = newSelection.filter(id => 
            clothingItems.find(item => item.id === id)?.category !== 'dress'
          );
        }
        
        // 같은 카테고리의 다른 아이템이 있으면 교체
        newSelection = newSelection.filter(id => 
          clothingItems.find(item => item.id === id)?.category !== category
        );
        
        newSelection.push(itemId);
      }
    }

    onSelectionChange(newSelection);
  }, [selectedClothes, clothingItems, onSelectionChange]);

  const isItemDisabled = useCallback((itemId, category) => {
    // 원피스가 선택되어 있고 현재 아이템이 원피스가 아닌 경우
    const hasDress = selectedClothes.some(id => 
      clothingItems.find(item => item.id === id)?.category === 'dress'
    );
    
    if (hasDress && category !== 'dress' && !selectedClothes.includes(itemId)) {
      return true;
    }
    
    // 상의나 하의가 선택되어 있고 원피스를 선택하려는 경우는 허용 (교체)
    return false;
  }, [selectedClothes, clothingItems]);

  const getCategoryDisplayName = (category) => {
    switch (category) {
      case 'top': return '상의';
      case 'bottom': return '하의';
      case 'dress': return '원피스';
      default: return category;
    }
  };

  const getFilteredItems = (category) => {
    return clothingItems.filter(item => item.category === category);
  };

  const categories = [
    { key: 'top', name: '상의' },
    { key: 'bottom', name: '하의' },
    { key: 'dress', name: '원피스' }
  ];

  const getCategoryItemCount = (category) => {
    return clothingItems.filter(item => item.category === category).length;
  };

  const handleRefresh = () => {
    // 선택된 옷들 초기화
    onSelectionChange([]);
    // 옷 데이터 새로고침
    fetchClothingItems();
  };

  if (loading) {
    return (
      <div className="loading">
        <p>옷 데이터를 불러오는 중...</p>
      </div>
    );
  }

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
        <p style={{ color: '#666', margin: 0 }}>
          원하는 옷을 선택하세요. 원피스 선택 시 다른 옷은 선택할 수 없습니다.
        </p>
        <button 
          onClick={handleRefresh}
          style={{ 
            padding: '8px 16px', 
            backgroundColor: '#000000ff', 
            color: 'white', 
            border: 'none', 
            borderRadius: '4px',
            cursor: 'pointer',
            fontSize: '14px'
          }}
        >
          새로고침
        </button>
      </div>

      {error && (
        <div className="error" style={{ marginBottom: '15px' }}>
          {error}
        </div>
      )}

      {/* 탭 메뉴 */}
      <div style={{ marginBottom: '20px' }}>
        <div style={{ 
          display: 'flex', 
          borderBottom: '2px solid #e0e0e0',
          marginBottom: '20px'
        }}>
          {categories.map((category) => (
            <button
              key={category.key}
              onClick={() => setActiveTab(category.key)}
              style={{
                padding: '12px 24px',
                border: 'none',
                backgroundColor: 'transparent',
                color: activeTab === category.key ? '#5a5a5aff' : '#666',
                borderBottom: activeTab === category.key ? '3px solid #494949ff' : '3px solid transparent',
                cursor: 'pointer',
                fontSize: '16px',
                fontWeight: activeTab === category.key ? 'bold' : 'normal',
                transition: 'all 0.3s ease',
                position: 'relative'
              }}
            >
              {category.name}
              <span style={{
                marginLeft: '8px',
                backgroundColor: activeTab === category.key ? '#525252ff' : '#ccc',
                color: 'white',
                borderRadius: '10px',
                padding: '2px 6px',
                fontSize: '12px',
                fontWeight: 'bold'
              }}>
                {getCategoryItemCount(category.key)}
              </span>
            </button>
          ))}
        </div>
      </div>
      
      <div className="clothing-grid">
        {getFilteredItems(activeTab).map((item) => (
          <div
            key={item.id}
            className={`clothing-item ${
              selectedClothes.includes(item.id) ? 'selected' : ''
            } ${isItemDisabled(item.id, item.category) ? 'disabled' : ''}`}
            onClick={() => {
              if (!isItemDisabled(item.id, item.category)) {
                handleItemClick(item.id, item.category);
              }
            }}
          >
            <div className="clothing-image">
              {item.image_url ? (
                <img 
                  src={item.image_url} 
                  alt={item.name}
                  style={{ 
                    width: '100%', 
                    height: '100%', 
                    objectFit: 'cover',
                    borderRadius: '5px' 
                  }}
                  onError={(e) => {
                    console.log(`이미지 로드 실패: ${item.image_url}`);
                    e.target.style.display = 'none';
                    if (e.target.nextSibling) {
                      e.target.nextSibling.style.display = 'flex';
                    }
                  }}
                  onLoad={() => {
                    console.log(`이미지 로드 성공: ${item.image_url}`);
                  }}
                />
              ) : null}
              <div 
                style={{ 
                  display: item.image_url ? 'none' : 'flex',
                  width: '100%', 
                  height: '100%', 
                  backgroundColor: '#f0f0f0',
                  alignItems: 'center', 
                  justifyContent: 'center',
                  color: '#666',
                  borderRadius: '5px',
                  fontSize: '14px'
                }}
              >
                {item.name}
              </div>
            </div>
            <div className="clothing-name">{item.name}</div>
            <div className="clothing-category">
              {getCategoryDisplayName(item.category)}
            </div>
          </div>
        ))}
      </div>
      
      {selectedClothes.length > 0 && (
        <div style={{ marginTop: '20px', padding: '15px', backgroundColor: '#f0e5d5', borderRadius: '5px', color: '#000000' }}>
          <strong style={{ color: '#000000' }}>선택된 옷:</strong>
          <ul style={{ margin: '10px 0 0 0', paddingLeft: '20px', color: '#000000' }}>
            {selectedClothes.map(id => {
              const item = clothingItems.find(item => item.id === id);
              return item ? (
                <li key={id} style={{ color: '#000000' }}>
                  {item.name} ({getCategoryDisplayName(item.category)})
                </li>
              ) : null;
            })}
          </ul>
        </div>
      )}
    </div>
  );
};

export default ClothingSelection;