def improved_region_growing(image, kmeans_mask=None, threshold=15):
    """
    改进版 Region Growing:
    1. 必须接收 kmeans_mask 来辅助寻找种子点
    2. 如果没有 kmeans_mask，则退化为寻找最亮区域
    """
    preprocessed = preprocess_image(image)
    h, w = preprocessed.shape
    
    seed_point = None
    
    # --- 策略 A: 使用 K-Means 的质心作为种子 (最推荐) ---
    if kmeans_mask is not None and np.sum(kmeans_mask) > 0:
        # 寻找 K-Means 掩膜的最大连通区域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(kmeans_mask, connectivity=8)
        
        # 排除背景(label 0)，找到面积最大的区域
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]) 
            # 获取质心坐标 (x, y)
            cx, cy = centroids[largest_label]
            seed_point = (int(cy), int(cx)) # 注意 numpy 是 (row, col) 即 (y, x)
            
    # --- 策略 B: 如果策略 A 失败，回退到找最亮点 ---
    if seed_point is None:
        blurred = cv2.GaussianBlur(preprocessed, (5, 5), 0)
        seed_point = np.unravel_index(np.argmax(blurred), blurred.shape)

    # 开始生长
    segmented = np.zeros_like(preprocessed, dtype=np.uint8)
    
    # 建立 mask (OpenCV 要求长宽各+2)
    mask = np.zeros((h + 2, w + 2), np.uint8)
    
    # FloodFill
    # loDiff 和 upDiff 控制生长的容差范围
    cv2.floodFill(preprocessed.copy(), mask, seed_point[::-1], 255,
                  loDiff=threshold, upDiff=threshold, flags=cv2.FLOODFILL_FIXED_RANGE)

    segmented = mask[1:-1, 1:-1]
    
    return segmented