def improved_kmeans_segmentation(image, k=4, spatial_weight=0.5):
    """
    改进版 K-Means：
    1. 引入空间特征 (x, y)，增强区域连续性
    2. 智能筛选肿瘤簇 (基于亮度和大小)
    3. 增加形态学后处理
    """
    # 1. 获取预处理后的图像 (假设您已经使用了最新的 preprocess_image)
    preprocessed = preprocess_image(image)
    h, w = preprocessed.shape
    
    # 2. 构建特征向量 [亮度, 权重*X, 权重*Y]
    # 归一化亮度到 0-1
    pixel_values = preprocessed.reshape(-1, 1).astype(np.float32) / 255.0
    
    # 生成坐标特征并归一化到 0-1
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    xv, yv = np.meshgrid(x, y)
    
    # spatial_weight 控制空间位置的重要性。
    # 0.5 意味着位置差异对聚类的影响是亮度差异的一半。
    x_features = xv.reshape(-1, 1).astype(np.float32) * spatial_weight
    y_features = yv.reshape(-1, 1).astype(np.float32) * spatial_weight
    
    # 合并特征
    features = np.hstack((pixel_values, x_features, y_features))
    
    # 3. 执行 K-Means
    # n_init=20 增加尝试次数，避免陷入局部最优
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20) 
    labels = kmeans.fit_predict(features)
    segmented = labels.reshape(h, w)
    
    # 4. 智能选择肿瘤簇
    cluster_stats = []
    for i in range(k):
        mask = (segmented == i)
        area = np.sum(mask)
        
        # 计算该簇在原图(预处理后)上的平均亮度
        mean_val = preprocessed[mask].mean()
        
        # 过滤掉面积太小(噪点)或太大(背景)的区域
        # 假设肿瘤面积至少大于 50 像素，且不超过全图的 1/4 (避免选中背景)
        if area > 50 and area < (h * w * 0.25):
            cluster_stats.append({'id': i, 'mean': mean_val})
    
    # 如果没找到合适的簇，退化为简单的最亮簇选择
    if not cluster_stats:
        cluster_means = [preprocessed[segmented == i].mean() for i in range(k)]
        tumor_label = np.argmax(cluster_means)
    else:
        # 按亮度从高到低排序，选最亮的那个
        cluster_stats.sort(key=lambda x: x['mean'], reverse=True)
        tumor_label = cluster_stats[0]['id']

    # 生成二值掩膜
    tumor_mask = (segmented == tumor_label).astype(np.uint8) * 255

    # 5. 简单的形态学清理 (填补内部空洞)
    # 这一步能把肿瘤内部可能存在的“黑洞”补上
    tumor_mask = ndimage.binary_fill_holes(tumor_mask).astype(np.uint8) * 255
    
    return tumor_mask