# GAI_Project4: 使用 DDPM 前向噪音增強深度圖像先驗（DIP）

本次作業將由擴散概率模型（DDPM）前向過程生成的噪音圖像整合到深度圖像先驗（DIP）框架中。傳統上，DIP 使用高斯分佈的噪音圖像進行訓練。然而，通過利用不同時間步驟的DDPM前向過程中的噪音，我們旨在增強DIP的學習效果，特別是對於暗影圖像。


使用DDPM前向過程中不同時間步驟t生成噪音圖像的方法如下：

# 獲取當前時間步驟的alpha_bar
alpha_bar = torch.tensor([self.alpha_bars[t]])                      
# 重複B次（用於批量計算）
alpha_bar = repeat(alpha_bar, 'C -> B C', B=B).to(self.device)      

noise = alpha_bar.sqrt().view(B, 1, 1, 1) * x.to(self.device) + (1 - alpha_bar).sqrt().view(B, 1, 1, 1) * eta

# 實驗結果
實驗結果顯示，使用從DDPM前向過程生成的噪音圖像可以獲得：

1.較暗影圖像的更高SSIM評估分數。
2.訓練過程中較低的損失，並且時間開支相當。

使用從DDPM前向過程生成的噪音圖像可以改善DIP的學習效果。
這種方法增強了模型從不同噪音水平中有效學習圖像特徵的能力，與傳統的DIP方法形成對比，後者僅依賴於高斯分佈的噪音圖像。

