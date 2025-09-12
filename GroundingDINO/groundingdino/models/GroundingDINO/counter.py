
"""
Counter modules.
"""
from torch import nn
import torch
import torch.nn.functional as F


# 8 16 32 / 1
class DensityRegressor(nn.Module):
    def __init__(self, counter_dim):
        super().__init__()
        # 1/32 -> 1/16
        self.conv0 = nn.Sequential(
            nn.Conv2d(counter_dim * 2, counter_dim, 7, padding=3),
            nn.ReLU()
        )
        
        # 1/16 -> 1/8 (不concat密度图，因为这是第一层)
        self.conv1 = nn.Sequential(
            nn.Conv2d(counter_dim * 3, counter_dim, 5, padding=2),
            nn.ReLU()
        )
        
        # 1/8 -> 1/4 (concat上一层的密度图)
        self.conv2 = nn.Sequential(
            nn.Conv2d(counter_dim * 3, counter_dim, 3, padding=1),
            nn.ReLU()
        )
        
        # 1/4 -> 1/2 (concat上一层的密度图)
        self.conv3 = nn.Sequential(
            nn.Conv2d(counter_dim * 3 + 1, counter_dim, 3, padding=1),
            nn.ReLU()
        )
        
        # Upsample branch (concat上一层的密度图)
        # self.up2x = nn.Sequential(
        #     nn.Conv2d(counter_dim + 1, counter_dim//2, 3, padding=1),  # +1 for m4
        #     nn.ReLU(),
        #     nn.Conv2d(counter_dim//2, counter_dim//4, 1),
        #     nn.ReLU(),
        # )
        self.up2x = nn.Sequential(
            nn.Conv2d(counter_dim + 1, counter_dim//2, 3, padding=1),  # +1 for m4
            nn.ReLU(),
            nn.Conv2d(counter_dim//2, counter_dim//4, 1),
            nn.ReLU(),
            nn.Conv2d(counter_dim//4, 1, 1),
            nn.ReLU(),
        )
        
        # Final density output
        # self.conv4 = nn.Sequential(
        #     nn.Conv2d(counter_dim//4, 1, 1),
        #     nn.ReLU()  # 保持ReLU确保最终输出非负
        # )
        
        # Pyramid heads - 使用Softplus避免梯度截断
        self.pyramid_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(counter_dim, 1, 1),
                nn.Softplus(beta=1, threshold=20)  # 平滑的非负激活
            )
            for _ in range(2)
        ])
        
        self._weight_init_()
        
    def forward(self, features, cnns, img_shape=[1000,1000], hidden_output=True):
        
        x = torch.cat([features[3], cnns[3]], dim=1)
        x1 = self.conv0(x) # 此时的x是1/64
        
        
        # Stage 1: 1/64 -> 1/32
        x = F.interpolate(x1, size=features[2].shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, features[2], cnns[2]], dim=1)
        x2 = self.conv1(x)
        
        # 生成第一个密度图 m2 (1/32 scale)
        # m2 = self.pyramid_heads[0](x2)
        
        
        # Stage 2: 1/32 -> 1/16 (不使用密度图)
        x = F.interpolate(x2, size=features[1].shape[-2:], mode="bilinear", align_corners=False)
        # m2_up = F.interpolate(m2, size=features[1].shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([
            x,
            features[1],
            cnns[1]
        ], dim=1)
        x3 = self.conv2(x)
        
        # 生成第一个高斯密度图 m3 (1/16 scale)
        m3 = self.pyramid_heads[0](x3)
        # Stage 3: 1/16 -> 1/8 (使用m3)
        x = F.interpolate(x3, size=features[0].shape[-2:], mode="bilinear", align_corners=False)
        m3_up = F.interpolate(m3, size=features[0].shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([
            x,
            features[0],
            cnns[0],
            m3_up  # 使用上一层的密度图
        ], dim=1)
        x4 = self.conv3(x)
        
        # 生成第二个高斯密度图 m4 (1/8 scale)
        m4 = self.pyramid_heads[1](x4)
        
        # Final stage: 给1/8的特征拼接最终的高斯密度图(使用m4)
        x = torch.cat([x4, m4], dim=1)  # 修复：concat m4到最终层
        x = self.up2x(x) # 生成最终的密度点图(1/8 scale)
        # Stage 4: 1/8 -> 1 (使用m4)
        # x = F.interpolate(x, size=img_shape, mode='bilinear', align_corners=False)
        # x = self.conv4(x)  # 最终密度图
        
        # 返回pyramid maps用于多尺度监督
        pyramid_maps = [m4, m3]  # 1/8 and 1/16
        
        if hidden_output:
            # x is 1/8 density point map, [x4, x3, x2, x1] scale: [1/8, 1/16, 1/32, 1/64]
            return x, [x4, x3, x2, x1], pyramid_maps
        else:
            return x
    
    def _weight_init_(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.out_channels == 1 and m.kernel_size == (1, 1):
                    # 密度头初始化为小的正值
                    nn.init.constant_(m.weight, 0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

if __name__ == '__main__':
    img_shape = [400, 600]
    encode_vision_feature = torch.rand(2,19947,256)
    spatial_shapes = torch.tensor([[100, 150],
        [ 50,  75],
        [ 25,  38],
        [ 13,  19]])
    N_, S_, C_ = encode_vision_feature.shape
    _cur = 0
    cnn = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        # mask_flatten_ = mask_flatten[:, _cur : (_cur + H_ * W_)].view(N_, H_, W_, 1)
        memory_flatten_ = encode_vision_feature[:, _cur : (_cur + H_ * W_)].view(N_, H_, W_, -1)
        cnn.append(memory_flatten_.permute(0,3,1,2))
        _cur += H_ * W_
    model = DensityRegressor(counter_dim=256)
    output = model(cnn, img_shape)
    print(1)
