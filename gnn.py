import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    GATv2Conv,
    GatedGraphConv,
    SAGPooling,
    global_add_pool
)


class BindingAffinityGNN(nn.Module):

    def __init__(self):

        super().__init__()

        # 消息聚合模块用gatv2
        self.gat = GATv2Conv(
            in_channels=20,
            out_channels=20,
            heads=10,
            concat=False,
            edge_dim=8
        )

        # 特征更新模块用gru
        self.gru = GatedGraphConv(
            out_channels=20,
            num_layers=1
        )

        # 池化用SAGpool
        self.pool = SAGPooling(
            in_channels=60,
            ratio=0.3
        )

        # 隐藏层
        self.fc1 = nn.Linear(60, 40)
        self.fc2 = nn.Linear(40, 30)
        self.out = nn.Linear(30, 1)

        self.dropout = nn.Dropout(0.3)

    def forward(self, data):

        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        h0 = x

        # 第一代
        m1 = self.gat(x, edge_index, edge_attr)
        h1 = self.gru(m1, edge_index)

        # 第二代
        m2 = self.gat(h1, edge_index, edge_attr)
        h2 = self.gru(m2, edge_index)

        # 拼接
        H = torch.cat([h0, h1, h2], dim=1)  # 60 dim

        # 池化
        H, edge_index, edge_attr, batch, _, _ = self.pool(
            H, edge_index, edge_attr, batch
        )

        # 读出
        g = global_add_pool(H, batch)

        # 导入隐藏层
        g = F.leaky_relu(self.fc1(g), negative_slope=0.01)
        g = self.dropout(g)

        g = F.leaky_relu(self.fc2(g), negative_slope=0.01)

        out = self.out(g)

        # 把[batch_size, 1]变成[batch_size]
        return out.view(-1)