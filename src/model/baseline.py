import torch
from torch import nn
import torch.nn.functional as F


missing_type_index = {'language': 1, 'video': 2, 'audio': 3}


class Head(nn.Module):
    def __init__(self, args, input_dims, output_dims):
        super(Head, self).__init__()

        self.head = nn.Sequential(
            nn.Linear(input_dims, args.fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(args.dropout_prob),
            nn.Linear(args.fusion_dim, output_dims)
        )

    def forward(self, inputs):
        return self.head(inputs)


class modal_sum(nn.Module):
    def __init__(self, args, output_dims):
        super(modal_sum, self).__init__()

        self.modality_types = args.modality_types
        self.modal_proj = nn.ModuleDict({modal: nn.Linear(args.feature_dims, args.fusion_dim) for modal in args.modality_types})
        self.norm = nn.LayerNorm(args.fusion_dim)
        self.head = Head(args, args.fusion_dim, output_dims)

    def forward(self, batch, missing_index):
        # batch: [batchsize, feature_dims]
        inputs = []
        for modal in self.modality_types:
            data = self.modal_proj[modal](batch[modal])
            data[missing_index == missing_type_index[modal]] = torch.zeros_like(data[missing_index == missing_type_index[modal]])
            inputs.append(data)
        inputs = sum(inputs)

        return self.head(self.norm(inputs))


class modal_concat_zero_padding(nn.Module):
    def __init__(self, args, output_dims):
        super(modal_concat_zero_padding, self).__init__()

        self.modality_types = args.modality_types
        self.modal_proj = nn.ModuleDict({modal: nn.Linear(args.feature_dims, args.fusion_dim) for modal in args.modality_types})
        self.norm = nn.LayerNorm(args.fusion_dim * len(args.modality_types))
        self.head = Head(args, args.fusion_dim * len(args.modality_types), output_dims)

    def forward(self, batch, missing_index):
        # batch: [batchsize, feature_dims]
        inputs = []
        for modal in self.modality_types:
            batch[modal][missing_index == missing_type_index[modal]] = torch.zeros_like(batch[modal][missing_index == missing_type_index[modal]])
            inputs.append(self.modal_proj[modal](batch[modal]))
        inputs = torch.cat(inputs, dim=-1)

        return self.head(self.norm(inputs))


# 均值填充--计算每个模态在数据集上的特征均值,使用这些均值来替代缺失模态的特征
class modal_mean_filling(nn.Module):
    def __init__(self, args, output_dims, statistics):
        super(modal_mean_filling, self).__init__()

        self.modality_types = args.modality_types
        self.modal_proj = nn.ModuleDict(
            {modal: nn.Linear(args.feature_dims, args.fusion_dim) for modal in args.modality_types})
        self.norm = nn.LayerNorm(args.fusion_dim)
        self.head = Head(args, args.fusion_dim, output_dims)

        # 注册均值buffer，用于存储当前使用的均值
        for modal in self.modality_types:
            self.register_buffer(f'statistics_{modal}', torch.tensor(statistics[modal], dtype=torch.float))

    def forward(self, batch, missing_index):
        inputs = []
        for modal in self.modality_types:
            # 用该模态的均值填充缺失数据
            data = batch[modal]
            data[missing_index == missing_type_index[modal]] = self.get_buffer(f'statistics_{modal}')
            data = self.modal_proj[modal](data)
            inputs.append(data)
        inputs = sum(inputs)

        return self.head(self.norm(inputs))


# 中位数填充模型--使用每个模态特征的中位数替代缺失值
class modal_median_filling(nn.Module):
    def __init__(self, args, output_dims, statistics):
        super(modal_median_filling, self).__init__()

        self.modality_types = args.modality_types
        self.modal_proj = nn.ModuleDict(
            {modal: nn.Linear(args.feature_dims, args.fusion_dim) for modal in args.modality_types})
        self.norm = nn.LayerNorm(args.fusion_dim)
        self.head = Head(args, args.fusion_dim, output_dims)

        # 注册均值buffer，用于存储当前使用的均值
        for modal in self.modality_types:
            self.register_buffer(f'statistics_{modal}', torch.tensor(statistics[modal], dtype=torch.float))

    def forward(self, batch, missing_index):
        inputs = []
        for modal in self.modality_types:
            data = batch[modal]
            # 用该模态的中位数填充缺失数据
            data[missing_index == missing_type_index[modal]] = self.get_buffer(f'statistics_{modal}')
            data = self.modal_proj[modal](data)
            inputs.append(data)
        inputs = sum(inputs)

        return self.head(self.norm(inputs))


# 使用其他模态信息进行线性回归填充的模型---学习不同模态间的映射关系,使用可用模态通过线性回归预测缺失模态,结合多个源模态的预测结果
class modal_regression_filling(nn.Module):
    def __init__(self, args, output_dims):
        super(modal_regression_filling, self).__init__()

        self.modality_types = args.modality_types
        self.modal_proj = nn.ModuleDict(
            {modal: nn.Linear(args.feature_dims, args.fusion_dim) for modal in args.modality_types})
        self.norm = nn.LayerNorm(args.fusion_dim)
        self.head = Head(args, args.fusion_dim, output_dims)

        # 添加模态间回归模型
        self.cross_modal_regressors = nn.ModuleDict()
        for source_modal in self.modality_types:
            for target_modal in self.modality_types:
                if source_modal != target_modal:
                    key = f"{source_modal}_to_{target_modal}"
                    self.cross_modal_regressors[key] = nn.Linear(args.fusion_dim, args.fusion_dim)

    def forward(self, batch, missing_index):
        # 首先获取所有模态的投影特征
        projected_features = {}
        for modal in self.modality_types:
            projected_features[modal] = self.modal_proj[modal](batch[modal])

        # 处理缺失模态
        for target_modal in self.modality_types:
            target_missing_mask = missing_index == missing_type_index[target_modal]
            if not target_missing_mask.any():
                continue

            # 使用其他模态进行预测
            predictions, missing_masks = [], []
            for source_modal in self.modality_types:
                if source_modal != target_modal:
                    key = f"{source_modal}_to_{target_modal}"

                    pred = self.cross_modal_regressors[key](projected_features[source_modal])
                    source_missing_mask = missing_index == missing_type_index[source_modal]

                    predictions.append(pred)
                    missing_masks.append(torch.ones_like(source_missing_mask, dtype=torch.float, device=pred.device).masked_fill(source_missing_mask, 0.0))

            predictions = torch.stack(predictions, dim=1)
            missing_masks = torch.stack(missing_masks, dim=-1).unsqueeze(-1)
            predictions = predictions * missing_masks

            avg_prediction = predictions.sum(dim=1) / missing_masks.sum(dim=1).clamp(min=1e-6)
            # 只填充缺失的位置
            filled = projected_features[target_modal].clone()
            filled[target_missing_mask] = avg_prediction[target_missing_mask]
            projected_features[target_modal] = filled

        # 融合所有填充后的模态
        inputs = sum(projected_features.values())

        return self.head(self.norm(inputs))


# 模态注意力融合模型---为不同模态分配动态权重,自动调整不同模态的重要性,对缺失模态的注意力权重置零
class modal_attention_fusion(nn.Module):
    def __init__(self, args, output_dims):
        super(modal_attention_fusion, self).__init__()

        self.modality_types = args.modality_types
        self.modal_proj = nn.ModuleDict(
            {modal: nn.Linear(args.feature_dims, args.fusion_dim) for modal in args.modality_types})
        self.norm = nn.LayerNorm(args.fusion_dim)
        self.head = Head(args, args.fusion_dim, output_dims)

        # 为每个模态添加注意力权重
        self.attention = nn.ModuleDict({modal: nn.Linear(args.fusion_dim, 1) for modal in args.modality_types})

    def forward(self, batch, missing_index):
        # 获取各模态特征，计算每个模态的注意力分数，创建注意力掩码矩阵
        modal_features, attention_scores = [], []
        for i, modal in enumerate(self.modality_types):
            modal_missing = missing_index == missing_type_index[modal]
            # 投影特征
            data = self.modal_proj[modal](batch[modal])
            # 处理缺失模态
            data[modal_missing] = torch.zeros_like(data[modal_missing])
            modal_features.append(data)

            # 对每个样本独立计算注意力分数，生成掩码
            attention_score = self.attention[modal](data)
            attention_score[modal_missing] = float('-inf')
            attention_scores.append(attention_score)

        # 将各模态的注意力分数拼接并归一化注意力权重
        all_scores = torch.cat(attention_scores, dim=1)  # [batch_size, num_modalities]
        attention_weights = F.softmax(all_scores, dim=1)  # [batch_size, num_modalities]

        # 加权融合模态特征
        modal_features = torch.stack(modal_features, dim=1)  # [batch_size, num_modalities, fusion_dim]
        modal_features = modal_features * attention_weights.unsqueeze(-1)

        # 求和得到融合特征
        fused_features = modal_features.sum(dim=1)

        return self.head(self.norm(fused_features))


# 使用MAE模块生成缺失模态特征---使用类似MAE(Masked Autoencoder)的架构,通过交叉模态编码器从可用模态推断缺失模态,每个模态都可以被其他模态重建
class modal_MAE_generation(nn.Module):
    def __init__(self, args, output_dims):
        super(modal_MAE_generation, self).__init__()

        self.modality_types = args.modality_types
        self.modal_proj = nn.ModuleDict(
            {modal: nn.Linear(args.feature_dims, args.fusion_dim) for modal in args.modality_types})
        self.norm = nn.LayerNorm(args.fusion_dim * len(args.modality_types))
        self.head = Head(args, args.fusion_dim * len(args.modality_types), output_dims)

        # 交叉模态编码器 - 用于从可用模态推断缺失模态
        self.cross_modal_encoders = nn.ModuleDict()
        for source_modal in self.modality_types:
            encoders = nn.ModuleDict()
            for target_modal in self.modality_types:
                if source_modal != target_modal:
                    encoders[target_modal] = nn.Sequential(
                        nn.Linear(args.fusion_dim, args.fusion_dim * 2),
                        nn.ReLU(),
                        nn.Linear(args.fusion_dim * 2, args.fusion_dim)
                    )
            self.cross_modal_encoders[source_modal] = encoders

        # 融合不同来源生成的模态特征
        self.modal_fusion = nn.ModuleDict({
            modal: nn.Linear(args.fusion_dim * (len(args.modality_types) - 1), args.fusion_dim)
            for modal in args.modality_types
        })

    def forward(self, batch, missing_index):
        # 步骤1: 投影所有可用模态特征
        modal_features = {}
        for modal in self.modality_types:
            modal_features[modal] = self.modal_proj[modal](batch[modal])

        # 步骤2: 重建缺失模态
        for target_modal in self.modality_types:
            target_missing_mask = missing_index == missing_type_index[target_modal]
            if not target_missing_mask.any():
                continue

            source_features = []
            # 从每个可用模态生成目标模态
            for source_modal in self.modality_types:
                if source_modal != target_modal:
                    source_missing_mask = missing_index == missing_type_index[source_modal]

                    # 使用源模态特征预测目标模态特征
                    src_to_tgt = self.cross_modal_encoders[source_modal][target_modal](
                        modal_features[source_modal])

                    # 确保只使用可用的源模态
                    src_to_tgt[source_missing_mask] = torch.zeros_like(src_to_tgt[source_missing_mask])
                    source_features.append(src_to_tgt)

            # 融合所有可用模态生成的目标模态特征
            combined = torch.cat(source_features, dim=-1)
            reconstructed_feature = self.modal_fusion[target_modal](combined)

            # 只替换缺失位置的特征
            filled = modal_features[target_modal].clone()
            filled[target_missing_mask] = reconstructed_feature[target_missing_mask]
            modal_features[target_modal] = filled

        # 步骤3: 融合所有模态特征
        inputs = torch.cat([modal_features[modal] for modal in self.modality_types], dim=-1)

        return self.head(self.norm(inputs))
