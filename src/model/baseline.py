import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

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
        self.head = Head(args, args.fusion_dim, output_dims)

    def forward(self, batch, missing_index,statistics):
        # batch: [batchsize, feature_dims]
        inputs = []
        for modal in self.modality_types:
            data = self.modal_proj[modal](batch[modal])
            data[missing_index == missing_type_index[modal]] = torch.zeros_like(data[missing_index == missing_type_index[modal]])
            inputs.append(data)
        inputs = sum(inputs)

        return self.head(inputs)


class modal_concat_zero_padding(nn.Module):
    def __init__(self, args, output_dims):
        super(modal_concat_zero_padding, self).__init__()

        self.modality_types = args.modality_types
        self.modal_proj = nn.ModuleDict({modal: nn.Linear(args.feature_dims, args.fusion_dim) for modal in args.modality_types})
        self.head = Head(args, args.fusion_dim * len(args.modality_types), output_dims)

    def forward(self, batch, missing_index,statistics):
        # batch: [batchsize, feature_dims]
        inputs = []
        for modal in self.modality_types:
            batch[modal][missing_index == missing_type_index[modal]] = torch.zeros_like(batch[modal][missing_index == missing_type_index[modal]])
            inputs.append(self.modal_proj[modal](batch[modal]))
        inputs = torch.cat(inputs, dim=-1)

        return self.head(inputs)


# 均值填充--计算每个模态在数据集上的特征均值,使用这些均值来替代缺失模态的特征
class modal_mean_filling(nn.Module):
    def __init__(self, args, output_dims):
        super(modal_mean_filling, self).__init__()

        self.modality_types = args.modality_types
        self.modal_proj = nn.ModuleDict(
            {modal: nn.Linear(args.feature_dims, args.fusion_dim) for modal in args.modality_types})
        self.head = Head(args, args.fusion_dim, output_dims)

        # 注册均值buffer，用于存储当前使用的均值
        self.register_buffer('modal_means', torch.zeros(len(args.modality_types), args.fusion_dim))

    def forward(self, batch, missing_index,statistics):
        # 更新模态均值
        # self._update_means(batch)
        # 每次前向传播时更新统计值
        with torch.no_grad():
            for i, modal in enumerate(self.modality_types):
                scalar_means = [element[0]
                                for element in statistics[modal]['mean']]
                # print(scalar_means)
                modal_mean = torch.tensor(scalar_means,
                                          dtype=torch.float32,
                                          device=self.modal_means.device)

                # modal_mean = torch.tensor(statistics[modal]['mean'])

                # 应用投影层存储处理后的均值
                self.modal_means[i] = self.modal_proj[modal](modal_mean)

        # batch: [batchsize, feature_dims]
        inputs = []
        for i, modal in enumerate(self.modality_types):
            data = self.modal_proj[modal](batch[modal])
            # 用该模态的均值填充缺失数据
            mask = missing_index == missing_type_index[modal]
            if mask.any():
                # 创建副本以避免修改原始张量
                filled_data = data.clone()
                filled_data[mask] = self.modal_means[i].unsqueeze(0).repeat(mask.sum(), 1)
                data = filled_data
            inputs.append(data)
        inputs = sum(inputs)

        return self.head(inputs)


# 中位数填充模型--使用每个模态特征的中位数替代缺失值
class modal_median_filling(nn.Module):
    def __init__(self, args, output_dims):
        super(modal_median_filling, self).__init__()

        self.modality_types = args.modality_types
        self.modal_proj = nn.ModuleDict(
            {modal: nn.Linear(args.feature_dims, args.fusion_dim) for modal in args.modality_types})
        self.head = Head(args, args.fusion_dim, output_dims)

        # 为每个模态存储中位数
        self.register_buffer('modal_median', torch.zeros(len(args.modality_types), args.fusion_dim))



    def forward(self, batch, missing_index,statistics):
        # 更新模态中位数
        with torch.no_grad():
            for i, modal in enumerate(self.modality_types):
                scalar_medians = [element[0]
                                for element in statistics[modal]['median']]
                # print(scalar_medians)
                modal_median = torch.tensor(scalar_medians,
                                          dtype=torch.float32,
                                          device=self.modal_median.device)

                # modal_mean = torch.tensor(statistics[modal]['mean'])

                # 应用投影层存储处理后的均值
                self.modal_median[i] = self.modal_proj[modal](modal_median)

        # batch: [batchsize, feature_dims]
        inputs = []
        for i, modal in enumerate(self.modality_types):
            data = self.modal_proj[modal](batch[modal])
            # 用该模态的中位数填充缺失数据
            mask = missing_index == missing_type_index[modal]
            if mask.any():
                # 创建新的张量而不是直接修改原始张量
                filled_data = data.clone()
                filled_data[mask] = self.modal_median[i].unsqueeze(0).repeat(mask.sum(), 1)
                data = filled_data
            inputs.append(data)
        inputs = sum(inputs)

        return self.head(inputs)


# 使用其他模态信息进行线性回归填充的模型----学习不同模态间的映射关系,使用可用模态通过线性回归预测缺失模态,结合多个源模态的预测结果
class modal_regression_filling(nn.Module):
    def __init__(self, args, output_dims):
        super(modal_regression_filling, self).__init__()

        self.modality_types = args.modality_types
        self.modal_proj = nn.ModuleDict(
            {modal: nn.Linear(args.feature_dims, args.fusion_dim) for modal in args.modality_types})
        self.head = Head(args, args.fusion_dim, output_dims)

        # 添加模态间回归模型
        self.cross_modal_regressors = nn.ModuleDict()
        for source_modal in self.modality_types:
            for target_modal in self.modality_types:
                if source_modal != target_modal:
                    key = f"{source_modal}_to_{target_modal}"
                    self.cross_modal_regressors[key] = nn.Linear(args.fusion_dim, args.fusion_dim)

    def forward(self, batch, missing_index,statistics):
        # 首先获取所有模态的投影特征
        projected_features = {}
        for modal in self.modality_types:
            projected_features[modal] = self.modal_proj[modal](batch[modal])

        # 处理缺失模态
        filled_features = {}
        for target_modal in self.modality_types:
            # 创建新张量而非直接修改
            filled_features[target_modal] = projected_features[target_modal].clone()

            missing_mask = missing_index == missing_type_index[target_modal]
            if not missing_mask.any():
                continue

            # 使用其他模态进行预测
            predictions = []
            weights = []  # 可选：为不同源模态赋予不同权重

            for source_modal in self.modality_types:
                if source_modal != target_modal:
                    # 检查源模态是否也缺失
                    source_missing_mask = missing_index == missing_type_index[source_modal]

                    # 只使用非缺失的源模态样本进行预测
                    valid_samples = ~source_missing_mask
                    if valid_samples.any():
                        key = f"{source_modal}_to_{target_modal}"
                        # 对所有样本做预测
                        pred = self.cross_modal_regressors[key](projected_features[source_modal])
                        predictions.append(pred)
                        # 简单方案：均等权重
                        weights.append(1.0)
                        # 高级方案：根据非缺失样本比例赋予权重
                        # weights.append(torch.sum(valid_samples).float())

            if predictions:
                # 根据权重平均所有预测结果
                if len(predictions) == 1:
                    # 只有一个预测时直接使用
                    avg_prediction = predictions[0]
                else:
                    # 多个预测时加权平均
                    weights = torch.tensor(weights, device=predictions[0].device)
                    weights = weights / weights.sum()  # 归一化权重

                    weighted_preds = []
                    for i, pred in enumerate(predictions):
                        weighted_preds.append(pred * weights[i])

                    avg_prediction = sum(weighted_preds)

                # 只填充缺失的位置
                filled_features[target_modal][missing_mask] = avg_prediction[missing_mask]
            else:
                # 如果没有可用的预测，填充零
                filled_features[target_modal][missing_mask] = torch.zeros_like(
                    filled_features[target_modal][missing_mask])

        # 融合所有填充后的模态
        inputs = sum([filled_features[modal] for modal in self.modality_types])

        return self.head(inputs)


# 模态注意力融合模型----为不同模态分配动态权重,自动调整不同模态的重要性,对缺失模态的注意力权重置零
class modal_attention_fusion(nn.Module):
    def __init__(self, args, output_dims):
        super(modal_attention_fusion, self).__init__()

        self.modality_types = args.modality_types
        self.modal_proj = nn.ModuleDict(
            {modal: nn.Linear(args.feature_dims, args.fusion_dim) for modal in args.modality_types})
        self.head = Head(args, args.fusion_dim, output_dims)

        # 为每个模态添加注意力权重
        self.attention = nn.ModuleDict({modal: nn.Linear(args.fusion_dim, 1) for modal in args.modality_types})

        # 添加模态间的交互
        self.fusion_layer = nn.Linear(args.fusion_dim * len(args.modality_types), args.fusion_dim)

        # 添加层归一化
        self.norm = nn.LayerNorm(args.fusion_dim)

    def forward(self, batch, missing_index,statistics):
        batch_size = batch[self.modality_types[0]].size(0)

        # 获取各模态特征
        modal_features = {}
        for modal in self.modality_types:
            # 投影特征
            data = self.modal_proj[modal](batch[modal])
            # 标记缺失模态位置
            modal_mask = missing_index == missing_type_index[modal]
            # 处理缺失模态
            if modal_mask.any():
                # 创建新张量而不是直接修改原始张量
                masked_data = data.clone()
                masked_data[modal_mask] = 0.0
                modal_features[modal] = masked_data
            else:
                modal_features[modal] = data

        # 计算每个模态的注意力分数
        attention_scores = {}
        for modal in self.modality_types:
            # 对每个样本独立计算注意力分数
            attention_scores[modal] = self.attention[modal](modal_features[modal])

        # 创建注意力掩码矩阵
        attention_mask = torch.ones(batch_size, len(self.modality_types), device=batch[self.modality_types[0]].device)
        for i, modal in enumerate(self.modality_types):
            modal_mask = missing_index == missing_type_index[modal]
            attention_mask[modal_mask, i] = 0.0

        # 将各模态的注意力分数拼接并应用掩码
        all_scores = torch.cat([attention_scores[modal] for modal in self.modality_types],
                               dim=1)  # [batch_size, num_modalities]
        masked_scores = all_scores * attention_mask

        # 对每个样本在非缺失模态间归一化注意力权重
        attention_weights = F.softmax(masked_scores, dim=1)  # [batch_size, num_modalities]

        # 为了处理所有模态都缺失的极端情况
        zero_mask = (attention_mask.sum(dim=1) == 0).unsqueeze(1)  # 所有模态都缺失的样本
        if zero_mask.any():
            # 对于所有模态都缺失的样本，使用均等权重
            equal_weights = torch.ones_like(attention_weights) / len(self.modality_types)
            attention_weights = torch.where(zero_mask, equal_weights, attention_weights)

        # 加权融合模态特征
        weighted_features = []
        for i, modal in enumerate(self.modality_types):
            # 提取该模态的权重并扩展维度以匹配特征维度
            weight = attention_weights[:, i].unsqueeze(1)  # [batch_size, 1]
            weighted_feature = modal_features[modal] * weight
            weighted_features.append(weighted_feature)

        # 求和得到融合特征
        fused_features = sum(weighted_features)

        # 添加层归一化提高稳定性
        normalized_features = self.norm(fused_features)

        return self.head(normalized_features)


# 使用MAE模块生成缺失模态特征----使用类似MAE(Masked Autoencoder)的架构,通过交叉模态编码器从可用模态推断缺失模态,每个模态都可以被其他模态重建
class modal_MAE_generation(nn.Module):
    def __init__(self, args, output_dims):
        super(modal_MAE_generation, self).__init__()

        self.modality_types = args.modality_types
        self.fusion_dim = args.fusion_dim

        self.modal_proj = nn.ModuleDict(
            {modal: nn.Linear(args.feature_dims, args.fusion_dim) for modal in args.modality_types})
        self.head = Head(args, args.fusion_dim * len(args.modality_types), output_dims)

        # 交叉模态编码器 - 用于从可用模态推断缺失模态
        self.cross_modal_encoders = nn.ModuleDict()
        for target_modal in self.modality_types:
            encoders = nn.ModuleDict()
            for source_modal in self.modality_types:
                if source_modal != target_modal:
                    encoders[source_modal] = nn.Sequential(
                        nn.Linear(args.fusion_dim, args.fusion_dim * 2),
                        nn.ReLU(),
                        nn.Linear(args.fusion_dim * 2, args.fusion_dim)
                    )
            self.cross_modal_encoders[target_modal] = encoders

        # 融合不同来源生成的模态特征
        self.modal_fusion = nn.ModuleDict({
            modal: nn.Linear(args.fusion_dim * (len(args.modality_types) - 1), args.fusion_dim)
            for modal in args.modality_types
        })

    def forward(self, batch, missing_index,statistics):
        # 获取当前设备
        device = next(self.parameters()).device

        # 步骤1: 投影所有可用模态特征
        modal_features = {}
        modal_masks = {}

        for modal in self.modality_types:
            # 原始特征投影
            features = self.modal_proj[modal](batch[modal])
            # 创建模态掩码 (1=可用, 0=缺失)
            mask = (missing_index != missing_type_index[modal]).float().unsqueeze(1).to(device)

            modal_features[modal] = features
            modal_masks[modal] = mask

        # 步骤2: 重建缺失模态
        for target_modal in self.modality_types:
            # 只处理有缺失样本的模态
            if torch.any(modal_masks[target_modal] == 0):
                source_features = []

                # 从每个可用模态生成目标模态
                for source_modal in self.modality_types:
                    if source_modal != target_modal:
                        # 使用源模态特征预测目标模态特征
                        src_to_tgt = self.cross_modal_encoders[target_modal][source_modal](
                            modal_features[source_modal])

                        # 确保只使用可用的源模态
                        src_to_tgt = src_to_tgt * modal_masks[source_modal]
                        source_features.append(src_to_tgt)

                # 融合所有可用模态生成的目标模态特征
                if len(source_features) > 0:
                    combined = torch.cat(source_features, dim=-1)
                    reconstructed_feature = self.modal_fusion[target_modal](combined)

                    # 只替换缺失位置的特征
                    modal_features[target_modal] = modal_features[target_modal] * modal_masks[target_modal] + \
                                                   reconstructed_feature * (1 - modal_masks[target_modal])

        # 步骤3: 融合所有模态特征（现已完全填充）
        all_features = []
        for modal in self.modality_types:
            all_features.append(modal_features[modal])

        inputs = torch.cat(all_features, dim=-1)

        return self.head(inputs)
