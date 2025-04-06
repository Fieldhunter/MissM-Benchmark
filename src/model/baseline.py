import torch
from torch import nn
import torch.nn.functional as F


missing_type_index = {'language': 1, 'video': 2, 'audio': 3}


class modal_sum(nn.Module):
    def __init__(self, args):
        super(modal_sum, self).__init__()

        self.modality_types = args.modality_types
        self.modal_proj = nn.ModuleDict({modal: nn.Linear(args.feature_dims, args.fusion_dim) for modal in args.modality_types})

        self.output_dropout = args.dropout_prob
        self.proj1 = nn.Linear(args.fusion_dim, args.fusion_dim)
        self.proj2 = nn.Linear(args.fusion_dim, 1)

    def forward(self, batch, missing_index):
        # batch: [batchsize, feature_dims]
        inputs = []
        for modal in self.modality_types:
            data = self.modal_proj[modal](batch[modal])
            data[missing_index == missing_type_index[modal]] = torch.zeros_like(data[missing_index == missing_type_index[modal]])
            inputs.append(data)
        inputs = sum(inputs)

        output = self.proj2(
            F.dropout(F.relu(self.proj1(inputs), inplace=True), p=self.output_dropout, training=self.training))

        return output


class modal_concat_zero_padding(nn.Module):
    def __init__(self, args):
        super(modal_concat_zero_padding, self).__init__()

        self.modality_types = args.modality_types
        self.modal_proj = nn.ModuleDict({modal: nn.Linear(args.feature_dims, args.fusion_dim) for modal in args.modality_types})

        self.output_dropout = args.dropout_prob
        self.proj1 = nn.Linear(args.fusion_dim * len(args.modality_types), args.fusion_dim)
        self.proj2 = nn.Linear(args.fusion_dim, 1)

    def forward(self, batch, missing_index):
        # batch: [batchsize, feature_dims]
        inputs = []
        for modal in self.modality_types:
            batch[modal][missing_index == missing_type_index[modal]] = torch.zeros_like(batch[modal][missing_index == missing_type_index[modal]])
            inputs.append(self.modal_proj[modal](batch[modal]))
        inputs = torch.cat(inputs, dim=-1)

        output = self.proj2(
            F.dropout(F.relu(self.proj1(inputs), inplace=True), p=self.output_dropout, training=self.training))

        return output

# 均值填充--计算每个模态在数据集上的特征均值,使用这些均值来替代缺失模态的特征
class modal_mean_filling(nn.Module):
    def __init__(self, args):
        super(modal_mean_filling, self).__init__()

        self.modality_types = args.modality_types
        self.modal_proj = nn.ModuleDict(
            {modal: nn.Linear(args.feature_dims, args.fusion_dim) for modal in args.modality_types})

        self.output_dropout = args.dropout_prob
        self.proj1 = nn.Linear(args.fusion_dim, args.fusion_dim)
        self.proj2 = nn.Linear(args.fusion_dim, 1)

        # 为每个模态存储均值
        self.register_buffer('modal_means', torch.zeros(len(args.modality_types), args.fusion_dim))
        self.mean_initialized = False

    def _update_means(self, batch):
        # 只在第一次前向传播时计算均值
        if not self.mean_initialized:
            for i, modal in enumerate(self.modality_types):
                self.modal_means[i] = self.modal_proj[modal](batch[modal]).mean(dim=0)
            self.mean_initialized = True

    def forward(self, batch, missing_index):
        # 更新模态均值
        self._update_means(batch)

        # batch: [batchsize, feature_dims]
        inputs = []
        for i, modal in enumerate(self.modality_types):
            data = self.modal_proj[modal](batch[modal])
            # 用该模态的均值填充缺失数据
            mask = missing_index == missing_type_index[modal]
            if mask.any():
                data[mask] = self.modal_means[i].unsqueeze(0).repeat(mask.sum(), 1)
            inputs.append(data)
        inputs = sum(inputs)

        output = self.proj2(
            F.dropout(F.relu(self.proj1(inputs), inplace=True), p=self.output_dropout, training=self.training))

        return output

# 中位数填充模型--使用每个模态特征的中位数替代缺失值
class modal_median_filling(nn.Module):
    def __init__(self, args):
        super(modal_median_filling, self).__init__()

        self.modality_types = args.modality_types
        self.modal_proj = nn.ModuleDict(
            {modal: nn.Linear(args.feature_dims, args.fusion_dim) for modal in args.modality_types})

        self.output_dropout = args.dropout_prob
        self.proj1 = nn.Linear(args.fusion_dim, args.fusion_dim)
        self.proj2 = nn.Linear(args.fusion_dim, 1)

        # 为每个模态存储中位数
        self.register_buffer('modal_medians', torch.zeros(len(args.modality_types), args.fusion_dim))
        self.median_initialized = False

    def _update_medians(self, batch):
        # 只在第一次前向传播时计算中位数
        if not self.median_initialized:
            for i, modal in enumerate(self.modality_types):
                projected = self.modal_proj[modal](batch[modal])
                self.modal_medians[i] = torch.median(projected, dim=0)[0]
            self.median_initialized = True

    def forward(self, batch, missing_index):
        # 更新模态中位数
        self._update_medians(batch)

        # batch: [batchsize, feature_dims]
        inputs = []
        for i, modal in enumerate(self.modality_types):
            data = self.modal_proj[modal](batch[modal])
            # 用该模态的中位数填充缺失数据
            mask = missing_index == missing_type_index[modal]
            if mask.any():
                data[mask] = self.modal_medians[i].unsqueeze(0).repeat(mask.sum(), 1)
            inputs.append(data)
        inputs = sum(inputs)

        output = self.proj2(
            F.dropout(F.relu(self.proj1(inputs), inplace=True), p=self.output_dropout, training=self.training))

        return output


#KNN填充模型--为每个缺失样本找到最相似的K个样本,基于余弦相似度计算样本间的相似性.使用加权平均的方式生成填充特征.
class modal_knn_filling(nn.Module):
    def __init__(self, args):
        super(modal_knn_filling, self).__init__()

        self.modality_types = args.modality_types
        self.modal_proj = nn.ModuleDict(
            {modal: nn.Linear(args.feature_dims, args.fusion_dim) for modal in args.modality_types})

        self.output_dropout = args.dropout_prob
        self.proj1 = nn.Linear(args.fusion_dim * len(args.modality_types), args.fusion_dim)
        self.proj2 = nn.Linear(args.fusion_dim, 1)

        # kNN参数
        self.k = args.get('knn_k', 3)  # 默认使用3个最近邻
        self.feature_bank = {modal: [] for modal in args.modality_types}

    def _find_knn(self, modal, available_indices, missing_indices):
        if len(self.feature_bank[modal]) == 0 or len(available_indices) == 0:
            # 如果没有特征库或没有可用数据，返回None
            return None

        features = torch.stack(self.feature_bank[modal])
        # 计算余弦相似度
        sim = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)

        # 对每个缺失样本，找到k个最相似的可用样本
        knn_values = []
        for missing_idx in missing_indices:
            # 获取与其他样本的相似度
            similarities = sim[missing_idx]
            # 只考虑可用样本
            valid_similarities = similarities[available_indices]
            # 找到top-k
            if len(valid_similarities) > 0:
                topk_values, topk_indices = valid_similarities.topk(min(self.k, len(valid_similarities)))
                # 获取相应的特征并加权平均
                topk_features = features[available_indices][topk_indices]
                weights = F.softmax(topk_values, dim=0).unsqueeze(1)
                knn_value = (topk_features * weights).sum(0)
                knn_values.append(knn_value)
            else:
                knn_values.append(torch.zeros_like(features[0]))

        return torch.stack(knn_values) if knn_values else None

    def forward(self, batch, missing_index):
        # batch: [batchsize, feature_dims]
        projected_features = {}

        # 首先获取所有模态的投影特征
        for modal in self.modality_types:
            projected_features[modal] = self.modal_proj[modal](batch[modal])
            # 更新特征库
            self.feature_bank[modal] = projected_features[modal].detach().cpu().tolist()

        # 处理缺失模态
        for modal in self.modality_types:
            missing_mask = missing_index == missing_type_index[modal]
            if not missing_mask.any():
                continue

            missing_indices = torch.where(missing_mask)[0]
            available_indices = torch.where(~missing_mask)[0]

            # 使用kNN填充
            knn_features = self._find_knn(modal, available_indices, missing_indices)
            if knn_features is not None:
                projected_features[modal][missing_mask] = knn_features.to(projected_features[modal].device)
            else:
                # 如果无法使用kNN，则填充零
                projected_features[modal][missing_mask] = torch.zeros_like(projected_features[modal][missing_mask])

        # 拼接各模态特征
        inputs = torch.cat([projected_features[modal] for modal in self.modality_types], dim=-1)

        output = self.proj2(
            F.dropout(F.relu(self.proj1(inputs), inplace=True), p=self.output_dropout, training=self.training))

        return output

# 使用其他模态信息进行线性回归填充的模型----学习不同模态间的映射关系,使用可用模态通过线性回归预测缺失模态,结合多个源模态的预测结果
class modal_regression_filling(nn.Module):
    def __init__(self, args):
        super(modal_regression_filling, self).__init__()

        self.modality_types = args.modality_types
        self.modal_proj = nn.ModuleDict(
            {modal: nn.Linear(args.feature_dims, args.fusion_dim) for modal in args.modality_types})

        # 添加模态间回归模型
        self.cross_modal_regressors = nn.ModuleDict()
        for source_modal in self.modality_types:
            for target_modal in self.modality_types:
                if source_modal != target_modal:
                    key = f"{source_modal}_to_{target_modal}"
                    self.cross_modal_regressors[key] = nn.Linear(args.fusion_dim, args.fusion_dim)

        self.output_dropout = args.dropout_prob
        self.proj1 = nn.Linear(args.fusion_dim, args.fusion_dim)
        self.proj2 = nn.Linear(args.fusion_dim, 1)

    def forward(self, batch, missing_index):
        # batch: [batchsize, feature_dims]
        projected_features = {}

        # 首先获取所有模态的投影特征
        for modal in self.modality_types:
            projected_features[modal] = self.modal_proj[modal](batch[modal])

        # 处理缺失模态
        for target_modal in self.modality_types:
            missing_mask = missing_index == missing_type_index[target_modal]
            if not missing_mask.any():
                continue

            # 使用其他模态进行预测
            predictions = []
            for source_modal in self.modality_types:
                if source_modal != target_modal:
                    key = f"{source_modal}_to_{target_modal}"
                    pred = self.cross_modal_regressors[key](projected_features[source_modal])
                    predictions.append(pred)

            if predictions:
                # 平均所有预测结果
                avg_prediction = torch.stack(predictions).mean(dim=0)
                projected_features[target_modal][missing_mask] = avg_prediction[missing_mask]
            else:
                # 如果没有可用的预测，填充零
                projected_features[target_modal][missing_mask] = torch.zeros_like(
                    projected_features[target_modal][missing_mask])

        # 融合所有模态
        inputs = sum([projected_features[modal] for modal in self.modality_types])

        output = self.proj2(
            F.dropout(F.relu(self.proj1(inputs), inplace=True), p=self.output_dropout, training=self.training))

        return output


# 模态注意力融合模型----为不同模态分配动态权重,自动调整不同模态的重要性,对缺失模态的注意力权重置零
class modal_attention_fusion(nn.Module):
    def __init__(self, args):
        super(modal_attention_fusion, self).__init__()

        self.modality_types = args.modality_types
        self.modal_proj = nn.ModuleDict(
            {modal: nn.Linear(args.feature_dims, args.fusion_dim) for modal in args.modality_types})

        # 为每个模态添加注意力权重
        self.attention = nn.ModuleDict({modal: nn.Linear(args.fusion_dim, 1) for modal in args.modality_types})

        self.output_dropout = args.dropout_prob
        self.proj1 = nn.Linear(args.fusion_dim, args.fusion_dim)
        self.proj2 = nn.Linear(args.fusion_dim, 1)

    def forward(self, batch, missing_index):
        # batch: [batchsize, feature_dims]
        modal_features = {}
        for modal in self.modality_types:
            data = self.modal_proj[modal](batch[modal])
            # 处理缺失模态
            data[missing_index == missing_type_index[modal]] = torch.zeros_like(
                data[missing_index == missing_type_index[modal]])
            modal_features[modal] = data

        # 计算注意力权重
        attention_weights = []
        for modal in self.modality_types:
            weight = F.softmax(self.attention[modal](modal_features[modal]), dim=0)
            # 对缺失模态的注意力权重置零
            weight[missing_index == missing_type_index[modal]] = 0
            attention_weights.append(weight)

        # 归一化权重
        total_weights = sum(attention_weights) + 1e-10  # 避免除零
        normalized_weights = [w / total_weights for w in attention_weights]

        # 加权融合
        inputs = sum([modal_features[modal] * normalized_weights[i] for i, modal in enumerate(self.modality_types)])

        output = self.proj2(
            F.dropout(F.relu(self.proj1(inputs), inplace=True), p=self.output_dropout, training=self.training))

        return output

# 使用MAE模块生成缺失模态特征----使用类似MAE(Masked Autoencoder)的架构,通过交叉模态编码器从可用模态推断缺失模态,每个模态都可以被其他模态重建
class modal_MAE_generation(nn.Module):
    def __init__(self, args):
        super(modal_MAE_generation, self).__init__()

        self.modality_types = args.modality_types
        self.fusion_dim = args.fusion_dim

        # 模态特征投影
        self.modal_proj = nn.ModuleDict(
            {modal: nn.Linear(args.feature_dims, args.fusion_dim) for modal in args.modality_types})

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

        # 输出层保持不变
        self.output_dropout = args.dropout_prob
        self.proj1 = nn.Linear(args.fusion_dim * len(args.modality_types), args.fusion_dim)
        self.proj2 = nn.Linear(args.fusion_dim, 1)

    def forward(self, batch, missing_index):
        # 步骤1: 投影所有可用模态特征
        modal_features = {}
        modal_masks = {}

        for modal in self.modality_types:
            # 原始特征投影
            features = self.modal_proj[modal](batch[modal])
            # 创建模态掩码 (1=可用, 0=缺失)
            mask = (missing_index != missing_type_index[modal]).float().unsqueeze(1)

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

        # 保持原有输出层不变
        output = self.proj2(
            F.dropout(F.relu(self.proj1(inputs), inplace=True), p=self.output_dropout, training=self.training))

        return output



