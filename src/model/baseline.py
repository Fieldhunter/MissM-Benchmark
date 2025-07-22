from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer
import torch
from torch import nn
from torch_geometric.nn import SuperGATConv
from torch_geometric.data import Batch, Data


missing_type_index = {'language': 1, 'video': 2, 'audio': 3, 'image': 4}


class fusion_gcn(nn.Module):
    def __init__(self, in_channels=256, hidden_dim=128, output_dim=256, heads=4):
        super().__init__()
        self.gat1 = SuperGATConv(in_channels, hidden_dim, heads=heads, concat=True)
        self.gat2 = SuperGATConv(hidden_dim * heads, output_dim, heads=1, concat=False)
        self.act = nn.GELU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gat1(x, edge_index)
        x = self.act(x)
        x = self.gat2(x, edge_index)

        return x


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


# arithmetic operation-based representation composition method (Fig. 7)
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


# Zero/Mean/Median values composition method (Fig. 3)
class modal_concat(nn.Module):
    def __init__(self, args, output_dims):
        super(modal_concat, self).__init__()

        self.modality_types = args.modality_types
        self.modal_proj = nn.ModuleDict({modal: nn.Linear(args.feature_dims, args.fusion_dim) for modal in args.modality_types})
        self.norm = nn.LayerNorm(args.fusion_dim * len(args.modality_types))
        self.head = Head(args, args.fusion_dim * len(args.modality_types), output_dims)

        for modal in self.modality_types:
            self.register_buffer(f'statistics_{modal}', torch.zeros(args.feature_dims, dtype=torch.float, device='cpu'))

    def forward(self, batch, missing_index):
        # batch: [batchsize, feature_dims]
        inputs = []
        for modal in self.modality_types:
            if (missing_index == missing_type_index[modal]).any():
                batch[modal][missing_index == missing_type_index[modal]] = self.get_buffer(f'statistics_{modal}')
            inputs.append(self.modal_proj[modal](batch[modal]))
        inputs = torch.cat(inputs, dim=-1)

        return self.head(self.norm(inputs))

    def set_statistics(self, statistics, modality_types):
        for modal in modality_types:
            self.register_buffer(f'statistics_{modal}', torch.tensor(statistics[modal], dtype=torch.float, device=self.modal_proj[modal].weight.device))


# Direct-to-Task Representation Generation Method (Fig. 8b)
class modal_regression(nn.Module):
    def __init__(self, args, output_dims):
        super(modal_regression, self).__init__()

        self.modality_types = args.modality_types
        self.modal_proj = nn.ModuleDict(
            {modal: nn.Linear(args.feature_dims, args.fusion_dim) for modal in args.modality_types})
        self.norm = nn.LayerNorm(args.fusion_dim * len(args.modality_types))
        self.head = Head(args, args.fusion_dim * len(args.modality_types), output_dims)

        # 添加模态间回归模型
        self.cross_modal_regressors = nn.ModuleDict()
        for source_modal in self.modality_types:
            for target_modal in self.modality_types:
                if source_modal != target_modal:
                    key = f"{source_modal}_to_{target_modal}"
                    self.cross_modal_regressors[key] = nn.Linear(args.feature_dims, args.fusion_dim)

    def forward(self, batch, missing_index):
        # 获取所有模态的投影特征
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

                    pred = self.cross_modal_regressors[key](batch[source_modal])
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
        inputs = torch.cat([projected_features[modal] for modal in self.modality_types], dim=-1)

        return self.head(self.norm(inputs))


# Retrieval-based modality composition method (Fig. 4)
class modal_concat_full(nn.Module):
    def __init__(self, args, output_dims):
        super(modal_concat_full, self).__init__()

        self.modality_types = args.modality_types
        self.modal_proj = nn.ModuleDict({modal: nn.Linear(args.feature_dims, args.fusion_dim) for modal in args.modality_types})
        self.norm = nn.LayerNorm(args.fusion_dim * len(args.modality_types))
        self.head = Head(args, args.fusion_dim * len(args.modality_types), output_dims)

    def forward(self, batch, missing_index):
        # batch: [batchsize, feature_dims]
        inputs = []
        for modal in self.modality_types:
            inputs.append(self.modal_proj[modal](batch[modal]))
        inputs = torch.cat(inputs, dim=-1)

        return self.head(self.norm(inputs))


# Intra-Modality Attention Method (Fig. 9a)
class modal_intra_channel_attention(nn.Module):
    def __init__(self, args, output_dims):
        super(modal_intra_channel_attention, self).__init__()

        self.modality_types = args.modality_types
        self.modal_proj = nn.ModuleDict(
            {modal: nn.Linear(args.feature_dims, args.fusion_dim) for modal in args.modality_types})
        self.norm = nn.LayerNorm(args.fusion_dim)
        self.head = Head(args, args.fusion_dim, output_dims)

        self.fusion_representation = nn.Parameter(torch.randn(1, args.fusion_dim))
        self.channel_attention = nn.Sequential(
            nn.Linear(args.fusion_dim * 2, args.fusion_dim // 4),
            nn.ReLU(),
            nn.Linear(args.fusion_dim // 4, args.fusion_dim),
            nn.Sigmoid()
        )

    def forward(self, batch, missing_index):
        # 获取所有模态的投影特征
        inputs = []
        for modal in self.modality_types:
            data = self.modal_proj[modal](batch[modal])
            B, _ = data.shape
            channel_attention = self.channel_attention(torch.cat([data, self.fusion_representation.expand(B, -1)], dim=-1))
            data = data * channel_attention
            data[missing_index == missing_type_index[modal]] = torch.zeros_like(data[missing_index == missing_type_index[modal]])
            inputs.append(data)
        inputs = sum(inputs)

        return self.head(self.norm(inputs))


# Inter-Modality Attention Method (Fig. 9b)
class modal_inter_attention(nn.Module):
    def __init__(self, args, output_dims):
        super(modal_inter_attention, self).__init__()

        self.modality_types = args.modality_types
        self.modal_proj = nn.ModuleDict(
            {modal: nn.Linear(args.feature_dims, args.fusion_dim) for modal in args.modality_types})
        self.norm = nn.LayerNorm(args.fusion_dim)
        self.head = Head(args, args.fusion_dim, output_dims)

        self.query_token = nn.Parameter(torch.randn(1, 1, args.fusion_dim))
        self.attn = nn.MultiheadAttention(args.fusion_dim, num_heads=4, batch_first=True)

    def forward(self, batch, missing_index):
        features = []
        attn_mask = []
        for modal in self.modality_types:
            feat = self.modal_proj[modal](batch[modal])  # [B, D]
            features.append(feat.unsqueeze(1))
            attn_mask.append((missing_index == missing_type_index[modal]).unsqueeze(1))
        tokens = torch.cat(features, dim=1)  # [B, num_modal, D]
        mask = torch.cat(attn_mask, dim=1)  # [B, num_modal]

        query = self.query_token.expand(tokens.shape[0], -1, -1)  # [B, 1, D]

        # 只用query做query，tokens做key/value
        attn_out, _ = self.attn(query, tokens, tokens, key_padding_mask=mask.bool())
        inputs = attn_out[:, 0, :]  # [B, D]

        return self.head(self.norm(inputs))


# Graph Fusion Method (Fig. 13)
class modal_graph_fusion(nn.Module):
    def __init__(self, args, output_dims):
        super(modal_graph_fusion, self).__init__()

        self.modality_types = args.modality_types
        self.modal_proj = nn.ModuleDict(
            {modal: nn.Linear(args.feature_dims, args.fusion_dim) for modal in args.modality_types})
        self.norm = nn.LayerNorm(args.fusion_dim)
        self.head = Head(args, args.fusion_dim, output_dims)

        self.gcn = fusion_gcn()

    def forward(self, batch, missing_index):
        # 获取所有模态的投影特征
        B = list(batch.values())[0].shape[0]
        projected_features = []
        missing_modal_index = torch.ones((B, len(self.modality_types)), device=self.head.head[0].weight.device)
        for i, modal in enumerate(self.modality_types):
            projected_features.append(self.modal_proj[modal](batch[modal]))
            missing_modal_index[:, i][missing_index==missing_type_index[modal]] = torch.zeros_like(missing_modal_index[:, i][missing_index==missing_type_index[modal]])
        
        projected_features = torch.stack(projected_features, dim=1)
        all_graphs = [Data(x=projected_features[i], edge_index=self.bulid_edge(missing_modal_index[i])) for i in range(len(projected_features))]
        _, M, C = projected_features.shape

        batch_graph = Batch.from_data_list(all_graphs)
        gcn_out = self.gcn(batch_graph).view(B, M, -1).mean(dim=-2)

        return self.head(self.norm(gcn_out))
    
    def bulid_edge(self, sample):
        start = []
        end = []
        for i in range(len(sample)):
            for j in range(i+1, len(sample)):
                if sample[i] == 1 and sample[j] == 1:
                    start.append(i)
                    end.append(j)
        
        return torch.tensor([start+end, end+start], dtype=torch.long, device=sample.device)


# Unified GNN Method (Fig. 14b)
class modal_unified_graph(nn.Module):
    def __init__(self, args, output_dims):
        super(modal_unified_graph, self).__init__()

        self.modality_types = args.modality_types
        self.norm = nn.LayerNorm(args.fusion_dim)
        self.head = Head(args, args.fusion_dim, output_dims)

        self.complete_gcn = fusion_gcn(in_channels=768, hidden_dim=384, output_dim=768)
        self.fusion_gcn = fusion_gcn(in_channels=768)

    def forward(self, batch, missing_index):
        # 获取所有模态的投影特征
        B = list(batch.values())[0].shape[0]
        features = []
        missing_modal_index = torch.ones((B, len(self.modality_types)), device=self.head.head[0].weight.device)
        for i, modal in enumerate(self.modality_types):
            features.append(batch[modal])
            missing_modal_index[:, i][missing_index==missing_type_index[modal]] = torch.zeros_like(missing_modal_index[:, i][missing_index == missing_type_index[modal]])
        
        features = torch.stack(features, dim=1)
        complete_graphs = [Data(x=features[i], edge_index=self.bulid_edge(missing_modal_index[i])) for i in range(len(features))]
        _, M, C = features.shape

        complete_batch_graph = Batch.from_data_list(complete_graphs)
        complete_gcn_out = self.complete_gcn(complete_batch_graph).view(B, M, -1)
        all_features = []
        for i, modal in enumerate(self.modality_types):
            batch[modal][missing_index==missing_type_index[modal]] = complete_gcn_out[:, i][missing_index==missing_type_index[modal]]
            all_features.append(batch[modal])
        
        all_features = torch.stack(all_features, dim=1)
        all_graphs = [Data(x=all_features[i], edge_index=self.bulid_edge(torch.ones((len(self.modality_types)), device=self.head.head[0].weight.device))) for i in range(len(all_features))]
        
        batch_graph = Batch.from_data_list(all_graphs)
        gcn_out = self.fusion_gcn(batch_graph).view(B, M, -1).mean(dim=-2)

        return self.head(self.norm(gcn_out))
    
    def bulid_edge(self, sample):
        start = []
        end = []
        for i in range(len(sample)):
            for j in range(i+1, len(sample)):
                if sample[i] == 1 and sample[j] == 1:
                    start.append(i)
                    end.append(j)
        
        return torch.tensor([start+end, end+start], dtype=torch.long, device=sample.device)


# Dedicated Training Method (Fig. 16)
class modal_dedicated_dnn(nn.Module):
    def __init__(self, args, output_dims):
        super(modal_dedicated_dnn, self).__init__()

        self.modality_types = args.modality_types
        dedicated_dnn = {modal: nn.Linear(args.feature_dims * (len(self.modality_types) - 1), args.fusion_dim) for modal in args.modality_types}
        dedicated_dnn['full'] = nn.Linear(args.feature_dims * len(self.modality_types), args.fusion_dim)
        self.dedicated_dnn = nn.ModuleDict(dedicated_dnn)
        self.norm = nn.LayerNorm(args.fusion_dim)
        self.head = Head(args, args.fusion_dim, output_dims)

    def forward(self, batch, missing_index):
        features = torch.stack([batch[modal] for modal in self.modality_types], dim=1)
        B, M, C = features.shape

        inputs = self.dedicated_dnn['full'](features.view(B, -1))
        for i, modal in enumerate(self.modality_types):
            inputs[missing_index==missing_type_index[modal]] = self.dedicated_dnn[modal](torch.cat([features[:, :i], features[:, i+1:]], dim=1).view(B, -1))[missing_index==missing_type_index[modal]]

        return self.head(self.norm(inputs))


# Mean Teacher Distillation Method (Fig. 12a) / Representation Distillation Method (Fig. 11)
class modal_distillation(nn.Module):
    def __init__(self, args, output_dims):
        super(modal_distillation, self).__init__()

        self.modality_types = args.modality_types
        self.modal_proj = nn.Sequential(
            nn.Linear(args.feature_dims * len(self.modality_types), args.fusion_dim),
            nn.ReLU(),
            nn.Linear(args.fusion_dim, args.fusion_dim)
        )
        self.norm = nn.LayerNorm(args.fusion_dim)
        self.head = Head(args, args.fusion_dim, output_dims)

    def forward(self, batch, missing_index):
        features = []
        for modal in self.modality_types:
            batch[modal][missing_index == missing_type_index[modal]] = torch.zeros_like(batch[modal][missing_index == missing_type_index[modal]])
            features.append(batch[modal])

        features = torch.cat(features, dim=-1)
        inputs = self.modal_proj(features)

        return features, self.head(self.norm(inputs))


# Self Distillation Method (Fig. 12b)
class modal_self_distillation(nn.Module):
    def __init__(self, args, output_dims):
        super(modal_self_distillation, self).__init__()

        self.modality_types = args.modality_types
        self.modal_proj = nn.Sequential(
            nn.Linear(args.feature_dims * len(self.modality_types), args.fusion_dim),
            nn.ReLU(),
            nn.Linear(args.fusion_dim, args.fusion_dim)
        )
        self.norm = nn.LayerNorm(args.fusion_dim)
        self.head = Head(args, args.fusion_dim, output_dims)

    def forward(self, batch, missing_index):
        if self.training:
            B, C = list(batch.values())[0].shape
            ori_features = []
            stu_features = []
            missing_mask = []
            for i, modal in enumerate(self.modality_types):
                batch[modal][missing_index == missing_type_index[modal]] = torch.zeros_like(batch[modal][missing_index == missing_type_index[modal]])
                ori_features.append(batch[modal])
                stu_features.append(self.modal_proj(torch.cat([torch.zeros((B, i*C), device=batch[modal].device), batch[modal], torch.zeros((B, (len(self.modality_types)-(i+1))*C), device=batch[modal].device)], dim=-1)))
                missing_mask.append(missing_index != missing_type_index[modal])
            
            tea_features = self.modal_proj(torch.cat(ori_features, dim=-1))

            return missing_mask, stu_features, tea_features, self.head(self.norm(tea_features))
        else:
            ori_features = []
            for i, modal in enumerate(self.modality_types):
                batch[modal][missing_index == missing_type_index[modal]] = torch.zeros_like(batch[modal][missing_index == missing_type_index[modal]])
                ori_features.append(batch[modal])
            
            return self.head(self.norm(self.modal_proj(torch.cat(ori_features, dim=-1))))


class finetune_model(nn.Module):
    def __init__(self, args, output_dims, encoder_model):
        super(finetune_model, self).__init__()

        self.encoder = encoder_model
        self.fusion_type = args.fusion_type
        if args.fusion_type == 'sum':
            self.fusion = modal_sum(args, output_dims)
        elif args.fusion_type == 'concat':
            self.fusion = modal_concat(args, output_dims)
        elif args.fusion_type == 'regression':
            self.fusion = modal_regression(args, output_dims)
        elif args.fusion_type == 'retrieval':
            self.fusion = modal_concat_full(args, output_dims)
        elif args.fusion_type == 'intra_attention':
            self.fusion = modal_intra_channel_attention(args, output_dims)
        elif args.fusion_type == 'inter_attention':
            self.fusion = modal_inter_attention(args, output_dims)
        elif args.fusion_type == 'graph_fusion':
            self.fusion = modal_graph_fusion(args, output_dims)
        elif args.fusion_type == 'unified_graph':
            self.fusion = modal_unified_graph(args, output_dims)
        elif args.fusion_type == 'dedicated_dnn':
            self.fusion = modal_dedicated_dnn(args, output_dims)
        elif args.fusion_type in ['Distill_tea', 'MTD_stu', 'KL_stu']:
            self.fusion = modal_distillation(args, output_dims)
        elif args.fusion_type == 'self_distill':
            self.fusion = modal_self_distillation(args, output_dims)

    def forward(self, data, missing_index):
        embedding = self.encoder(data)

        return self.fusion(embedding, missing_index)
