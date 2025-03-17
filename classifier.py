import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

class ModalFusion(nn.Module):
    """多模态融合模块"""

    def __init__(self, fusion_type, modal_dims, fusion_dim):
        super(ModalFusion, self).__init__()
        self.fusion_type = fusion_type
        self.modal_dims = modal_dims

        # 为每个模态创建投影层，将各模态特征映射到相同维度
        self.projections = nn.ModuleDict()
        for modal_name, dim in modal_dims.items():
            self.projections[modal_name] = nn.Linear(dim, fusion_dim)



        if fusion_type == 'concat':
            self.fusion_layer = nn.Linear(fusion_dim * len(modal_dims), fusion_dim)
        elif fusion_type == 'attention':
            self.attention = nn.MultiheadAttention(fusion_dim, num_heads=8, batch_first=True)
            self.norm = nn.LayerNorm(fusion_dim)

    def forward(self, modal_features):
        """
        多模态特征融合
        Args:
            modal_features: 字典，键为模态名称，值为对应模态的特征
        """
        projected_features = {}
        for modal_name, features in modal_features.items():
            if modal_name in self.projections:
                projected_features[modal_name] = self.projections[modal_name](features)

        if self.fusion_type == 'concat':
            # 拼接融合
            concat_features = torch.cat(list(projected_features.values()), dim=1)
            return self.fusion_layer(concat_features)

        elif self.fusion_type == 'attention':
            # 处理每个模态以确保维度一致
            modal_features = []
            for modal_name, features in projected_features.items():
                if features.dim() == 2:
                    # 为2D张量添加序列维度
                    features = features.unsqueeze(1)  # [batch, 1, dim]
                modal_features.append(features)

            # 沿序列维度连接
            features_seq = torch.cat(modal_features, dim=1)  # [batch, seq_len, dim]

            # 跨模态的自注意力
            attn_output, _ = self.attention(features_seq, features_seq, features_seq)

            # 平均池化或只取第一个token
            fused = torch.mean(attn_output, dim=1)  # [batch, dim]
            return self.norm(fused)

        elif self.fusion_type == 'mean':
            # 平均融合
            return torch.mean(torch.stack(list(projected_features.values())), dim=0)

        elif self.fusion_type == 'max':
            # 最大值融合
            return torch.max(torch.stack(list(projected_features.values())), dim=0)[0]

        else:
            raise ValueError(f"不支持的融合类型: {self.fusion_type}")


class MultiModalClassifier(nn.Module):
    """多模态分类模型"""

    def __init__(self, config, device):
        super(MultiModalClassifier, self).__init__()
        self.config = config
        self.device = device
        self.modality_types = config.get('modality_types', ['text', 'vision', 'audio'])

        # 特征提取器
        self.feature_extractors = nn.ModuleDict()
        modal_dims = {}

        # 文本特征提取器
        if 'text' in self.modality_types:
            text_model_name = config.get('text_model', 'bert-base-chinese')
            self.text_config = AutoConfig.from_pretrained(text_model_name)
            self.text_model = AutoModel.from_pretrained(text_model_name)
            modal_dims['text'] = self.text_config.hidden_size
            self.feature_extractors['text'] = self.text_model

        # 视觉特征提取器
        # if 'vision' in self.modality_types:
        #     vision_model_name = config.get('vision_model', 'resnet50')
        #     if vision_model_name == 'resnet50':
        #         from torchvision.models import resnet50, ResNet50_Weights
        #         self.vision_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        #         # 移除最后的分类层
        #         self.vision_model = nn.Sequential(*list(self.vision_model.children())[:-1])
        #         modal_dims['vision'] = 2048  # ResNet50的特征维度
        #     self.feature_extractors['vision'] = self.vision_model

        self.orig_d_v = 1
        self.orig_d_a = 1

        conv1d_kernel_size_a = 3
        conv1d_kernel_size_v = 3
        self.d_l = self.d_a = self.d_v = 30  #args.hidden_dim
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=conv1d_kernel_size_v, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=conv1d_kernel_size_a, padding=0, bias=False)

        modal_dims['audio'] = 31
        modal_dims['vidio'] = 707
        # 音频特征提取器
        # if 'audio' in self.modality_types:
        #     audio_model_name = config.get('audio_model', 'wav2vec2-base')
        #     if audio_model_name == 'wav2vec2-base':
        #         from transformers import Wav2Vec2Model
        #         self.audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        #         modal_dims['audio'] = self.audio_model.config.hidden_size
        #     self.feature_extractors['audio'] = self.audio_model

        # 多模态融合
        self.fusion_dim = config.get('fusion_dim', 768)
        self.fusion_type = config.get('fusion_type', 'attention')
        self.fusion = ModalFusion(self.fusion_type, modal_dims, self.fusion_dim)

        # 分类头
        self.num_labels = config.get('num_labels', 1)
        self.dropout = nn.Dropout(config.get('dropout_prob', 0.1))
        # self.classifier = nn.Linear(self.fusion_dim, self.num_labels)
        # 为每种情感创建一个分类器
        self.classifiers = nn.ModuleDict({
            'A': nn.Linear(self.fusion_dim, 1),
            'M': nn.Linear(self.fusion_dim, 1),
            'T': nn.Linear(self.fusion_dim, 1),
            'V': nn.Linear(self.fusion_dim, 1)
        })


    def extract_features(self, batch):
        """从不同模态提取特征"""
        features = {}

        # 提取文本特征
        if 'text' in self.modality_types and 'text' in batch:
            text = batch['text']
            input_ids = text[:, 0, :].long()  # 第1维作为 input_ids
            attention_mask = text[:, 1, :].long()  # 第2维作为 attention_mask
            token_type_ids = text[:, 2, :].long()  # 第3维作为 token_type_ids

            # 构造 BERT 的输入
            bert_inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids
            }

            text_outputs = self.text_model(**bert_inputs)
            # 使用[CLS]标记的输出作为文本特征
            features['text'] = text_outputs.last_hidden_state[:, 0]

        # 提取视觉特征
        if 'vision' in self.modality_types and 'vision' in batch:
            # vision_features = self.vision_model(batch['vision'])
            # 调整维度
            vision = batch['vision']
            x_v = vision
            proj_x_v = self.proj_v(x_v)
            # proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
            # proj_x_v = proj_x_v.permute(2, 0, 1)

            features['vision'] = proj_x_v

        # 提取音频特征
        if 'audio' in self.modality_types and 'audio' in batch:
            audio = batch['audio']
            # audio_outputs = self.audio_model(batch['audio'])
            x_a = audio
            proj_x_a = self.proj_a(x_a) if hasattr(self, 'proj_a') and x_a.size(1) != self.d_a else x_a
            # proj_x_a = proj_x_a.permute(2, 0, 1)
            # 使用音频序列的平均值作为特征
            features['audio'] = proj_x_a

        return features

    def forward(self, batch):
        """模型前向传播"""
        # 提取多模态特征
        modal_features = self.extract_features(batch)

        # 多模态融合
        fused_features = self.fusion(modal_features)

        # 为每个标签创建一个独立的分类器
        results = {}
        for emotion in ['A', 'M', 'T', 'V']:  # 假设这些是您的标签
            results[emotion] = self.classifiers[emotion](fused_features)

        return results  # 返回字典格式的结果


        # # 分类
        # logits = self.classifier(self.dropout(fused_features))
        #
        # # 如果是训练模式，计算损失
        # if 'labels' in batch:
        #     # 处理多标签情况 - 根据实际情况调整
        #     labels = next(iter(batch['labels'].values()))  # 如果只有一种标签
        #     # 或者处理多种标签的情况
        #     # labels = {k: v for k, v in batch['labels'].items()}
        #
        #     loss_fn = nn.CrossEntropyLoss()  # 根据任务调整损失函数
        #     loss = loss_fn(logits, labels)
        #     return {'loss': loss, 'logits': logits}

        return logits


class EarlyFusionModel(nn.Module):
    """简单的早期融合模型"""

    def __init__(self, config, device):
        super(EarlyFusionModel, self).__init__()
        self.config = config
        self.device = device

        # 模态和维度设置
        self.text_dim = config.get('text_dim', 768)
        self.vision_dim = config.get('vision_dim', 2048)
        self.audio_dim = config.get('audio_dim', 768)
        self.fusion_dim = config.get('fusion_dim', 512)
        self.num_labels = config.get('num_labels', 1)

        # 模态投影层
        self.text_projection = nn.Linear(self.text_dim, self.fusion_dim)
        self.vision_projection = nn.Linear(self.vision_dim, self.fusion_dim)
        self.audio_projection = nn.Linear(self.audio_dim, self.fusion_dim)

        # 融合后的分类层
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim * 3, self.fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.fusion_dim, self.num_labels)
        )

    def forward(self, batch):
        """模型前向传播"""
        text_features = self.text_projection(batch.get('text_features'))
        vision_features = self.vision_projection(batch.get('vision_features'))
        audio_features = self.audio_projection(batch.get('audio_features'))

        # 拼接特征
        combined = torch.cat([text_features, vision_features, audio_features], dim=1)

        # 分类
        logits = self.classifier(combined)

        return logits


def get_modal(config, device):
    """根据配置获取对应的多模态模型

    Args:
        config: 模型配置
        device: 设备类型

    Returns:
        初始化好的模型实例
    """
    model_name = config.get('name', 'multimodal')

    if model_name == 'multimodal':
        return MultiModalClassifier(config, device)
    elif model_name == 'early_fusion':
        return EarlyFusionModel(config, device)
    else:
        raise ValueError(f"不支持的模型类型: {model_name}")
