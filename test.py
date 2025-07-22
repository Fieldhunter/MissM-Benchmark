import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.tensorboard import SummaryWriter
from src.dataset.data_loader import testing_loader
from src.model.baseline import finetune_model
from languagebind import LanguageBind, transform_dict, to_device, LanguageBindImageTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    # 数据集相关参数
    parser.add_argument('--train_mode', type=str, default="classification", help='regression or classification')
    parser.add_argument('--datasetName', type=str, default='eNTERFACE', help='support mosi/sims/eNTERFACE')
    parser.add_argument('--csv_path', type=str, default='/big-data/person/yuanjiang/MLMM_datasets/eNTERFACE/label.csv', help='')
    parser.add_argument('--modality_types', type=list, default=['video', 'audio'], help="['language', 'video', 'audio']") # 严格遵守help顺序

    # 缺失相关参数
    parser.add_argument('--test_missing_type', type=list, default=['video', 'audio', 'mixed'], help='language/video/audio/mixed')
    
    # 模型相关参数
    parser.add_argument('--model_ckpt_dir', type=str, default='./final_model', help='the ckpt of models')
    parser.add_argument('--feature_dims', type=int, default=768, help='the output dims of languagebind')
    parser.add_argument('--fusion_type', type=str, default='self_distill', help='sum/concat/regression/retrieval/intra_attention/inter_attention/graph_fusion/unified_graph/dedicated_dnn/MTD_stu/KL_stu/self_distill')
    parser.add_argument('--test_types', type=list, default=['self_distill'], help='sum/[concat_zero/concat_median/concat_mean]/regression/retrieval/intra_attention/inter_attention/graph_fusion/unified_graph/dedicated_dnn/MTD_stu/KL_stu/self_distill')
    parser.add_argument('--fusion_dim', type=int, default=256)
    parser.add_argument('--dropout_prob', type=float, default=0.1)

    # 其他相关参数
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_criterion():
    return nn.CrossEntropyLoss()


def calculate_statistics(data_list, type):
    """计算列表数据的均值和中位数"""
    data_array = np.concatenate(data_list, axis=0)

    if type == 'mean':
        return np.mean(data_array, axis=0).tolist()
    else:
        return np.median(data_array, axis=0).tolist()


def test(args):
    # 设置随机种子
    set_seed(args.seed)

    # 创建保存目录
    result_txt_file = './new_txt_experiment'
    os.makedirs(result_txt_file, exist_ok=True)

    # 初始化Encoder
    clip_type = {}
    tokenizer = None
    if 'language' in args.modality_types:
        tokenizer = LanguageBindImageTokenizer.from_pretrained(f'lb203/LanguageBind_Image',
                                                                cache_dir='./cache_dir/tokenizer_cache_dir')
    if 'video' in args.modality_types:
        clip_type['video'] = 'LanguageBind_Video'
    if 'audio' in args.modality_types:
        clip_type['audio'] = 'LanguageBind_Audio'
    encoder_model = LanguageBind(clip_type=clip_type, cache_dir='./cache_dir')
    modality_transform = {c: transform_dict[c](encoder_model.modality_config[c]) for c in clip_type.keys()}

    # 加载数据集
    print("Loading data...")
    train_loader, test_loader, output_dims = testing_loader(args, args.csv_path, tokenizer, modality_transform)

    # 初始化模型
    print("Initializing model...")
    model = finetune_model(args, output_dims, encoder_model)
    model.load_state_dict(torch.load(os.path.join(args.model_ckpt_dir, f'{args.datasetName}_{args.fusion_type}.pth'), map_location='cpu')['model_state_dict'])
    model = model.to(args.device)
    model.eval()
    criterion = get_criterion().to(args.device)

    for test_type in args.test_types:
        if test_type == 'concat_median' or test_type == 'concat_mean':
            train_embeddings = {modal: [] for modal in args.modality_types}
            with torch.no_grad():
                for data, _, _ in train_loader:
                    for k, v in data.items():
                        for i, j in v.items():
                            v[i] = j.squeeze(1)
                        data[k] = to_device(v, args.device)

                    embedding = model.encoder(data)
                    for modal in args.modality_types:
                        train_embeddings[modal].append(embedding[modal].detach().cpu().numpy())
            
            if test_type == 'concat_median':
                statistics = {modal: calculate_statistics(train_embeddings[modal], 'median') for modal in args.modality_types}
            else:
                statistics = {modal: calculate_statistics(train_embeddings[modal], 'mean') for modal in args.modality_types}
            model.fusion.set_statistics(statistics, args.modality_types)
        
        # 在测试集上进行评估
        print("Evaluating on test set...")
        for test_missing_type in args.test_missing_type:
            test_name = f'{args.datasetName}_{test_type}_{test_missing_type}'
            with open(f'{result_txt_file}/{test_name}.txt', "w", encoding="utf-8") as fout:
                for i in test_loader[test_missing_type].keys():
                    print(f"Testing with missing ratio: {i}")
                    
                    total_loss = 0.0
                    ori_preds = []
                    all_preds = []
                    all_labels = []

                    with torch.no_grad():
                        for data, label, missing_index in tqdm(test_loader[test_missing_type][i]):
                            # 处理数据
                            for k, v in data.items():
                                for l, j in v.items():
                                    v[l] = j.squeeze(1)
                                data[k] = to_device(v, args.device)
                            labels = label['label'].to(args.device)
                            missing_index = missing_index.to(args.device)

                            # 前向传播
                            if args.fusion_type in ['MTD_stu', 'KL_stu']:
                                _, outputs = model(data, missing_index)
                            else:
                                outputs = model(data, missing_index)
                            loss = criterion(outputs, labels)
                            total_loss += loss.item()

                            # 处理预测结果
                            preds = torch.argmax(outputs, dim=1)
                            step_ori_preds = torch.nn.functional.softmax(outputs, dim=-1)

                            ori_preds.extend(step_ori_preds.detach().cpu().numpy())
                            all_preds.extend(preds.detach().cpu().numpy())
                            all_labels.extend(labels.detach().cpu().numpy())

                    # 计算指标
                    test_metrics = {
                        'loss': total_loss / len(test_loader),
                        'accuracy': accuracy_score(all_labels, all_preds),
                        'f1': f1_score(all_labels, all_preds, average='macro'),
                        'auc': roc_auc_score(all_labels, ori_preds, multi_class='ovo')
                    }

                    fout.write(f"Testing with missing ratio: {i}\n")
                    fout.write("Test Results:\n")
                    fout.write(f"Test Loss: {test_metrics['loss']:.4f}\n")

                    fout.write(f"Test Accuracy: {test_metrics['accuracy']:.4f}\n")
                    fout.write(f"Test F1 Score: {test_metrics['f1']:.4f}\n")
                    fout.write(f"Test AUC: {test_metrics['auc']:.4f}\n")
                    fout.write("\n")

    return model, test_metrics


if __name__ == "__main__":
    args = parse_args()

    model, test_metrics = test(args)
    print("Test completed!")
