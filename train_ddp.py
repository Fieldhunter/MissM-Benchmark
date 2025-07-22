import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.tensorboard import SummaryWriter
from src.dataset.data_loader import training_loader
from src.model.baseline import finetune_model
from languagebind import LanguageBind, transform_dict, to_device, LanguageBindImageTokenizer
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser()
    # 数据集相关参数
    parser.add_argument('--train_mode', type=str, default="classification", help='regression or classification')
    parser.add_argument('--datasetName', type=str, default='mvsa', help='support mosi/sims/eNTERFACE/AVE/mvsa')
    parser.add_argument('--csv_path', type=str, default='/big-data/person/yuanjiang/MLMM_datasets/mvsa_multiple/label.csv', help='')
    parser.add_argument('--modality_types', type=list, default=['language', 'image'], help="['language', 'video', 'audio', 'image']/") # 严格遵守help顺序

    # 缺失相关参数
    parser.add_argument('--train_missing', type=bool, default=False)
    
    # 模型相关参数
    parser.add_argument('--feature_dims', type=int, default=768, help='the output dims of languagebind')
    parser.add_argument('--fusion_type', type=str, default='sum', help='sum/concat/regression/retrieval/intra_attention/inter_attention/graph_fusion/unified_graph/dedicated_dnn/[Distill_tea/MTD_stu/KL_stu]/self_distill')
    parser.add_argument('--fusion_dim', type=int, default=256)
    parser.add_argument('--dropout_prob', type=float, default=0.1)

    # 训练相关参数
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_path', type=str, default='checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs')

    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt


def gather_tensor(tensor, n):
    rt = [torch.zeros_like(tensor) for _ in range(n)]
    dist.all_gather(rt, tensor)
    return torch.cat(rt, dim=0)


class KL_loss(nn.Module):
    def __init__(self, temperature=0.15):
        super(KL_loss, self).__init__()
        self.temperature = temperature

    def forward(self, g_s, g_t):
        student_distance = F.log_softmax(g_s / self.temperature, dim=1)
        teacher_distance = F.softmax(g_t.detach() / self.temperature, dim=1)
        loss_distill_dis = F.kl_div(student_distance, teacher_distance, reduction='batchmean')
        return loss_distill_dis


def get_criterion(args):
    if args.fusion_type == 'MTD_stu':
        return nn.MSELoss(), nn.CrossEntropyLoss()
    elif args.fusion_type == 'KL_stu' or args.fusion_type == 'self_distill':
        return KL_loss(), nn.CrossEntropyLoss()
    else:
        return nn.CrossEntropyLoss()


def evaluate(model, dataloader, criterion, world_size, device):
    model.eval()
    total_loss = 0.0
    ori_preds = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, label, missing_index in tqdm(dataloader):
            # 处理数据
            for k, v in data.items():
                for i, j in v.items():
                    v[i] = j.squeeze(1)
                data[k] = to_device(v, device)
            labels = label['label'].to(device)
            missing_index = missing_index.to(device)

            # 前向传播
            if args.fusion_type in ['Distill_tea', 'MTD_stu', 'KL_stu']:
                _, outputs = model(data, missing_index)
            else:
                outputs = model(data, missing_index)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # 处理预测结果
            preds = torch.argmax(outputs, dim=1)
            step_ori_preds = torch.nn.functional.softmax(outputs, dim=-1)
            preds = gather_tensor(preds, world_size)
            step_ori_preds = gather_tensor(step_ori_preds, world_size)
            labels = gather_tensor(labels, world_size)

            ori_preds.extend(step_ori_preds.detach().cpu().numpy())
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

    # 计算指标
    metrics = {
        'loss': total_loss / len(dataloader),
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds, average='macro'),
        'auc': roc_auc_score(all_labels, ori_preds, multi_class='ovo')
    }

    return metrics


def train(args):
    # 设置随机种子
    set_seed(args.seed)

    local_rank = int(os.environ['LOCAL_RANK'])
    device = 'cuda:{}'.format(local_rank)
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    # 创建保存目录
    experiment_name = f'{args.datasetName}_{args.fusion_type}'
    save_path = './experiments/' + experiment_name + '/' + args.save_path
    log_dir = './experiments/' + experiment_name + '/' + args.log_dir
    final_model_path = './final_model'

    if local_rank == 0:
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(final_model_path, exist_ok=True)

        # 初始化TensorBoard
        writer = SummaryWriter(log_dir=log_dir)
    
    if args.fusion_type in ['regression', 'unified_graph', 'dedicated_dnn', 'MTD_stu', 'KL_stu', 'self_distill']:
        args.train_missing = True

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
    if 'image' in args.modality_types:
        clip_type['image'] = 'LanguageBind_Image'
    encoder_model = LanguageBind(clip_type=clip_type, cache_dir='./cache_dir')
    modality_transform = {c: transform_dict[c](encoder_model.modality_config[c]) for c in clip_type.keys()}

    # 加载数据集
    print("Loading data...")
    train_loader, valid_loader, output_dims = training_loader(args, args.csv_path, tokenizer, modality_transform)

    # 初始化模型
    print("Initializing model...")
    model = finetune_model(args, output_dims, encoder_model).to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank], broadcast_buffers=True, find_unused_parameters=False) # find_unused_parameters=True

    if args.fusion_type == 'MTD_stu' or args.fusion_type == 'KL_stu':
        tea_model = finetune_model(args, output_dims, encoder_model)
        tea_model.load_state_dict(torch.load(f'./final_model/{args.datasetName}_Distill_tea.pth', map_location='cpu')['model_state_dict'])
        tea_model.eval()
        tea_model = tea_model.to(device)
        tea_model = DDP(tea_model, device_ids=[local_rank], broadcast_buffers=True, find_unused_parameters=True) # find_unused_parameters=True

    # 定义损失函数和优化器
    if args.fusion_type in ['MTD_stu', 'KL_stu', 'self_distill']:
        distill_loss, criterion = get_criterion(args)
        distill_loss = distill_loss.to(device)
        criterion = criterion.to(device)
    else:
        criterion = get_criterion(args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)

    # 训练循环
    if local_rank == 0:
        print("Starting training...")
    best_val_metric = 0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.num_epochs}')

        for data, label, missing_index in progress_bar:
            optimizer.zero_grad()

            # 处理数据
            for k, v in data.items():
                for i, j in v.items():
                    v[i] = j.squeeze(1)
                data[k] = to_device(v, device)
            labels = label['label'].to(device)
            missing_index = missing_index.to(device)

            # 前向传播
            if args.fusion_type == 'MTD_stu' or args.fusion_type == 'KL_stu':
                with torch.no_grad():
                    rep_t, _ = tea_model(data, torch.zeros_like(missing_index).to(device))
                rep_s, outputs = model(data, missing_index)
                loss = distill_loss(rep_s, rep_t) + criterion(outputs, labels)
            elif args.fusion_type == 'self_distill':
                missing_mask, stu_features, tea_features, outputs = model(data, missing_index)
                dl = 0
                for i, mask in enumerate(missing_mask):
                    t = tea_features[mask]
                    s = stu_features[i][mask]
                    dl += distill_loss(s, t)
                loss = 0.01 * dl / len(missing_mask) + criterion(outputs, labels)
            else:
                if args.fusion_type == 'Distill_tea':
                    _, outputs = model(data, missing_index)
                else:
                    outputs = model(data, missing_index)
                loss = criterion(outputs, labels)

            # 反向传播
            loss.backward()
            optimizer.step()

            if args.fusion_type == 'MTD_stu':
                with torch.no_grad():
                    for param_tea, param_stu in zip(tea_model.parameters(), model.parameters()):
                        param_tea.data = param_tea.data * 0.999 + param_stu.data * (1. - 0.999)

            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss / len(train_loader)
        if local_rank == 0:
            writer.add_scalar('Loss/train', avg_train_loss, epoch)

        if local_rank == 0:
            # 验证模型
            print("Evaluating model...")
        val_metrics = evaluate(model, valid_loader, criterion, world_size, device)

        if local_rank == 0:
            for k, v in val_metrics.items():
                writer.add_scalar(f'{k}/val', v, epoch)

            # 打印结果
            print(f"Epoch {epoch + 1}/{args.num_epochs}")
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"Val F1 Score: {val_metrics['f1']:.4f}")
            print(f"Val AUC: {val_metrics['auc']:.4f}")

        # 学习率调整
        scheduler.step(val_metrics['accuracy'])

        # 保存最佳模型
        metric_name = 'accuracy'
        current_metric = val_metrics[metric_name]

        if current_metric > best_val_metric:
            print(f"New best model with val {metric_name}: {current_metric:.4f}")
            best_val_metric = current_metric
            best_epoch = epoch
            patience_counter = 0

            if local_rank == 0:
                # 保存模型
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_metrics': val_metrics,
                    'args': args
                }, os.path.join(save_path, 'best_model.pth'))
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs.")

        if patience_counter >= args.patience:
            print(f"Early stopping after {epoch + 1} epochs.")
            break
    
    # 恢复最佳模型
    print(f"Loading best model from epoch {best_epoch}")
    model.load_state_dict(torch.load(os.path.join(save_path, 'best_model.pth'), map_location=device)['model_state_dict'])
    model.eval()
        
    if local_rank == 0:
        torch.save({
                    'model_state_dict': model.module.state_dict()
                }, os.path.join(final_model_path, f'{args.datasetName}_{args.fusion_type}.pth'))
    
    if local_rank == 0:
        # 关闭TensorBoard
        writer.close()

    return model


if __name__ == "__main__":
    args = parse_args()

    model = train(args)
    print("Training completed!")
