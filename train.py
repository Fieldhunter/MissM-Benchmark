import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.tensorboard import SummaryWriter
from src.dataset.data_loader import data_loader
from src.model.MULT import MultiModal_Sentiment_Analysis
from src.model.baseline import modal_sum, modal_concat_zero_padding, modal_mean_filling, modal_median_filling, modal_knn_filling, modal_regression_filling, modal_attention_fusion, modal_MAE_generation


def parse_args():
    parser = argparse.ArgumentParser()
    # 数据相关参数
    parser.add_argument('--train_mode', type=str, default="classification", help='regression or classification')
    parser.add_argument('--datasetName', type=str, default='eNTERFACE', help='support mosi/sims/eNTERFACE')
    parser.add_argument('--modality_types', type=list, default=['video', 'audio'], help="['language', 'video', 'audio']")
    parser.add_argument('--dataPath', type=str, default='../MLMM_datasets/CH-SIMS/unaligned_39.pkl')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--missing', type=bool, default=True)
    parser.add_argument('--missing_ratio', type=float, default=0.3, help='0.3/0.5/0.7')
    parser.add_argument('--missing_type', type=str, default='mixed', help='language/video/audio/mixed')

    # 模型相关参数
    parser.add_argument('--feature_dims', type=int, default=768, help='the output dims of languagebind')
    parser.add_argument('--fusion_type', type=str, default='concat')
    parser.add_argument('--fusion_dim', type=int, default=256)
    parser.add_argument('--dropout_prob', type=float, default=0.1)

    # 训练相关参数
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--name', type=str, default='test/')
    parser.add_argument('--save_path', type=str, default='checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs')

    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_criterion(args):
    if args.train_mode == 'regression':
        return nn.L1Loss()
    else:
        return nn.CrossEntropyLoss()


def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, label, missing_index in tqdm(dataloader):
            # 处理数据
            for k, v in data.items():
                data[k] = v.to(args.device)
            labels = label['label'].to(args.device)

            # 前向传播
            outputs = model(data, missing_index)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # 处理预测结果
            if args.train_mode == 'regression':
                preds = outputs.detach().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.detach().cpu().numpy())
            else:
                preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.detach().cpu().numpy())

    # 计算指标
    if args.train_mode == 'regression':
        # 对于回归任务，计算MSE
        metrics = {'loss': total_loss / len(dataloader),
                   'mse': np.mean((np.array(all_preds) - np.array(all_labels)) ** 2)}
    else:
        # 对于分类任务，计算准确率和F1分数
        metrics = {
            'loss': total_loss / len(dataloader),
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds, average='weighted')
        }

    return metrics


def train(args):
    # 设置随机种子
    set_seed(args.seed)

    # 创建保存目录
    save_path = './experiments/'+args.name+args.save_path
    log_dir = './experiments/'+args.name+args.log_dir
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 初始化TensorBoard
    writer = SummaryWriter(log_dir=log_dir)

    # 加载数据集
    print("Loading data...")
    train_loader, test_loader, valid_loader, output_dims = data_loader(args.batch_size, args.datasetName, args.missing, args.missing_type, args.missing_ratio)

    # 初始化模型
    print("Initializing model...")
    # model = modal_sum(args, output_dims).to(args.device)
    # model = modal_concat_zero_padding(args, output_dims).to(args.device)
    # model = modal_mean_filling(args, output_dims).to(args.device)
    # model = modal_median_filling(args, output_dims).to(args.device)
    # model = modal_knn_filling(args, output_dims).to(args.device)
    # model = modal_regression_filling(args, output_dims).to(args.device)
    model = modal_attention_fusion(args, output_dims).to(args.device)
    # model = modal_MAE_generation(args, output_dims).to(args.device)

    # 定义损失函数和优化器
    criterion = get_criterion(args)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    # 训练循环
    print("Starting training...")
    best_val_metric = float('inf') if args.train_mode == 'regression' else 0
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
                data[k] = v.to(args.device)
            labels = label['label'].to(args.device)

            # 前向传播
            outputs = model(data, missing_index)
            loss = criterion(outputs, labels)

            # 反向传播
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)

        # 验证模型
        print("Evaluating model...")
        val_metrics = evaluate(model, valid_loader, criterion)

        for k, v in val_metrics.items():
            writer.add_scalar(f'{k}/val', v, epoch)

        # 打印结果
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")

        if args.train_mode == 'classification':
            print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"Val F1 Score: {val_metrics['f1']:.4f}")

        # 学习率调整
        scheduler.step(val_metrics['loss'])

        # 保存最佳模型
        metric_name = 'loss' if args.train_mode == 'regression' else 'accuracy'
        current_metric = val_metrics[metric_name]

        if (args.train_mode == 'regression' and current_metric < best_val_metric) or \
                (args.train_mode == 'classification' and current_metric > best_val_metric):
            print(f"New best model with val {metric_name}: {current_metric:.4f}")
            best_val_metric = current_metric
            best_epoch = epoch
            patience_counter = 0

            # 保存模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'args': args
            }, os.path.join(args.save_path, 'best_model.pth'))
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs.")

        if patience_counter >= args.patience:
            print(f"Early stopping after {epoch + 1} epochs.")
            break

    # 在测试集上进行评估
    print(f"Loading best model from epoch {best_epoch}")
    checkpoint = torch.load(os.path.join(args.save_path, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    print("Evaluating on test set...")
    test_metrics = evaluate(model, test_loader, criterion)

    print("Test Results:")
    print(f"Test Loss: {test_metrics['loss']:.4f}")

    if args.train_mode == 'classification':
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test F1 Score: {test_metrics['f1']:.4f}")

    # 关闭TensorBoard
    writer.close()

    return model, test_metrics


if __name__ == "__main__":
    args = parse_args()
    print(f"Using device: {args.device}")
    
    model, test_metrics = train(args)
    print("Training completed!")
