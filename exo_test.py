import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import sys
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append("../../")
from model import TCN
from utils import data_generator_my, create_data_loaders
import numpy as np

"""参数解析器"""
# 创建参数解析器，设置程序描述
parser = argparse.ArgumentParser(description='Exo Modeling - Knee Joint Moment Prediction')
# 添加命令行参数
parser.add_argument('--cuda', action='store_false',                # action='store_false' 表示默认启用 CUDA（即不加 --cuda 时使用 GPU）
                    help='use CUDA (default: True)')                            # 是否使用 CUDA（默认启用）
parser.add_argument('--gpu_id', type=int, default=0,
                    help='ID of the GPU to use (default: 0)')                   # GPU 的 ID
parser.add_argument('--dropout', type=float, default=0.25,
                    help='dropout applied to layers (default: 0.25)')           # Dropout 率
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: 0.2)')      # 梯度裁剪阈值，-1 表示不裁剪
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit (default: 100)')                    # 训练的总轮数
parser.add_argument('--ksize', type=int, default=4,
                    help='kernel size (default: 5)')                            # 卷积核大小
parser.add_argument('--levels', type=int, default=5,
                    help='# of levels (default: 4)')                            # TCN 的层数
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')                       # 日志打印间隔
parser.add_argument('--lr', type=float, default=1e-5,
                    help='initial learning rate (default: 1e-3)')               # 初始学习率
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')                    # 优化器类型
parser.add_argument('--nhid', type=int, default=80,
                    help='number of hidden units per layer (default: 150)')     # 每层隐藏单元数
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')                         # 随机种子
parser.add_argument('--batch_size', type=int, default=64,
                    help='input batch size for training')                       # 训练集批次输入大小
parser.add_argument('--train_path', type=str, default=r"D:\exoskeleton\Code\TCN-test\file_txt\txt2\Complete\all\txt_3060\train.txt",
                    help='path to train file')                                  # 训练集路径
parser.add_argument('--val_path', type=str, default=r"D:\exoskeleton\Code\TCN-test\file_txt\txt2\Complete\all\txt_3060\valid.txt",
                    help='path to val file')                                    # 验证集路径
parser.add_argument('--window_scale', type=int, default=180,
                    help='window size (default: 150)')                          # 滑动窗口大小
parser.add_argument('--stride', type=int, default=10,
                    help='stride (default: 10)')                                # 步长

args = parser.parse_args()      # 解析命令行参数并存储到 args 对象中


"""随机种子与CUDA设置"""
# Set the random seed manually for reproducibility. 设置随机种子以确保实验的可复现性
torch.manual_seed(args.seed)        # 固定随机种子

# GPU设备设置
device = torch.device(f'cuda:{args.gpu_id}' if args.cuda else 'cpu')
print(f"\n>> Using device: {device}")
print(f">> Available GPUs: {torch.cuda.device_count()}")
if args.cuda:
    print(f">> Selected GPU: {torch.cuda.get_device_name(args.gpu_id)}\n")

print("解析参数args...")
print(args)     # 打印解析后的参数


"""数据加载与模型初始化"""
input_size = 4      # 输入特征数（4个关节角度）
output_size = 2     # 输出维度（左右膝关节力矩）

# 加载数据集
print("加载数据集Loading datasets...")
train_loader, valid_loader = create_data_loaders(
    train_path=args.train_path,
    valid_path=args.val_path,
    batch_size=args.batch_size,
    window_scale=args.window_scale,
    stride=args.stride,
)

# 模型定义
n_channels = [args.nhid] * args.levels      # TCN每层的通道数
kernel_size = args.ksize                    # 卷积核大小
dropout = args.dropout                      # Dropout率
model = TCN(input_size, output_size, n_channels, kernel_size, dropout)     # 初始化TCN模型
model.to(device)        # 如果启用 CUDA，将模型移至 GPU

"""损失函数与优化器"""
criterion = nn.MSELoss()        # 均方误差损失函数
# criterion = nn.HuberLoss(delta=1.0)  # 替换MSE为Huber Loss
lr = args.lr                    # 学习率
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)        # 根据参数动态选择优化器（如 Adam）
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)         # 动态降低学习率

"""训练函数"""
def train(epoch):
    model.train()           # 将模型设置为训练模式
    total_loss = 0.0        # 初始化训练总损失
    with tqdm(train_loader, desc=f'Epoch {epoch}', unit='batch') as pbar:       # 使用 tqdm 显示进度条
        for x, y in pbar:   # 遍历训练数据
            x, y = x.to(device), y.to(device)       # 如果启用 CUDA，将数据移至 GPU

            optimizer.zero_grad()             # 清零梯度
            output = model(x)
            loss = criterion(output, y) * 1000
            total_loss += loss.item()               # 累加损失和计数

            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)       # 如果启用梯度裁剪，裁剪梯度

            loss.backward()         # 反向传播
            optimizer.step()        # 更新模型参数
            pbar.set_description("Epoch {:3d} | lr {:.6f} | loss {:.6f}".format(epoch, lr, loss.item()))        # 更新进度条描述
            pbar.update(1)          # 更新进度条

        avg_loss = total_loss / len(train_loader)       # 计算平均损失
        print(" Train loss: {:.6f}".format(avg_loss))    # 打印训练损失
        train_loss_history.append(avg_loss)             # 保存训练损失
        return avg_loss


"""评估函数"""
def evaluate():
    model.eval()            # 将模型设置为评估模式
    total_loss = 0.0        # 初始化总损失
    with torch.no_grad():   # 禁用梯度计算
        for x, y in valid_loader:       # 遍历训练数据
            x, y = x.to(device), y.to(device)       # 如果启用 CUDA，将数据移至 GPU
            output = model(x)
            loss = criterion(output, y) * 1000
            total_loss += loss.item()               # 累加损失和计数

        avg_loss = total_loss / len(valid_loader)       # 计算平均损失
        print(" Validation loss: {:.6f}".format(avg_loss))      # 打印评估损失
        valid_loss_history.append(avg_loss)             # 保存评估损失

    return avg_loss


def plot_loss_curves(train_loss_history, valid_loss_history, file_number, save_folder='loss_curves'):
    # 绘制损失曲线
    # 确保保存路径存在
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    filename = f'loss_curves_{file_number}.png'
    save_path = os.path.join(save_folder, filename)

    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_history, label='Training Loss')
    plt.plot(valid_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True)

    # 保存图像
    plt.savefig(save_path)
    plt.close()

    print(f"Loss curve saved to {save_path}")


if __name__ == "__main__":
    best_vloss = 1e8            # 初始化最佳验证损失
    model_name = "weight_180_10_hid80.pt"     # 设置模型保存路径
    file_number = 1        # 保存的图片编号

    # 新增损失记录功能
    train_loss_history = []
    valid_loss_history = []

    for epoch in range(1, args.epochs + 1):     # 开始训练
        train_loss = train(epoch)       # 训练一个 epoch
        valid_loss = evaluate()         # 在验证集上评估模型
        scheduler.step(valid_loss)      # 学习率动态调整

        # 保存最佳模型
        if valid_loss < best_vloss:     # 如果当前验证损失小于最佳验证损失，保存模型
            with open(model_name, "wb") as f:
                # torch.save(model, f)
                torch.save(model.state_dict(), model_name)
                print("Saved model!\n")

            best_vloss = valid_loss

        if epoch > 10 and valid_loss > max(valid_loss_history[-3:]):        # 如果连续 3 次验证损失未下降，学习率衰减
            lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # 每5个epoch保存一次损失曲线
        if epoch % 5 == 0:
            plot_loss_curves(train_loss_history, valid_loss_history, file_number)

    # 最终保存损失曲线
    plot_loss_curves(train_loss_history, valid_loss_history, file_number)

    print('-' * 89)
    model.load_state_dict(torch.load(model_name))
