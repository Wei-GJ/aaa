import numpy as np
import csv
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


class ExoDataset(Dataset):
    """创建pytorch的数据加载器加速训练"""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_line = self.data[idx]
        x = np.array(data_line[:, 0:4], dtype=np.float32)
        y = np.array(data_line[-1, 4:6], dtype=np.float32)
        return x, y

def data_generator_my(path, window_scale=180, stride=10):
    # 归一化参数定义
    hip_angle_max, hip_angle_min = 128.9409, -41.3181      # 髋膝关节角度的最大值和最小值
    knee_angle_max, knee_angle_min = 143.8431, -43.6216
    sequences = []      # 用于存储切分后的子序列

    # 获取所有参与训练的csv数据
    csv_path_list = []                  # 用于存储所有 CSV 文件的路径
    with open(path, 'r') as file:       # 读取包含 CSV 路径的文本文件；打开指定路径的文件，并以只读模式 ('r') 读取内容；with 语句确保文件在使用完毕后自动关闭
        lines = file.readlines()        # 读取文件的所有行，并将其存储在列表 lines 中，每个元素是文件的一行内容（字符串）
        for line in lines:              # 遍历 lines 列表中的每一行
            csv_path_list.append(line.strip())      # 去除每行首尾的空白字符（如换行符 \n）并将处理后的路径添加到列表中
    print("数据集文件数量:%d" % len(csv_path_list))  # 打印 csv_path_list 的长度，即CSV 文件数量

    # 从每个 CSV 文件中提取数据
    with tqdm(total=len(csv_path_list)) as pbar:        # total 参数指定了进度条的总长度（文件数量），使用 tqdm 显示进度条
        for csv_path in csv_path_list:              # 遍历每个 CSV 文件
            data = []                   # 用于存储当前 CSV 文件的数据
            with open(csv_path, 'r') as csv_file:
                csv_reader = csv.DictReader(csv_file)   # 使用 csv.DictReader将 CSV 文件的每一行读取为字典，键为列名，值为对应的数据
                for row in csv_reader:
                    # 提取指定列并转换为浮点数，从每一行中的字典中提取指定的列（髋关节角度、膝关节角度、膝关节力矩），并将其转换为浮点数
                    temp = [float(row['Hip_Angle_Left']), float(row['Hip_Angle_Right']),
                            float(row['Knee_Angle_Left']), float(row['Knee_Angle_Right']),
                            float(row['Knee_Moment_Left']), float(row['Knee_Moment_Right'])]
                    data.append(temp)       # 将每一行的数据添加到 data 列表中
            """
                问题1：数据长度小于窗口长度，当 len(data) < window_scale 时未处理，可能导致数据丢失。
            """
            # 对序列文件使用滑动窗口和步长进行切分
            if len(data) > window_scale:        # 如果数据长度大于窗口大小
                cut_num = (len(data) - window_scale) // stride      # 计算可以切分的子序列数量
                if cut_num == 0:                # 如果数据长度不足以切分多个子序列，# 直接取开头和结尾的部分
                    sequences.append(data[0: window_scale])
                    sequences.append(data[(len(data) - window_scale): len(data)])
                else:                           # 否则使用滑动窗口切分数据
                    for i in range(cut_num + 1):                    # 遍历所有可能的窗口起始位置
                        sequences.append(data[i * stride: i * stride + window_scale])
                    if (len(data) - window_scale) % stride != 0:        # 如果剩余部分不足以覆盖一个完整的窗口，单独处理结尾部分
                        sequences.append(data[(len(data) - window_scale): len(data)])
            pbar.set_description("当前序列长度:%d" % (len(data)))     # 动态更新进度条描述
            pbar.update(1)                      # 更新进度条，每次循环时调用，表示完成了一个文件的处理
    print("滑动窗口所有序列数量:%d" % len(sequences), "滑动窗口单序列长度:%d" % len(sequences[0]), "滑动窗口单序列特征数量:%d" % len(sequences[0][0]))        # 打印切分后的子序列数量、每个子序列的长度、每个子序列中特征的数量

    # 数据归一化
    # sequences 是一个列表，其中每个元素是一个子序列（也是一个列表），每个子序列的形状为 [seq_len, num_features]，其中：seq_len 是子序列的长度，num_features 是每个时间步的特征数
    # sequences 是一个 NumPy 数组，形状为 [num_sequences, seq_len, num_features]
    sequences = np.array(sequences)             # 将列表转换为 NumPy 数组
    sequences[:, :, 0:2] = (sequences[:, :, 0:2] - hip_angle_min) / (hip_angle_max - hip_angle_min)         # 对所有子序列的前两列（髋关节角度）进行归一化，第一个: 表示选择所有子序列；第二个: 表示选择每个子序列的所有时间步；0:2 表示选择每个时间步的前两列（髋关节角度）
    sequences[:, :, 2:4] = (sequences[:, :, 2:4] - knee_angle_min) / (knee_angle_max - knee_angle_min)      # 对所有子序列的第 2-3 列（膝关节角度）进行归一化
    return sequences            # 返回归一化后的子序列


def create_data_loaders(train_path, valid_path, batch_size, window_scale, stride):
    """
    创建数据加载器
    参数：
        train_path: 训练集路径
        valid_path: 验证集路径
        batch_size: 批量大小
        num_workers: 数据加载线程数
    返回：
        train_loader, valid_loader
    """
    # 加载原始数据
    train_data = data_generator_my(train_path, window_scale, stride)
    valid_data = data_generator_my(valid_path, window_scale, stride)

    # 创建数据集
    train_dataset = ExoDataset(train_data)
    valid_dataset = ExoDataset(valid_data)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,  # 验证集使用更大批量
        shuffle=False,
        pin_memory=True
    )

    return train_loader, valid_loader


if __name__ == "__main__":
    sequences = data_generator_my(r"D:\exoskeleton\Code\TCN-test\file_txt\txt_3060\train.txt", window_scale=180, stride=10)  # 注意：Windows路径中的反斜杠 \\ 需转义，建议使用原始字符串或正斜杠（如 r"D:/ExoDatasets/..."）
    train_loader, valid_loader = create_data_loaders(r"D:\exoskeleton\Code\TCN-test\file_txt\txt_3060\train.txt", r"D:\exoskeleton\Code\TCN-test\file_txt\txt_3060\valid.txt", batch_size=64, window_scale=180, stride=10)
