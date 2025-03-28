import csv
import math
import os
import os.path as osp
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
from model import TCN
import matplotlib
matplotlib.use('Agg')  # 解决Tkinter问题的关键配置
import matplotlib.pyplot as plt
from tqdm import tqdm
hip_angle_max, hip_angle_min = 128.9409, -41.3181  # 髋膝关节角度的最大值和最小值
knee_angle_max, knee_angle_min = 143.8431, -43.6216
import torch
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)
window_scale = 180


def read_squeeze_from_csv(csv_path, stride=1):
    sequences = []          # 用于存储切分后的子序列
    data = []               # 用于存储原始数据
    with open(csv_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)  # 使用 csv.DictReader将 CSV 文件的每一行读取为字典，键为列名，值为对应的数据
        for row in csv_reader:
            # 提取指定列并转换为浮点数，从每一行中的字典中提取指定的列（髋关节角度、膝关节角度、膝关节力矩），并将其转换为浮点数
            temp = [float(row['Hip_Angle_Left']), float(row['Hip_Angle_Right']),
                    float(row['Knee_Angle_Left']), float(row['Knee_Angle_Right']),
                    float(row['Knee_Moment_Left']), float(row['Knee_Moment_Right'])]
            data.append(temp)  # 将每一行的数据添加到 data 列表中
    #print(len(data))
    """
        问题1：数据长度小于窗口长度，当 len(data) < window_scale 时未处理，可能导致数据丢失。
    """
    # 对序列文件使用滑动窗口和步长进行切分
    if len(data) > window_scale:  # 如果数据长度大于窗口大小
        cut_num = (len(data) - window_scale) // stride  # 计算可以切分的子序列数量
        if cut_num == 0:  # 如果数据长度不足以切分多个子序列，# 直接取开头和结尾的部分
            sequences.append(data[0: window_scale])
            sequences.append(data[(len(data) - window_scale): len(data)])
        else:  # 否则使用滑动窗口切分数据
            for i in range(cut_num + 1):  # 遍历所有可能的窗口起始位置
                sequences.append(data[i * stride: i * stride + window_scale])
            if (len(data) - window_scale) % stride != 0:  # 如果剩余部分不足以覆盖一个完整的窗口，单独处理结尾部分
                print('验证步长1')
                sequences.append(data[(len(data) - window_scale): len(data)])

    # 数据归一化
    # sequences 是一个列表，其中每个元素是一个子序列（也是一个列表），每个子序列的形状为 [seq_len, num_features]，其中：seq_len 是子序列的长度，num_features 是每个时间步的特征数
    # sequences 是一个 NumPy 数组，形状为 [num_sequences, seq_len, num_features]
    sequences = np.array(sequences)             # 将列表转换为 NumPy 数组
    sequences[:, :, 0:2] = (sequences[:, :, 0:2] - hip_angle_min) / (hip_angle_max - hip_angle_min)         # 对所有子序列的前两列（髋关节角度）进行归一化，第一个: 表示选择所有子序列；第二个: 表示选择每个子序列的所有时间步；0:2 表示选择每个时间步的前两列（髋关节角度）
    sequences[:, :, 2:4] = (sequences[:, :, 2:4] - knee_angle_min) / (knee_angle_max - knee_angle_min)      # 对所有子序列的第 2-3 列（膝关节角度）进行归一化
    #print(len(sequences))
    return data, sequences      # 返回原始数据和切分后的子序列


if __name__ == "__main__":
    vis = False                 # 是否可视化预测结果
    path = r"D:\exoskeleton\Code\TCN-test\file_txt\txt2\Complete\all\txt_3060\valid.txt"       # 包含所有验证集 CSV 文件路径的文本文件
    model_name = r"D:\exoskeleton\Code\TCN-test\exo_gait\weight_240_10_hid80.pt"                                # 预训练模型文件路径
    model = TCN(4, 2, [80] * 5, 5, 0.15).to(device)        # 初始化 TCN 模型并移至 GPU
    # model.load_state_dict(torch.load(model_name))              # 从 model_name 指定的路径加载预训练模型
    model.load_state_dict(torch.load(model_name, map_location=device))
    model.eval()                # 将模型设置为评估模式
    # X_valid = data_generator_my("D:\\ExoDatasets\\Phase3\\Select\\val.txt", stride=300)

    # 获取所有参与训练的csv数据
    csv_path_list = []          # 用于存储所有 CSV 文件路径
    # 整个验证集上的误差平均值
    rmse_r_total, rmse_l_total = 0.0, 0.0       # 用于累加左右膝关节力矩的均方根误差
    time_total = 0.0            # 用于累加模型预测的总用时
    count_total = 0             # 用于累加总样本数
    count_r, count_l = 0, 0     # 用于累加左右膝关节力矩的样本数

    with open(path, 'r') as file:       # 读取包含 CSV 路径的文本文件；打开指定路径的文件，并以只读模式 ('r') 读取内容；with 语句确保文件在使用完毕后自动关闭
        lines = file.readlines()        # 读取文件的所有行，并将其存储在列表 lines 中，每个元素是文件的一行内容（字符串）
        for line in lines:              # 遍历 lines 列表中的每一行
            csv_path_list.append(line.strip())      # 使用 strip()去除每行首尾的空白字符（如换行符 \n）并将处理后的路径添加到列表中
    print(len(csv_path_list))           # 打印 CSV 文件的数量

    # 从每个 CSV 文件中提取数据
    with tqdm(total=len(csv_path_list)) as pbar:        # total 参数指定了进度条的总长度（文件数量），使用 tqdm 显示处理 CSV 文件的进度条
        for csv_path in csv_path_list:                  # 遍历每个 CSV 文件路径
            try:            # 捕获可能发生的异常（如文件读取错误），避免程序崩溃
                data, X_valid = read_squeeze_from_csv(csv_path)     # 读取 CSV 文件并切分数据，返回原始数据 data 和切分后的子序列 X_valid
                moment_pred_l = []                  # 用于存储左膝关节力矩的预测值
                moment_pred_r = []                  # 用于存储右膝关节力矩的预测值
                with torch.no_grad():               # 禁用梯度计算
                    for idx in range(len(X_valid)):         # 遍历每个子序列数据
                        data_line = X_valid[idx]            # 获取当前子序列数据
                        t1 = time.time()                    # 记录开始时间
                        x, y = torch.Tensor(data_line[:, 0:4]).float(), torch.Tensor(data_line[:, 4:6]).float()     # 提取所有行的前4列输入特征和最后2列目标值，并将特征转为PyTorch 张量和将其数据类型转换为 float32
                        x, y = x.cuda(), y.cuda()           # 将数据移至 GPU

                        # 模型预测
                        output = model(x.unsqueeze(0)).squeeze(0)       # 增加 batch 维度后输入模型，并计算输出；在 x 的第 0 维度增加一个维度，使其形状变为 [1, seq_len, input_size]；将调整形状后的 x 输入模型，得到形状为 [1, seq_len, output_size] 的输出；移除输出的第 0 维度，得到形状为 [seq_len, output_size] 的最终输出
                        output = output.cpu()               # 将模型输出移至 CPU
                        output = output.detach().numpy()    # 将输出转换为 NumPy 数组
                        t2 = time.time()                    # 记录结束时间
                        # print(output)
                        moment_pred_l.append(output[0])     # 存储左膝关节力矩的预测值
                        moment_pred_r.append(output[1])     # 存储右膝关节力矩的预测值
                        #print(t2 - t1)
                        time_total = time_total + (t2 - t1)     # 累加模型预测的用时，用于计算总用时和平均用时
                count_total += len(X_valid)                     # 累加处理的样本数，用于计算平均用时
                # 由于滑动窗口的偏移，预测值的长度比原始数据短 window_scale - 1，在预测值前面填充 window_scale - 1 个 0，使其长度与原始数据一致
                moment_pred_l = [0] * (window_scale - 1) + moment_pred_l        # 对左膝关节力矩的预测值进行填充（滑动窗口的偏移）
                moment_pred_r = [0] * (window_scale - 1) + moment_pred_r        # 对右膝关节力矩的预测值进行填充
                moment_true_l = np.array(data)[:, 4].tolist()       # 提取左膝关节力矩的真实值
                moment_true_r = np.array(data)[:, 5].tolist()       # 提取右膝关节力矩的真实值
                #print(len(moment_pred_l))

                # 结果可视化（添加内存释放）
                if vis:
                    fig_stride = 2000       # 每张图显示的时间步数（数据点数量）
                    fig = plt.figure(figsize=(16, 9))       # 创建一个大小为 16x9 的画布，用于绘制图表
                    fig_num = len(moment_pred_l) // fig_stride + 1      # 根据总时间步数和每张图的时间步数，计算需要绘制的图的数量
                    for i in range(fig_num):        # 遍历每张图，绘制预测值和真实值
                        plt.subplot(fig_num, 1, i + 1)      # 创建子图，排列方式为 fig_num 行 1 列
                        start_idx, end_idx = fig_stride * i, fig_stride * i + fig_stride        # 计算起始和结束索引
                        if i == fig_num - 1:
                            end_idx = len(moment_pred_l)        # 最后一张图的结束索引
                        # 绘制预测值和真实值
                        plt.plot(moment_pred_l[start_idx: end_idx], color='purple')
                        plt.plot(moment_true_l[start_idx: end_idx], linestyle='--', color='purple')
                        plt.plot(moment_pred_r[start_idx: end_idx], color='pink')
                        plt.plot(moment_true_r[start_idx: end_idx], linestyle='--', color='pink')
                        plt.grid(True)      # 显示网格
                    #plt.show()
                    plt.savefig(csv_path.replace(".csv", "_result.png"), dpi=240)       # 保存图像，将图表保存为 PNG 文件，分辨率为 240 DPI
                    plt.close(fig)  # 确保图形资源释放

                # 计算均方根误差
                rmse_r, rmse_l = 0.0, 0.0
                for i in range(len(moment_pred_l)):
                    if i >= window_scale - 1:       # 忽略滑动窗口填充部分（前 window_scale - 1 个数据点）
                        rmse_l += (moment_pred_l[i] - moment_true_l[i]) ** 2        # 累加左右膝关节力矩的误差
                        rmse_r += (moment_pred_r[i] - moment_true_r[i]) ** 2
                rmse_l_total = rmse_l_total + rmse_l        # 将当前文件的误差累加到左右膝关节力矩总误差中
                rmse_r_total = rmse_r_total + rmse_r
                rmse_l = math.sqrt(rmse_l / len(moment_pred_l))     # 计算当前文件的左右膝关节力矩均方根误差
                rmse_r = math.sqrt(rmse_r / len(moment_pred_l))
                #print(rmse_r, rmse_l)
                count_l = count_l + len(moment_pred_l)      # 累加左右膝关节力矩的样本数，用于计算总误差
                count_r = count_r + len(moment_pred_r)
                with open(csv_path.replace(".csv", "_" + str(format(rmse_l, '.3f')) + "_" + str(format(rmse_r, '.3f')) + ".txt"), 'w') as file:
                    file.write(str(rmse_l) + ' ' + str(rmse_r))     # 将当前文件的左右膝关节力矩的均方根误差写入文本文件
            except:
                print(f"数据长度小于滑动窗口大小: {csv_path}")     # 打印出错的 CSV 文件路径
            finally:
                pbar.update(1)          # 更新进度条
                plt.close('all')  # 强制释放所有图形资源

    # 计算总均方根误差
    rmse_l_total = math.sqrt(rmse_l_total / count_l)        # 计算整个验证集的左右膝关节力矩的总均方根误差
    rmse_r_total = math.sqrt(rmse_r_total / count_r)
    print("总误差:", rmse_l_total, rmse_r_total)       # 打印左右膝关节力矩的总误差
    print("总用时:", time_total)                       # 打印模型预测的总用时
    print("单个序列平均用时:", time_total / count_total)    # 打印单个序列的平均用时
