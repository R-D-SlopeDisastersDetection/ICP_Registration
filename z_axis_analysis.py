import plyfile
import numpy as np
import matplotlib.pyplot as plt

def z_axis_analysis(path):
    # 读取PLY文件
    ply_data = plyfile.PlyData.read(path)

    # 提取顶点数据
    vertices = ply_data['vertex']

    # 获取Z轴坐标
    z_coordinates = vertices['z']
    # 进行统计
    mean_z = np.mean(z_coordinates)
    std_dev_z = np.std(z_coordinates)
    max_z = np.max(z_coordinates)
    min_z = np.min(z_coordinates)

    print(f"均值: {mean_z}")
    print(f"标准差: {std_dev_z}")
    print(f"最大值: {max_z}")
    print(f"最小值: {min_z}")
    # print(z_coordinates)

    # 绘制直方图
    plt.figure(figsize=(8, 6))
    n, bins, patches = plt.hist(z_coordinates, bins=30, edgecolor='black', alpha=0.7)
    for count, bin_edge in zip(n, bins):
        # 计算柱顶的高度
        height = count
        # 柱的中心位置
        x = (bin_edge + bins[bins.tolist().index(bin_edge) + 1]) / 2
        # 在柱顶显示数量
        plt.text(x, height, str(int(height)), ha='center', va='bottom')
    plt.title('')
    plt.xlabel('height')
    plt.ylabel('quantity')
    plt.grid(True)
    plt.show()

path = "071_072/072_box1top.ply"
z_axis_analysis(path)