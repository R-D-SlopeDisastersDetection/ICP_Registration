import copy
import csv
import os
import sys
from datetime import datetime
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import similarity as sim
from PointCloudGridCutting.CloudPointGridCutting import CloudPointGridCutting as Grid


class BlockAnalyze:
    def __init__(self, source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, block_x: int, block_y: int,
                 source_name: str = "Default Source", target_name: str = "Default Target"):
        """
        :param source: 原始点云，target点云会和source做对比并且找出target和source不同的位置
        :param target: 目标点云，指的是疑似发生变化后扫描的点云，或者是日常巡检时的点云，将会用来和source做对比并且找出异常处
        :param block_x: 网格化参数，将点云x轴切分为block_x块
        :param block_y: 网格化参数，将点云的y轴切分为block_y块
        :param source_name: 源点云名称，主要用作结果输出的图表和文件的名称
        :param target_name: 目标点云名称，主要用作结果输出的图表和文件的名称
        """
        self.source = source
        self.target = target
        self.source_name = source_name
        self.target_name = target_name
        self.block_x = block_x
        self.block_y = block_y
        # 结果数组，存储着每一个块的分析结果，其中( , ,0)为距离均值，( , ,1)为距离方差，(, , 2)为RMSE，( , ,3)为重叠率
        self.analyse_result = np.zeros((self.block_x, self.block_y, 4))
        self.color_pcd = o3d.geometry.PointCloud()
        self.exception_pcd = o3d.geometry.PointCloud()
        '''
        一个4*1的矩阵，用于存储所有块的最小评估值。其中第一个为最小均值，第二个为最小标准差，第三个为最小rmse，第四个为最小重叠率
        '''
        self.evaluate_min = np.full((1, 4), sys.float_info.max)
        self.exception_block_arr = []

        # 获取当前时间
        date = datetime.now().strftime("%Y-%m-%d")
        count = 1
        # 结果表格的文件名,由日期+源点云+目标点云名字构成
        filename = date + "_grid_analyse_result_" + source_name + "_" + target_name + "_" + str(count) + ".csv"
        dir_path = "grid_analyse_result/"
        filepath = dir_path + filename
        # 用于避免文件重名：如果文件名存在，则count+1
        while os.path.exists(filepath):
            count = count + 1
            filename = date + "_grid_analyse_result_" + source_name + "_" + target_name + "_" + str(count) + ".csv"
            filepath = dir_path + filename
        self.filepath = filepath

    def update_min_matrix(self, mean, std, rmse, fitness):
        """
        用于更新最小评估矩阵，最小评估矩阵用于记录所有块中的最小评估值，比如所有块中的最小均值，在着色的时候最小值将会用作为起始点
        :return:
        """
        if self.evaluate_min[0][0] > mean:
            self.evaluate_min[0][0] = mean
        if self.evaluate_min[0][1] > std:
            self.evaluate_min[0][1] = std
        if self.evaluate_min[0][2] > rmse:
            self.evaluate_min[0][2] = rmse
        if self.evaluate_min[0][3] > fitness:
            self.evaluate_min[0][3] = fitness

    def draw_color(self, target_block: np.ndarray, empty_array: np.ndarray, threshold: float = 0.1, standard: int = 0):
        """
        将target_block中的点云复制为新点云，并且根据评估值的大小对点云进行着色：其中变化越小的越偏向绿色，变化越大的越偏向红色。
        需要注意的是，我们将读取所有分块中评估值最小的作为起始点，而非将0作为起始点。因为系统的误差普遍在3-5cm内，因此直接将0作为起始点没有
        评估意义，但此法仍有问题：如果一块点云中所有的点均出现了大范围偏移，那么使用评估最小值作为起始点反而导致总体偏绿。
        :param threshold: 阈值，以最小均值距离min为起点，[min, mean+threshold]区间由绿变红，超过mean+threshold一律为红，默认为0.1m
        :param target_block: 目标点云，基于目标点云着色
        :param empty_array: 空块数组，用于指定哪些块为空
        :param standard: 用于规定使用哪种评估标准来进行着色，0为均值，1为标准差，2为RMSE，3为重叠率，目前仅支持均值分析
        :return:
        """
        color_block = target_block.copy()
        color_rgb = [0.0, 0.0, 0.0]
        for j in range(self.block_y):
            for i in range(self.block_x):
                if empty_array[i][j] == 1:
                    # o3d着色的值为0到1，0.1 m为最大值，0.1 米以上的统一着色为(1,0,0)
                    color_rgb[0] = min(
                        (self.analyse_result[i][j][standard] - self.evaluate_min[0, standard]) * (1 / threshold), 1)
                    color_rgb[1] = max(
                        1 - ((self.analyse_result[i][j][standard] - self.evaluate_min[0, standard]) * (1 / threshold)),
                        0)
                    color_block[i][j].paint_uniform_color(color_rgb)
                    self.color_pcd += color_block[i][j]

    def block_empty(self, source_blocks: np.ndarray, threshold):
        """
        测量哪些块为空块，空块有两种情况：1则是最常见的，直接点云数量为0，这种则直接认为是空块；2是在切分过程中切分的边边角角，比如说只
        切到了边缘的一小块，这种情况也可被认为是空块。现在采用最简单直接的方法，即当块的点数少于所有块的均值的threshold时，认为该块时边缘
        块，可被抛弃，一般为5%

        :TODO：
          - 但此法和地面点分离算法共用时，可能会地面点较少植被点偏多的块在滤除了植被后，可能会因为点太少被误判为空块。后续可通过分析疑似
          - 空块内部的点分布规律进一步优化空块判断

        返回一个block_y*block_x的np数组，其中0表示为空块，1表示非空块
        :param source_blocks:
        :param threshold
        :return:
        """
        empty_array = np.zeros((self.block_x, self.block_y))
        avg = 0
        count = 0
        for i in range(self.block_x):
            for j in range(self.block_y):
                if len(source_blocks[i][j].points) != 0:
                    count += 1
                avg += len(source_blocks[i][j].points)
        avg = avg / count  # 均值

        for i in range(self.block_x):
            for j in range(self.block_y):
                if len(source_blocks[i][j].points) < avg * threshold:
                    empty_array[i][j] = 0
                else:
                    empty_array[i][j] = 1
        return empty_array

    def visualize_color_pcd(self, visualize_pcd: o3d.geometry.PointCloud, visualize_target=False,
                            z_axis_dis: float = 1.0):
        """
        可视化方法，默认只会可视化着色后的pcd，可设置同时可视化目标点云
        :param visualize_pcd:
        :param visualize_target: bool, 是否同时可视化目标点云
        :param z_axis_dis: float, 目标点云和着色点云的z轴距离，主要是防止两个点云重叠导致难以观察
        :return:
        """
        visual_list = [visualize_pcd.voxel_down_sample(0.1)]
        if visualize_target:
            visual_target = copy.deepcopy(self.target)
            visualize_pcd.translate((0, 0, z_axis_dis))
            visual_list.append(visual_target)
        o3d.visualization.draw_geometries(visual_list)

    def save_color_blocks(self, z_axis_dis=1.0, voxel_sample=False, voxel_size=0.1):
        """
        将着色后的颜色点云和异常保存在/grid_analyse_result/result_visualize/文件夹下，可选采用体素下采样减少点云数量，使得其更易于观察
        :param z_axis_dis: 在z轴上移动多少米，主要用于和目标点云错开，以防止颜色点云和源点云重叠导致难以观察
        :TODO 当color_block没有点的时候抛出异常
        :param voxel_size: 下采样大小，单位为米
        :param voxel_sample: 是否使用体素下采样
        :return:
        """
        count = 1
        date = datetime.now().strftime("%Y-%m-%d_%H-%M")
        filepath = "grid_analyse_result/result_visualize/"
        filename = date + "_" + self.source_name + "_" + self.target_name + "_" + str(count) + ".ply"
        # 检查点云是否为空
        if self.exception_pcd.points != 0:
            exception_filepath = filepath + "ExceptionPCD_" + filename
            exception_pcd = copy.deepcopy(self.exception_pcd)
            if voxel_sample:
                exception_pcd = exception_pcd.voxel_down_sample(voxel_size)
            exception_pcd.translate([0, 0, z_axis_dis])
            while os.path.exists(filepath + filename):
                count = count + 1
                filename = date + self.source_name + "_" + self.target_name + "_" + str(count) + ".ply"
            o3d.io.write_point_cloud(exception_filepath, exception_pcd)
            print("保存异常点点云，文件名为" + exception_filepath)

        if self.color_pcd.points != 0:
            count = 1
            color_pcd_path = filepath + "ColorPCD_" + filename
            color_pcd = copy.deepcopy(self.color_pcd)
            if voxel_sample:
                color_pcd = color_pcd.voxel_down_sample(voxel_size)
            color_pcd.translate([0, 0, z_axis_dis])
            while os.path.exists(filepath + filename):
                count = count + 1
                filename = date + self.source_name + "_" + self.target_name + "_" + str(count) + ".ply"
            o3d.io.write_point_cloud(color_pcd_path, color_pcd)
            print("保存颜色点点云，文件名为" + color_pcd_path)

    def statistics_analyze(self, eva_matrix: int = 0, threshold: int = 3):
        """
        使用z-score评估异常点，更多异常点检测算法，请查询/BlockAnalyze文件夹下的exception_detect.md文件
        :param threshold:
        :param eva_matrix: 使用何种评价指标：0.Mean, 1.Std, 2.RMSE, 3.Fitness，目前仅支持Mean
        :return:
        """
        data = []
        for i in range(self.block_x):
            data = data + self.analyse_result[i, :, eva_matrix].tolist()

        # 点云直方图，用于展示点云块的统计结果
        plt.hist(data, bins=100, edgecolor='black')
        plt.title(self.target_name + ' Compared with ' + self.source_name)
        plt.xlabel('Mean')
        plt.ylabel('Num')
        plt.show()

        # 统计所有的块的最邻近点对距离均值的均值
        mean = np.nanmean(self.analyse_result[:, :, 0])
        # 统计所有的块的最邻近点对距离标准差的均值
        std = np.nanstd(self.analyse_result[:, :, 0])
        z_scores = (self.analyse_result[:, :, 0] - mean) / std
        exception_blocks = Grid(self.block_y, self.block_x, self.target).grid_cutting()[0]
        for i in range(self.block_x):
            for j in range(self.block_y):
                if z_scores[i, j] > threshold:
                    exception_blocks[i][j].paint_uniform_color([1, 0, 0])
                    # 向异常数组中添加异常块的行号和列号，以及他的变化深度
                    self.exception_block_arr.append([i, j, self.analyse_result[i, j, 0]])
                else:
                    exception_blocks[i][j] = o3d.geometry.PointCloud()
                # 向异常点云中添加指定异常块
                self.exception_pcd += exception_blocks[i][j]
