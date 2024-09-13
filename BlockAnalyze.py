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
        self.source = source
        self.target = target
        self.source_name = source_name
        self.target_name = target_name
        self.block_x = block_x
        self.block_y = block_y
        self.analyse_result = np.zeros((self.block_x, self.block_y, 4))
        self.color_pcd = o3d.geometry.PointCloud()
        self.exception_pcd = o3d.geometry.PointCloud()
        '''
        一个4*1的矩阵，用于存储所有块的最小评估值。其中第一个为最小均值，第二个为最小标准差，第三个为最小rmse，第四个为最小重叠率
        '''
        self.evaluate_min = np.full((1, 4), sys.float_info.max)

        date = datetime.now().strftime("%Y-%m-%d")
        count = 1
        filename = date + " grid_analyse_result_" + str(count) + ".csv"
        dir_path = "grid_analyse_result/"
        filepath = dir_path + filename
        while os.path.exists(filepath):
            count = count + 1
            filename = date + " grid_analyse_result_" + str(count) + ".csv"
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

    def draw_color(self, target_block: np.ndarray, empty_array: np.ndarray, standard: int = 0):
        """
        将target_block中的点云复制为新点云，并且根据评估值的大小对点云进行着色：其中变化越小的越偏向绿色，变化越大的越偏向红色。
        需要注意的是，我们将读取所有分块中评估值最小的作为起始点，而非将0作为起始点。因为系统的误差普遍在3-5cm内，因此直接将0作为起始点没有
        评估意义，但此法仍有问题：如果一块点云中所有的点均出现了大范围偏移，那么使用评估最小值作为起始点反而导致总体偏绿。
        :param target_block:
        :param empty_array:
        :param standard: 用于规定使用哪种评估标准来进行着色，0为均值，1为标准差，2为RMSE，3为重叠率
        :return:
        """
        color_block = target_block.copy()
        color_rgb = [0.0, 0.0, 0.0]
        for j in range(self.block_y):
            for i in range(self.block_x):
                if empty_array[i][j] == 1:
                    # o3d着色的值为0到1，0.1 m为最大值，0.1 米以上的统一着色为(1,0,0)
                    color_rgb[0] = min(
                        (self.analyse_result[i][j][standard] - self.evaluate_min[0, standard]) * (1 / 0.1), 1)
                    color_rgb[1] = max(
                        1 - ((self.analyse_result[i][j][standard] - self.evaluate_min[0, standard]) * (1 / 0.1)), 0)
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

    def visualize_color_pcd(self, visualize_pcd: o3d.geometry.PointCloud, visualize_target=False, z_axis_dis: float = 1.0):
        """
        可视化方法，默认只会可视化着色后的pcd，可设置同时可视化目标点云
        :param visualize_pcd:
        :param visualize_target: bool, 是否同时可视化目标点云
        :param z_axis_dis: float, 目标点云和着色点云的z轴距离，主要是防止两个点云重叠导致难以观察
        :return:
        """
        visual_list = [visualize_pcd]
        if visualize_target:
            visual_target = copy.deepcopy(self.target)
            visual_target.translate((0, 0, z_axis_dis))
            visual_list.append(visual_target)
        o3d.visualization.draw_geometries(visual_list)

    def save_color_blocks(self, voxel_sample=False, voxel_size=0.05):
        """
        :TODO 当color_block没有点的时候抛出异常
        :param voxel_size: 下采样大小，单位为米
        :param voxel_sample: 是否使用体素下采样
        :return:
        """
        if self.color_pcd.points == 0:
            print("颜色点云中没有点，请先对点云进行着色！")
        elif voxel_sample:
            color_down_pcd = self.color_pcd.voxel_down_sample(voxel_size)
            o3d.io.write_point_cloud("grid_analyse_result/result_visualize/color_down_pcd.ply", color_down_pcd)
        else:
            o3d.io.write_point_cloud("grid_analyse_result/result_visualize/color_pcd.ply", self.color_pcd)

    def statistics_analyze(self, eva_matrix: int = 0, threshold: int = 4):
        """
        使用z-score评估异常点，更多异常点检测算法，请查询根目录下的exception_detect.md文件
        :param threshold:
        :param eva_matrix: 使用何种评价指标：0.Mean, 1.Std, 2.RMSE, 3.Fitness
        :return:
        """
        data = []
        for i in range(self.block_x):
            data = data + self.analyse_result[i, :, eva_matrix].tolist()

        plt.hist(data, bins=100, edgecolor='black')
        plt.title(self.target_name + ' Compared with ' + self.source_name)
        plt.xlabel('Mean')
        plt.ylabel('Num')
        plt.show()

        mean = np.nanmean(self.analyse_result[:, :, 0])
        std = np.nanstd(self.analyse_result[:, :, 0])
        z_scores = (self.analyse_result[:, :, 0] - mean) / std
        exception_blocks = Grid(self.block_y, self.block_x, self.target).grid_cutting()
        for i in range(self.block_x):
            for j in range(self.block_y):
                if z_scores[i, j] > threshold:
                    exception_blocks[i][j].paint_uniform_color([1, 0, 0])
                else:
                    exception_blocks[i][j] = o3d.geometry.PointCloud()
                self.exception_pcd += exception_blocks[i][j]


class BlockAnalyzeByChange(BlockAnalyze):
    def __init__(self, source: o3d.geometry.PointCloud, baseline: o3d.geometry.PointCloud,
                 target: o3d.geometry.PointCloud, block_x: int, block_y: int,
                 source_name: str = "Default Source", target_name: str = "Default Target"):
        super().__init__(source, target, block_x, block_y, source_name, target_name)
        self.baseline = baseline
        self.baseline_result = np.zeros((block_x, block_y, 4))
        self.target_result = np.zeros((block_x, block_y, 4))

    def block_analyze(self, decimal_places: int = 5):
        """
        分块分析：获取以二维数组存储的点云分块，原始网格和目标网格的对应块进行分析，若块非空，则将分析结果放入analyse_result矩阵中。
        该函数会将源点云划分为block_x行block_y列的二维数组，数组元素为点云。可视作为标准二维坐标轴顺时针旋转90°的所得到的网格
        :param decimal_places:
        :return:
        """
        source_grid = Grid(self.block_y, self.block_x, self.source)
        target_grid = Grid(self.block_y, self.block_x, self.target)
        baseline_grid = Grid(self.block_y, self.block_x, self.baseline)
        source_blocks = source_grid.grid_cutting()
        target_blocks = target_grid.grid_cutting()
        baseline_block = baseline_grid.grid_cutting()

        for i in range(self.block_x):
            for j in range(self.block_y):
                mean_base, std_base = sim.point2point_mean_and_std_deviation(source_blocks[i][j], baseline_block[i][j])
                mean_target, std_target = sim.point2point_mean_and_std_deviation(source_blocks[i][j],
                                                                                 target_blocks[i][j])

                self.baseline_result[i][j][0], self.baseline_result[i][j][1], self.target_result[i][j][0], \
                    self.target_result[i][j][1] = round(mean_base, decimal_places), round(std_base, decimal_places), \
                    round(mean_target, decimal_places), round(std_target, decimal_places)

                self.analyse_result[i][j][0], self.analyse_result[i][j][1] = \
                    round(mean_target - mean_base, decimal_places), round(std_target - std_base, decimal_places)

        self.draw_color(target_blocks, np.ones((self.block_x, self.block_y)))

    def save_result_as_csv(self):
        file = open(self.filepath, "w+")
        writer = csv.writer(file)
        abstract = [['Grid Analyse Result\n'],
                    ['Source Cloud Points: ' + self.source_name + '\n',
                     'Target Cloud Points: ' + self.target_name + '\n']]
        writer.writerows(abstract)
        # performance_matrix = ["Mean", "Std Deviation", "RMSE", "Fitness"]
        # item = ["baseline", "target", "change"]
        writer.writerows([[], [], ["Mean: Change Between BaseLine And Target"]])
        writer.writerows(self.analyse_result[:, :, 0].tolist())
        writer.writerows([[], [], ["Mean: Baseline"]])
        writer.writerows(self.baseline_result[:, :, 0].tolist())
        writer.writerows([[], [], ["Mean: Target"]])
        writer.writerows(self.target_result[:, :, 0].tolist())
        file.close()

    def draw_color(self, target_block: np.ndarray, empty_array: np.ndarray, standard: int = 0):
        """
        将target_block中的点云复制为新点云，并且根据评估值的大小对点云进行着色：其中变化越小的越偏向绿色，变化越大的越偏向红色。
        由于是基于历史记录横向对比，因此当target隆起时会导致值为正，target凹陷时，值为负；因此在评估时需要取绝对值
        :param target_block:
        :param empty_array:
        :param standard: 用于规定使用哪种评估标准来进行着色，0为均值，1为标准差，2为RMSE，3为重叠率
        :return:
        """
        color_block = target_block.copy()
        color_rgb = [0.0, 0.0, 0.0]
        for j in range(self.block_y):
            for i in range(self.block_x):
                if empty_array[i][j] == 1:
                    # o3d着色的值为0到1，0.1 m为最大值，0.1 米以上的统一着色为(1,0,0)
                    color_rgb[0] = min(abs(self.analyse_result[i][j][standard]) * (1 / 0.1), 1)
                    color_rgb[1] = max(1 - (abs(self.analyse_result[i][j][standard]) * (1 / 0.1)), 0)
                    color_block[i][j].paint_uniform_color(color_rgb)
                    self.color_pcd += color_block[i][j]


class BlockAnalyzeByTerra(BlockAnalyze):
    """
    网格化分析类，用于对网格化后的区块进行分析
    TODO: 改进方向：使用历史数据横向对比进行着色，而非直接使用均值着色
    """

    def __init__(self, source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, block_x: int, block_y: int,
                 source_name: str = "Default Source", target_name: str = "Default Target"):
        super().__init__(source, target, block_x, block_y, source_name, target_name)
        self.color_pcd = o3d.geometry.PointCloud()

    def block_analyze(self, drop_threshold: float = 0.05, decimal_places: int = 5):
        """
        分块分析：获取以二维数组存储的点云分块，原始网格和目标网格的对应块进行分析，若块非空，则将分析结果放入analyse_result矩阵中。
        该函数会将源点云划分为block_x行block_y列的二维数组，数组元素为点云。可视作为标准二维坐标轴顺时针旋转90°的所得到的网格
        :param decimal_places: 保留几位小数
        :param drop_threshold: 用于规定点数低于所有区块平均点数的多少的块会被视为空块丢弃，不做处理
        :return:
        """
        source_grid = Grid(self.block_y, self.block_x, self.source)
        target_grid = Grid(self.block_y, self.block_x, self.target)
        source_blocks = source_grid.grid_cutting()
        target_blocks = target_grid.grid_cutting()
        empty_array = self.block_empty(source_blocks, drop_threshold)

        for i in range(self.block_x):
            for j in range(self.block_y):
                if empty_array[i][j] == 1:
                    mean, std = sim.point2point_mean_and_std_deviation(source_blocks[i][j], target_blocks[i][j])
                    res = o3d.pipelines.registration.evaluate_registration(source_blocks[i][j], target_blocks[i][j],
                                                                           0.05, np.identity(4))
                    self.analyse_result[i, j, 0], self.analyse_result[i, j, 1], self.analyse_result[i, j, 2], \
                        self.analyse_result[i, j, 3] = round(mean, decimal_places), round(std, decimal_places), \
                        round(res.inlier_rmse, decimal_places), round(res.fitness, decimal_places)
                    self.update_min_matrix(mean, std, res.inlier_rmse, res.fitness)
        self.draw_color(target_blocks, empty_array)

    def save_result_as_csv(self):
        file = open(self.filepath, "w+")
        writer = csv.writer(file)
        abstract = [['Grid Analyse Result\n'],
                    ['Source Cloud Points: ' + self.source_name + '\n',
                     'Target Cloud Points: ' + self.target_name + '\n']]
        performance_matrix = ["Mean", "Std Deviation", "RMSE", "Fitness"]
        writer.writerows(abstract)
        for i in range(0, 4):
            writer.writerows([[], [performance_matrix[i]]])
            writer.writerows(self.analyse_result[:, :, i].tolist())
        file.close()
