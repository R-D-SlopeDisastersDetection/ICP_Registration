import string

import laspy
import numpy as np
import CSF
import open3d as o3d


class CSF2(object):
    def __init__(self, inputfile: string, outputfile: string, filetype: string,
                 bSloopSmooth: bool = False, cloth_resolution: float = 0.5, rigidness: float = 3,
                 time_step: float = 0.65, class_threshold: float = 0.03, interations: int = 500):
        """
        init for this class
        :param inputfile: input file path
        :param outputfile:  output file path
        :param filetype:    file type
        :param bSloopSmooth: 是否进行边坡后处理。当有陡变地形是设置为ture
        :param cloth_resolution: 布料网格分辨率，一般与点云间距相当，单位为m
        :param rigidness:   布料刚性参数，可选值1，2，3. 1表示平坦地形。2表示有缓坡的地形。3表示有较陡的地形（比如山地）。
        :param time_step:   密度峰值之间的时间步长
        :param class_threshold: 点云与布料模拟点的距离阈值
        :param interations: 最大迭代次数
        """
        self.csf = CSF.CSF()
        self.inputfile = inputfile
        self.outputfile = outputfile
        self.filetype = filetype
        self.bSloopSmooth = bSloopSmooth
        self.cloth_resolution = cloth_resolution
        self.rigidness = rigidness
        self.time_step = time_step
        self.class_threshold = class_threshold
        self.interations = interations
        self.csf.params.bSloopSmooth = self.bSloopSmooth
        self.csf.params.cloth_resolution = self.cloth_resolution
        self.csf.params.rigidness = self.rigidness
        self.csf.params.time_step = self.time_step
        self.csf.params.class_threshold = self.class_threshold
        self.csf.params.interations = self.interations
        self.ground = CSF.VecInt()
        self.non_ground = CSF.VecInt()
        
    def process(self):
        """
        数据处理
        :return:
        """
        if self.filetype == 'las':
            self.las_process()
        if self.filetype == 'ply':
            self.ply_process()

    def view_cloud(self):
        """
        可视化
        :return:
        """
        if self.filetype == 'las':
            self.las_view_cloud()

        if self.filetype == 'ply':
            self.ply_view_cloud()

    def las_process(self):
        """
        las格式数据处理
        :return:
        """
        infile = laspy.read(self.inputfile)
        points = infile.points
        xyz = np.vstack((infile.x, infile.y, infile.z)).transpose()

        self.csf.setPointCloud(xyz)
        self.csf.do_filtering(self.ground, self.non_ground)

        outfile = laspy.LasData(infile.header)
        outfile.points = points[np.array(self.ground)]
        outfile.write(self.outputfile)

    def ply_process(self):
        infile = o3d.io.read_point_cloud(self.inputfile)
        xyz = np.array(infile.points)
        colors = np.array(infile.colors)
        normals = np.array(infile.normals)

        self.csf.setPointCloud(xyz)
        self.csf.do_filtering(self.ground, self.non_ground)

        outfile = o3d.geometry.PointCloud()
        tmp = np.array(self.ground)
        outfile.points = o3d.utility.Vector3dVector(xyz[tmp])
        outfile.colors = o3d.utility.Vector3dVector(colors[tmp])
        outfile.normals = o3d.utility.Vector3dVector(normals[tmp])

        o3d.io.write_point_cloud(self.outputfile, outfile)

    def ply_view_cloud(self):
        """
        ply格式数据可视化
        :return:
        """
        infile = o3d.io.read_point_cloud(self.outputfile)
        o3d.visualization.draw_geometries([infile])

    def las_view_cloud(self):
        """
        las格式数据可视化
        :return:
        """
        las = laspy.read(self.outputfile)
        points = np.stack([las.x, las.y, las.z]).transpose()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        if las.red.max() >= 256:
            las.red >>= 8
        if las.green.max() >= 256:
            las.green >>= 8
        if las.blue.max() >= 256:
            las.blue >>= 8
        colors = np.stack([las.red * 256 / 65535, las.green * 256 / 65535, las.blue * 256 / 65535], axis=0).transpose(
            (1, 0))
        pcd.colors = o3d.utility.Vector3dVector(colors)

        o3d.visualization.draw_geometries([pcd])

    def set_inputfile(self, inputfile: str):

        self.inputfile = inputfile

    def get_inputfile(self):
        return self.inputfile

    def set_outputfile(self, outputfile: str):
        self.outputfile = outputfile

    def get_outputfile(self):
        return self.outputfile

    def set_filetype(self, filetype):
        self.filetype = filetype

    def get_filetype(self):
        return self.filetype

    def set_bSloopSmooth(self, bSloopSmooth: bool):
        self.bSloopSmooth = bSloopSmooth
        self.csf.params.bSloopSmooth = self.bSloopSmooth

    def get_bSloopSmooth(self):
        return self.bSloopSmooth

    def set_cloth_resolution(self, cloth_resolution: float):
        self.cloth_resolution = cloth_resolution
        self.csf.params.cloth_resolution = self.cloth_resolution

    def get_cloth_resolution(self):
        return self.cloth_resolution

    def set_rigidness(self, rigidness: float):
        self.rigidness = rigidness
        self.csf.params.rigidness = self.rigidness

    def get_rigidness(self):
        return self.rigidness

    def set_time_step(self, time_step: float):
        self.time_step = time_step
        self.csf.params.time_step = self.time_step

    def get_time_step(self):
        return self.time_step

    def set_class_threshold(self, class_threshold: float):
        self.class_threshold = class_threshold
        self.csf.params.class_threshold = self.class_threshold

    def get_class_threshold(self):
        return self.class_threshold

    def set_interations(self, interations: int):
        self.interations = interations
        self.csf.params.interations = self.interations

    def get_interations(self):
        return self.interations
