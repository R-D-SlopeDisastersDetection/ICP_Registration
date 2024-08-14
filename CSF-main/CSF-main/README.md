# CSF
该项目适用于CSF地面点滤除算法，目前支持las文件及ply文件</br>

参数包括以下
~~~
        :param inputfile: input file path
        :param outputfile:  output file path
        :param filetype:    file type, "las" or "ply"
        :param bSloopSmooth: 是否进行边坡后处理。当有陡变地形是设置为ture。默认为False
        :param cloth_resolution: 布料网格分辨率，一般与点云间距相当，单位为m，默认为0.5
        :param rigidness:   布料刚性参数，可选值1，2，3. 1表示平坦地形。2表示有缓坡的地形。3表示有较陡的地形（比如山地）。默认为3
        :param time_step:   密度峰值之间的时间步长，默认为0.65
        :param class_threshold: 点云与布料模拟点的距离阈值，默认为0.03
        :param interations: 最大迭代次数，默认为500
~~~

参数配有相对应的set和get方法，支持实时更新到csf中，并配有相对应的输出文件可视化函数。</br>
示例如下：
~~~
from CSF2 import CSF2

if __name__ == '__main__':
    csf = CSF2(inputfile='Mesh[shanzhan_k533_groundpoint.ply',outputfile='test.ply',filetype='ply') #类实例化
    csf.process()           #进行滤波算法处理
    csf.view_cloud()        #输出文件的可视化
~~~