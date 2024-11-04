# 点云校准以及网格化技术

## 点云配准类ICP_Reg

## 多轮次配准测试类BatchRun

## 网格化分析BlockAnalyze类
网格化分析属于地形地势变化算法的一部分，在进行此步骤之前需要满足以下前置条件：
* 保证两个点云的大小（长和宽）以及位置（经纬度）一致
* 使用CSF地面点过滤算法将植被滤除，否则植被的变化会被当作为地形地貌变化从而导致假阳性
* 保证两个点云的位置偏差在5cm以内，如果不是，可通过点云配准实现

![img.png](resource/img.png)


网格化分析的基本思路是将两个大小和位置一样的点云source和target，在x轴上切分为block_x块，y轴上切分为block_y块，设`source_ij`为source点云的
第i行第j列的点云。对比`source_ij`和`target_ij`的最邻近点对均值、标准差和RMSE，用于衡量两个块的变化程度。其中最邻近点对值越小，变化越小。需要注
意的是我们的十字坐标系是由标准坐标系顺时针旋转90°而来的

![img.png](resource/img1.png)

也就是block_x是行数，block_y是列数，该功能主要由`BlockAnalyzeByTerra`实现，示例代码如下：
```python
import open3d
import numpy as np
import math
from BlockAnalyze.BlockAnalyzeByTerra import BlockAnalyzeByTerra

pcd1 = open3d.io.read_point_cloud("dataset_reg/K533/CSF_Test/118_Disaster1.ply")
pcd2 = open3d.io.read_point_cloud("dataset_reg/K533/CSF_Test/119_Disaster1.ply")

# 将两个点云分成长x_size和宽y_size的块，返回x轴分块数和y轴分块数
x_size = 0.5
y_size = 0.5

xyz = np.array(pcd1.points)
colors = np.array(pcd1.colors)

x_min = xyz[:, 0].min()
x_max = xyz[:, 0].max()
y_min = xyz[:, 1].min()
y_max = xyz[:, 1].max()

x_blocks_num = math.ceil((x_max - x_min) / x_size)
y_blocks_num = math.ceil((y_max - y_min) / y_size)

print("Grid Cutting With X_Blocks ", x_blocks_num, ", Y_Blocks ", y_blocks_num)
block_analyze = BlockAnalyzeByTerra(pcd1, pcd2, x_blocks_num, y_blocks_num) # 类初始化
# 启动地形分析
block_analyze.block_analyze()   
# 是否展示分析结果
block_analyze.visualize_color_pcd(True)
# 是否将结果保存为csv文件
block_analyze.save_result_as_csv()
# 是否保存根据结果着色的点云
block_analyze.save_color_blocks(True, 0.2)
```
<br>

使用基本的基于地形变化的网格化分析有如下的缺点。假设点云密度为`x points/m^2`，对于平整的柏油路点云块A和崎岖的边坡点云块B，A的最邻近点对
距离dist_a会小于B的最邻近点对距离dist_b，主要原因是在点云数量相同的情况下，平整的柏油路点云块A上的点云基本位于同一平面上，而崎岖的边坡
点云块B的点云则会在点云块B的拟合平面上进行上下波动，因为其表面凹凸不平。这主要是因为对于同样大小的点云块，点云块A的物体表面积小于点云块B
的物体表面积导致的，即使两块的点云数量一致，点云B的dist_b也会比点云块A的dist_a大，这导致单纯使用最邻近点对的分析方法无法统一变量。还有一个例子就是
，植被茂盛的地方经过植被过滤之后，可能只会留下20%甚至更少的点云，这种情况下的最邻近点对平均距离肯定是更远的，这会导致植被茂密的地方mean偏高。

一个基本的方法是，在刚开始时对边坡进行两次扫描：第一次是用作基础点云source，第二次是被称为baseline的基准点云，首先将Source和Baseline进行对比，
这样我们可以获得边坡的基本特征Cmp1，由于source和baseline是短时间内的两次扫描，可以基本认为边坡是没变化的，那么Cmp1中每个块最邻近点对的mean值或者std值
可以视作为它的“底噪”，也就是上面提到的由于植被过滤或者边坡崎岖导致的mean值或者std值增大。

![img.png](resource/img2.png)

接下来就是传统的，将刚开始扫描的source点云和后面需要时扫描的target点云做对比，获得的变化特征Cmp2，我们将Cmp2的结果减去Cmp1的“底噪”，就可以知道
真实的地表变化程度，这是滤除由地形崎岖和植被滤除导致的假阳性效果较好的方法。

该功能由`BlockAnalyzeByChange`类实现，示例代码如下
```python
import open3d as o3d
import math
import numpy as np
from BlockAnalyze.BlockAnanlyzeByChange import BlockAnalyzeByChange

# 需要三个点云：source \baseline \target
pcd1 = o3d.io.read_point_cloud("dataset_reg/K533/CSF_Test/118_Disaster1.ply")
pcd2 = o3d.io.read_point_cloud("dataset_reg/K533/CSF_Test/119_Disaster1.ply")
pcd3 = o3d.io.read_point_cloud("dataset_reg/K533/CSF_Test/135_Disaster1.ply")

# 将两个点云分成长x_size和宽y_size的块，返回x轴分块数和y轴分块数
x_size = 0.5
y_size = 0.5

xyz = np.array(pcd1.points)
colors = np.array(pcd1.colors)

x_min = xyz[:, 0].min()
x_max = xyz[:, 0].max()
y_min = xyz[:, 1].min()
y_max = xyz[:, 1].max()

x_blocks_num = math.ceil((x_max - x_min) / x_size)
y_blocks_num = math.ceil((y_max - y_min) / y_size)

print("Grid Cutting With X_Blocks ", x_blocks_num, ", Y_Blocks ", y_blocks_num)

block_analyze = BlockAnalyzeByChange(pcd1, pcd2, pcd3, x_blocks_num, y_blocks_num, "118_Disaster_2", "135")
# 启动块分析
block_analyze.block_analyze(0.1)
# 查看根据变化着色的点云
block_analyze.visualize_color_pcd(block_analyze.color_pcd, True)
# 使用统计学找异常点
block_analyze.statistics_analyze(0, 3)
# 查看统计异常点云
block_analyze.visualize_color_pcd(block_analyze.exception_pcd, True)
# 将结果保存为csv
block_analyze.save_result_as_csv()
# 将着色点云保存为pcd文件
block_analyze.save_color_blocks()
```


