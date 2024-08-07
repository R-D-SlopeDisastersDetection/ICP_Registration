DBSCAN点云分类

point cloud has 4 clusters
Segmentation fault (core dumped)

体素网格重合度
```
[Open3D Error] (open3d::geometry::VoxelGrid& open3d::geometry::VoxelGrid::operator+=(const open3d::geometry::VoxelGrid&)) 
/root/Open3D/cpp/open3d/geometry/VoxelGrid.cpp:132: Could not combine VoxelGrid because origin differs (this=%f,%f,%f, other=%f,%f,%f)
```
应该是体素网格的起始点没有对准

```
Traceback (most recent call last):
  File "/home/ubuntu/ChuHaoH/Code/Open3dTest/main.py", line 6, in <module>
    similarity.voxel_similarity(path1, path2)
  File "/home/ubuntu/ChuHaoH/Code/Open3dTest/similarity.py", line 92, in voxel_similarity
    union = voxel_grid1 | voxel_grid2
            ~~~~~~~~~~~~^~~~~~~~~~~~~
TypeError: unsupported operand type(s) for |: 'open3d.cuda.pybind.geometry.VoxelGrid' and 'open3d.cuda.pybind.geometry.VoxelGrid'
```
可能吃chatgpt写错了，找点别的样例代码


体素网络可视化
```
RuntimeError: [Open3D Error] (class std::shared_ptr<class open3d::geometry::PointCloud> __cdecl open3d::geometry::TriangleMesh::SamplePointsPoissonDisk
(unsigned __int64,double,const class std::shared_ptr<class open3d::geometry::PointCloud>,bool)) D:\a\Open3D\Open3D\cpp\open3d\geometry\TriangleMesh.cpp:527: Input mesh has no triangles
```


参考坐标和大地坐标的转换