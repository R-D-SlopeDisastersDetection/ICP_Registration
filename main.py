import similarity, curved_rebuild, coordinate_transform

path1 = "terra_pcd/cloudb14209456efa07c_Block_6.pcd"
path2 = "terra_pcd/cloudb14209456efa07c_Block_6.pcd"

# similarity.voxel_similarity(path1, path2)
curved_rebuild.poisson_mesh_rebuild(path1)
