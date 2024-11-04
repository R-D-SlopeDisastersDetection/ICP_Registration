from pyproj import Proj, transform


# # 定义WGS84坐标系（大地坐标）
# wgs84 = Proj(init='epsg:4326')
# # 定义UTM投影坐标系（这里假设为UTM Zone 33N）
# utm33n = Proj(init='epsg:32633')
#
# # 示例UTM坐标
# east, north = 646447.474, 2579150.153
#
# # 转换为大地坐标（经度、纬度）
# longitude, latitude = transform(utm33n, wgs84, east, north)
#
# print(f"Longitude: {longitude}, Latitude: {latitude}")

# p = Proj(proj="CSC2000", ellipsis="WGS84")
# x=646506.103, y=2580182.152

def coordinate_transform(x, y, proj='tmerc', lat=0, lon=111, ellps='WGS84'):
    """

    :param x: 大地坐标的经度
    :param y:
    :param proj:
    :param lat:
    :param lon:
    :param ellps:
    :return:
    """
    proj = Proj(proj=proj, lat_0=lat, lon_0=lon, k=1, x_0=500000, y_0=0, ellps=ellps, preserve_units=False)
    return proj(x, y, inverse=True)


class Util:
    pass
