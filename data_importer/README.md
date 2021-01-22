# 简介

这是nyu和msra 的手部数据集的导入工程,包含数据的一些预处理

# Install

- git clone srv01@192.168.6.131:/opt/hub/data_importer


## 数据依赖(需要自己配置)
- 数据放在根目录下的./data
- 我已经建立了一个 Dataserver(可以通过地址下载)
- 对于在本地的导入的data, 可以在data文件夹内,运行

	lns -f /data/datasets/nyu NYU
	lns -f /data/datasets/cvpr15_MSRAHandGestureDB MSRA15

- cache,cache放在 ./data_importer/cache, 对于本地可以直接(推荐生成cache)

	lns -f /data/datasets/dm_cache_new cache

## 安装成一个插件

	cd {path of data_importer}
	pip Install -e .




