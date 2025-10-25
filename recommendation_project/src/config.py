from pathlib import Path

# __file__ 是当前文件 (config.py) 的路径
# Path(__file__) 将它变成一个 Path 对象
# .resolve() 获取绝对路径
# .parent 获取父目录 (src 文件夹)
# .parent 再次获取父目录 (my_project 文件夹，也就是项目根目录)
ROOT_DIR = Path(__file__).resolve().parent

# 你也可以在这里定义其他重要的路径
DATA_DIR = ROOT_DIR / "data"