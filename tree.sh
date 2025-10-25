#!/bin/bash

# 定义项目根目录名称
PROJECT_NAME="recommendation_project"

# 创建项目主目录
mkdir "$PROJECT_NAME"
cd "$PROJECT_NAME"

# 创建一级目录
mkdir -p data/unprocessed data/processed
mkdir configs
mkdir notebooks
mkdir -p src/data_processing src/model src/training
mkdir -p saved_models/checkpoints

# 创建空的 Python 包文件 (__init__.py)
touch src/__init__.py
touch src/data_processing/__init__.py
touch src/model/__init__.py
touch src/training/__init__.py

# 创建空的脚本和配置文件
touch configs/config.yaml
touch notebooks/1-data_exploration.ipynb
touch notebooks/2-model_prototyping.ipynb
touch src/dataset.py
touch src/data_processing/preprocess.py
touch src/model/encoders.py
touch src/model/two_tower.py
touch src/training/trainer.py
touch src/train.py
touch src/evaluate.py
touch src/predict.py
touch requirements.txt
touch README.md

# 创建 .gitignore 并写入基本内容
echo "# Python cache
__pycache__/
*.pyc

# Data files
data/

# Saved models
saved_models/

# IDE files
.vscode/
.idea/

# Notebook checkpoints
.ipynb_checkpoints/" > .gitignore

echo "项目结构 '$PROJECT_NAME' 创建成功！"
