

# 基于knn标签传播半监督算法的细胞标注工具

输入:

输入包含三个文件：

1. metaData.txt 表示已标注的细胞的细胞以及类型
2. train_data.txt 已标注的细胞的基因表达谱
3. 需要标注的基因表达谱

# Run

```

mkdir data

把文件放到data文件下

修改client_config.py的路径

python semi_classifier.py


```
