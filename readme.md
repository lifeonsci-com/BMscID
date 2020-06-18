

# 基于knn标签传播半监督算法的细胞标注工具

**环境: python3.5以上**

输入:

准备以下三个文件：

1. metaData.txt 表示已标注的细胞的细胞以及类型
2. train_data.txt 已标注的细胞的基因表达谱
3. 需要标注的基因表达谱

# Run

```
git clone git@github.com:lifeonsci-com/GeneClassifer.git

pip install -r requirements.txt

mkdir data
mkdir result

把需要的文件放到data文件下

修改client_config.py的文件名

python semi_classifier.py


```



















