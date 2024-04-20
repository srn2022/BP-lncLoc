## BP-lncLoc可以用来预测lncRNA亚细胞定位，包括4定位和5定位。

### 文件说明

| 文件        | 说明                    |
| :---------- | ----------------------- |
| main.py     | 基于cao和su数据集       |
| protonet.py | 模型层                  |
| utils.py    | mian调用的其他通用函数  |

### 运行示例

```shell

python3 main.py or main_4class.py #main.py是5定位(cpu版本) ， main_4class.py是4定位（GPU版本）
```

"""训练集验证集测试集分别有自己支持集，查询集，在get_data形成，get_data中分别形成三个过程的数据索引，
用支持集形成原型，再用查询集去查询欧式距离，main对应protonet,6/10训练集，2/10验证集，2/10测试集"""

运行环境
torch==1.8.0
numpy==1.21.6
openpyxl==3.0.10

