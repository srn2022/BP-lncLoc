"""提取LncRNA的k-mer特征"""

import csv
import pandas as pd

def get_basic_group(n, k, letters, m='', k_letter=''):
    if m == '':
        m = n - k
        k_letter = list(letters[:n])
    if n == m + 1: return k_letter
    num_of_perm = []
    for perm in get_basic_group(n - 1, k, letters, m, k_letter):
        temp_letter = [i for i in k_letter]
        num_of_perm += [perm + i for i in temp_letter]
    # print('perm数量', num_of_perm)
    return num_of_perm

def get_kmer_features():
    # data = pd.read_excel("./sequence/lncRNA.xlsx")
    data = pd.read_csv("./sequence/lncRNA.csv", header=None)
    labels = data.iloc[:, 0].tolist()  # 第一列是标签
    k = 0
    while 1 > k or k > 8:
        k = int(input('input k(>=1)：'))

    basic = ['A', 'C', 'G', 'T']
    xulie = []

    with open('./sequence/lncRNA.csv', 'r') as f:
        reader = csv.reader(f)
        for i in enumerate(reader):
            xulie.append(i[1][0])

    # with open('./sequence/lncRNA.csv', 'r') as f:
    #     reader = csv.reader(f)
    #     for i in enumerate(reader):
    #         xulie.append(i[1][1])

    basic_group = get_basic_group(len(basic), k, basic)
    base_group_count = list(dict(zip(basic_group, [0 for _ in range(len(basic_group))])) for _ in range(len(xulie)))

    for i in range(len(xulie)):
        for j in range(len(xulie[i])-k+1):
            base_group_count[i][xulie[i][j:j+k]] += 1

    # normalization
    feat_f = [list(base_group_count[i].values()) for i in range(len(base_group_count))]
    feat = []
    for i in range(len(feat_f)):
        sum_f = 0
        for j in range(len(basic_group)):
            sum_f += feat_f[i][j]
        feat.append([round(feat_f[i][n] / sum_f, 4) for n in range(len(feat_f[0]))])
    print('序列数量:', len(feat))

    feature_data = pd.DataFrame(feat)
    feature_data.insert(0, "Label", labels)

    # 将DataFrame保存为CSV文件
    # output_path = './other_dataset/caozhen/' + str(k) + '_mer.csv'
    # feature_data.to_csv(output_path, index=False, header=False)

    # # 将DataFrame保存到Excel文件
    output_path = './sequence/' + str(k) + '_mer.xlsx'
    feature_data.to_excel(output_path, index=False, header=False) #

    print('结果已保存至 Excel 文件：', output_path)


if __name__ == '__main__':
    get_kmer_features()
