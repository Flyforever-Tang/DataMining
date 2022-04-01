# 引入必要的库
import pandas as pd
import matplotlib.pyplot as plt
from efficient_apriori import apriori

# 分析Wine Reviews数据集的winemag-data_first150k.csv'文件
# csv文件路径
csv_file_path = 'D:/Data/data_mining/1/Wine Reviews/winemag-data_first150k.csv'

# 读取数据，并查看前10行数据
data_frame = pd.read_csv(csv_file_path)
data_frame = data_frame.dropna(how='any').astype(str)
print(data_frame[:10])

# 选择出标称属性对应的列
nominal_columns = [1, 3, 6, 7, 8, 9, 10]
data_frame = data_frame[[column for column in data_frame.columns[nominal_columns]]]

# 预处理成适合进行关联规则挖掘的形式
apriori_data = []
for _, data in data_frame.iterrows():
    apriori_data.append(data)
print(apriori_data[:10])
# 设置显示的最大数目
max_visual_num = 30

# 频繁模式
item_sets, rules = apriori(apriori_data, min_support=0.005, min_confidence=0.3)
print({key: {key2: item_sets[key][key2]
             for key2 in list(item_sets[key].keys())[:max_visual_num // len(list(item_sets.keys()))]}
       for key in list(item_sets.keys())})

# 关联规则
rules = rules[:max_visual_num]
# 依次显示关联规则，及其支持度、置信度、Lift评价和卡方评价
supports, confidences, lifts, convictions = [], [], [], []
for i, rule in enumerate(rules):
    rules[i] = str(rules[i])
    print(rule)
    # 支持度
    supports.append(rule.support)
    print('support:', supports[-1])
    # 置信度
    confidences.append(rule.confidence)
    print('confidence:', confidences[-1])
    # Lift评价
    lifts.append(rule.lift)
    print('Lift:', lifts[-1])
    # 卡方评价
    convictions.append(rule.conviction)
    print('Conviction:', convictions[-1])
# 可视化
# 支持度
plt.title('supports')
plt.xticks(range(len(rules)), rules, rotation=90)
plt.bar(range(len(supports)), supports)
plt.show()
# 置信度
plt.title('confidences')
plt.xticks(range(len(rules)), rules, rotation=90)
plt.bar(range(len(confidences)), confidences)
plt.show()
# Lifts评价
plt.title('lifts')
plt.xticks(range(len(rules)), rules, rotation=90)
plt.bar(range(len(lifts)), lifts)
plt.show()
# 卡方评价
plt.title('convictions')
plt.xticks(range(len(rules)), rules, rotation=90)
plt.bar(range(len(convictions)), convictions)
plt.show()

''''''

# 分析Wine Reviews数据集的winemag-data-130k-v2.csv'文件
# csv文件路径
csv_file_path = 'D:/Data/data_mining/1/Wine Reviews/winemag-data-130k-v2.csv'

# 读取数据，并查看前10行数据
data_frame = pd.read_csv(csv_file_path)
data_frame = data_frame.dropna(how='any').astype(str)
print(data_frame[:10])

# 选择出标称属性对应的列
nominal_columns = [1, 3, 6, 7, 8, 9, 10, 12, 13]
data_frame = data_frame[[column for column in data_frame.columns[nominal_columns]]]

# 预处理成适合进行关联规则挖掘的形式
apriori_data = []
for _, data in data_frame.iterrows():
    apriori_data.append(data)
print(apriori_data[:10])
# 设置显示的最大数目
max_visual_num = 30

# 频繁模式
item_sets, rules = apriori(apriori_data, min_support=0.005, min_confidence=0.3)
print({key: {key2: item_sets[key][key2]
             for key2 in list(item_sets[key].keys())[:max_visual_num // len(list(item_sets.keys()))]}
       for key in list(item_sets.keys())})
# 关联规则
rules = rules[:max_visual_num]
# 依次显示关联规则，及其支持度、置信度、Lift评价和卡方评价
supports, confidences, lifts, convictions = [], [], [], []
for i, rule in enumerate(rules):
    rules[i] = str(rules[i])
    print(rule)
    # 支持度
    supports.append(rule.support)
    print('support:', supports[-1])
    # 置信度
    confidences.append(rule.confidence)
    print('confidence:', confidences[-1])
    # Lift评价
    lifts.append(rule.lift)
    print('Lift:', lifts[-1])
    # 卡方评价
    convictions.append(rule.conviction)
    print('Conviction:', convictions[-1])
# 可视化
# 支持度
plt.title('supports')
plt.xticks(range(len(rules)), rules, rotation=90)
plt.bar(range(len(supports)), supports)
plt.show()
# 置信度
plt.title('confidences')
plt.xticks(range(len(rules)), rules, rotation=90)
plt.bar(range(len(confidences)), confidences)
plt.show()
# Lifts评价
plt.title('lifts')
plt.xticks(range(len(rules)), rules, rotation=90)
plt.bar(range(len(lifts)), lifts)
plt.show()
# 卡方评价
plt.title('convictions')
plt.xticks(range(len(rules)), rules, rotation=90)
plt.bar(range(len(convictions)), convictions)
plt.show()
