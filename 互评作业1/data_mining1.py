# 引入必要的库
import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import *
from enum import Enum
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer


# 属性类型，对应无关属性、标称属性、数值属性
class AttributeType(Enum):
    Nonsense = 0
    Nominal = 1
    Numeric = 2


# 缺失值处理方法
class MissingProcessing(object):
    # 将缺失部分剔除
    @staticmethod
    def eliminate(data_array: np.ndarray) -> list:
        return [data for data in data_array if not np.isnan(data)]

    # 用最高频率值来填补缺失值
    @staticmethod
    def frequencyFill(data_array: np.ndarray) -> np.ndarray:
        if len(data_array.shape) == 1:
            data_array = data_array.reshape(-1, 1)
        return SimpleImputer(strategy='most_frequent').fit_transform(data_array).reshape(-1)

    # 通过属性的相关关系来填补缺失值，这里使用贝叶斯回归算法
    @staticmethod
    def relevanceFill(data_array: np.ndarray) -> np.ndarray:
        if len(data_array.shape) == 1:
            data_array = data_array.reshape(-1, 1)
        return IterativeImputer().fit_transform(data_array).reshape(-1)

    # 通过数据对象之间的相似性来填补缺失值，这里使用knn算法
    @staticmethod
    def similarityFill(data_array: np.ndarray) -> np.ndarray:
        if len(data_array.shape) == 1:
            data_array = data_array.astype(float).reshape(-1, 1)
        return KNNImputer().fit_transform(data_array).reshape(-1)

    def __init__(self,
                 method: Callable[[np.ndarray], Union[list, np.ndarray]]):
        self.method = method

    def __call__(self, data_array: np.ndarray) -> Union[list, np.ndarray]:
        return self.method(data_array)


# 数据摘要
def dataSummary(data_frame: DataFrame,
                attribute_types: List[AttributeType],
                missing_method: Callable[[np.ndarray], Union[list, np.ndarray]]) -> List:
    # 统计数据取值的频数
    def getFrequency(data_array: np.ndarray) -> dict:
        frequency_dict = {}
        for data in data_array:
            try:
                frequency_dict[data] += 1
            except KeyError:
                frequency_dict[data] = 1
        return frequency_dict

    # 获得数据的5数概括、nan值个数以及处理nan值后的数据
    def statistics(data_array: np.ndarray) -> Tuple[float, float, float, float, float, int, list]:
        nan_sum = sum(1 for data in data_array if np.isnan(data))
        if nan_sum > 0:
            data_array = missing_method(data_array)
        describe = pd.Series(list(data_array)).describe()
        return describe['min'], describe['25%'], describe['50%'], describe['75%'], describe['max'],  nan_sum, data_array

    summary_results = []
    values = data_frame.values

    for i in range(len(data_frame.columns)):
        # 根据数据属性类别获得不同的摘要结果
        if attribute_types[i] == AttributeType.Nominal:
            summary_results.append(getFrequency(values[:, i]))
        elif attribute_types[i] == AttributeType.Numeric:
            summary_results.append(statistics(values[:, i]))
        else:
            summary_results.append(None)

    return summary_results


# 数据可视化
def visualize(summary_results: List[Union[dict, Tuple, None]],
              attribute_types: List[AttributeType],
              attribute_names: List[str]) -> None:
    visualization_count = attribute_types.count(AttributeType.Numeric) * 3
    fig_x_sum = 3
    fig_y_sum = (visualization_count + fig_x_sum - 1) // fig_x_sum
    fig_index = 1
    plt.figure(figsize=(fig_x_sum * 3, fig_y_sum * 2))
    for summary_result, attribute_type, attribute_name in zip(summary_results, attribute_types, attribute_names):
        # 直方图、盒图只对数值属性的数据有效
        if attribute_type == AttributeType.Numeric:
            plt.subplot(fig_y_sum, fig_x_sum, fig_index)
            plt.title(f'Hist of {attribute_name}')
            plt.hist(summary_result[-1])
            fig_index += 1
            plt.subplot(fig_y_sum, fig_x_sum, fig_index)
            plt.title(f'Boxplot of {attribute_name}\n(with outliers)')
            plt.boxplot(summary_result[-1], showfliers=True)
            fig_index += 1
            plt.subplot(fig_y_sum, fig_x_sum, fig_index)
            plt.title(f'Boxplot of {attribute_name}\n(without outliers)')
            plt.boxplot(summary_result[-1], showfliers=False)
            fig_index += 1
        else:
            continue
    plt.tight_layout()
    plt.show()


# 数据处理流程
def processing(csv_file_path: str,
               attribute_types: List[AttributeType],
               missing_method: Callable[[np.ndarray], Union[list, np.ndarray]] = MissingProcessing.eliminate,
               max_display_sum: int = 10) -> None:
    data_frame = pd.read_csv(csv_file_path)
    summary_results = dataSummary(data_frame, attribute_types, missing_method)
    for i, summary_result in enumerate(summary_results):
        if summary_result is not None:
            if isinstance(summary_result, dict):
                # 为了便于查看，只显示前max_display_sum条统计结果
                print(data_frame.columns[i],
                      dict(sorted(summary_result.items(), key=lambda x: x[1], reverse=True)[:max_display_sum]))
            else:
                print(data_frame.columns[i], summary_result[:6])
    visualize(summary_results, attribute_types, data_frame.columns)


# 数据分析流程
# 分析Wine Reviews数据集的winemag-data_first150k.csv'文件
# csv文件路径
csv_file_path = 'D:/Data/data_mining/1/Wine Reviews/winemag-data_first150k.csv'
# 标注出无关属性、数值属性的列序号，其余的为标称属性
nonsense_columns = [0, 2]
Numeric_columns = [4, 5]
attributeType = [AttributeType.Nonsense if i in nonsense_columns
                 else AttributeType.Numeric if i in Numeric_columns
                 else AttributeType.Nominal
                 for i in range(11)]
# 依次采用不同的缺失数据处理方法进行处理
# 将缺失部分剔除
processing(csv_file_path, attributeType, MissingProcessing.eliminate)
# 用最高频率值来填补缺失值
processing(csv_file_path, attributeType, MissingProcessing.frequencyFill)
# 通过属性的相关关系来填补缺失值
processing(csv_file_path, attributeType, MissingProcessing.relevanceFill)
# 通过数据对象之间的相似性来填补缺失值
processing(csv_file_path, attributeType, MissingProcessing.similarityFill)

# 分析Wine Reviews数据集的winemag-data-130k-v2.csv'文件
# csv文件路径
csv_file_path = 'D:/Data/data_mining/1/Wine Reviews/winemag-data-130k-v2.csv'
# 标注出无关属性、数值属性的列序号，其余的为标称属性
nonsense_columns = [0, 2, 11]
Numeric_columns = [4, 5]
attributeType = [AttributeType.Nonsense if i in nonsense_columns
                 else AttributeType.Numeric if i in Numeric_columns
                 else AttributeType.Nominal
                 for i in range(14)]
# 依次采用不同的缺失数据处理方法进行处理
# 将缺失部分剔除
processing(csv_file_path, attributeType, MissingProcessing.eliminate)
# 用最高频率值来填补缺失值
processing(csv_file_path, attributeType, MissingProcessing.frequencyFill)
# 通过属性的相关关系来填补缺失值
processing(csv_file_path, attributeType, MissingProcessing.relevanceFill)
# 通过数据对象之间的相似性来填补缺失值
processing(csv_file_path, attributeType, MissingProcessing.similarityFill)

# 分析Chicago Building Violations数据集的building-violations.csv'文件
# csv文件路径
csv_file_path = 'D:/Data/data_mining/1/Chicago Building Violations/building-violations.csv'
# 标注出无关属性、数值属性的列序号，其余的为标称属性
nonsense_columns = [0, 1, 2, 5, 8, 25]
Numeric_columns = [22, 23, 24, 26, 29, 30, 31]
attributeType = [AttributeType.Nonsense if i in nonsense_columns
                 else AttributeType.Numeric if i in Numeric_columns
                 else AttributeType.Nominal
                 for i in range(32)]
# 依次采用不同的缺失数据处理方法进行处理
# 将缺失部分剔除
processing(csv_file_path, attributeType, MissingProcessing.eliminate)
# 用最高频率值来填补缺失值
processing(csv_file_path, attributeType, MissingProcessing.frequencyFill)
# 通过属性的相关关系来填补缺失值
processing(csv_file_path, attributeType, MissingProcessing.relevanceFill)
# 通过数据对象之间的相似性来填补缺失值
processing(csv_file_path, attributeType, MissingProcessing.similarityFill)
