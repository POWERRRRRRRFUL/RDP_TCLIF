import os
import json
import pandas as pd

# 使用当前目录作为 results 文件夹路径
results_path = './'  # 当前路径，即 Python 文件与训练文件夹同级

# 初始化表格的字典
data = {}

# 遍历当前目录中的所有子文件夹
for folder in os.listdir(results_path):
    folder_path = os.path.join(results_path, folder)
    if os.path.isdir(folder_path):
        # 找到 args.json 和 test_res.csv 文件
        args_file = os.path.join(folder_path, 'args.json')
        results_file = os.path.join(folder_path, 'test_res.csv')

        # 检查文件是否存在
        if os.path.exists(args_file) and os.path.exists(results_file):
            # 读取 args.json 获取参数设置
            with open(args_file, 'r') as f:
                args = json.load(f)

            # 读取 test_res.csv 获取第二行的最后一个值作为 accuracy
            df = pd.read_csv(results_file, header=None)
            last_accuracy = df.iloc[1, -1]  # 获取第二行的最后一个值

            # 准备参数列内容，从 args.json 中提取关键参数
            parameters = {
                'Network': args.get('network'),  # 网络模型
                'Task': args.get('task'),
                'Optimizer': args.get('optim'),
                'Learning Rate': args.get('lr'),
                'Batch Size': args.get('batch_size'),
                'Epochs': args.get('epochs'),  # 训练轮数
                'Neuron Model': args.get('neuron'),  # 神经元模型
            }

            # 按神经元模型分组
            neuron_model = parameters.pop('Neuron Model')
            if neuron_model not in data:
                data[neuron_model] = {'Parameters': [], 'Final Accuracy': []}

            # 将参数和准确性添加到相应的神经元模型组
            data[neuron_model]['Parameters'].append(parameters)
            data[neuron_model]['Final Accuracy'].append(last_accuracy)

# 构造输出表格
output_data = []

for neuron, values in data.items():
    # 创建一个新的 DataFrame，用于每个神经元模型的数据
    params_df = pd.DataFrame(values['Parameters'])
    header_row = pd.DataFrame([params_df.columns], columns=params_df.columns)  # 创建参数名称行
    params_df = pd.concat([header_row, params_df], ignore_index=True)  # 插入参数名称行

    # 逐列检查并合并相同的 Network 参数
    params_df['Network'] = params_df['Network'].mask(params_df['Network'].duplicated(), '')

    # 创建准确率数据，确保与参数对齐
    accuracies = pd.Series([''] * (params_df.shape[0] - len(values['Final Accuracy'])) + values['Final Accuracy'],
                           name='Final Accuracy')

    # 将神经元名称、参数和准确性合并
    neuron_column = pd.Series([neuron] + [''] * (params_df.shape[0] - 1), name='Neuron Model')
    combined_df = pd.concat([neuron_column, params_df, accuracies], axis=1)
    output_data.append(combined_df)

# 合并所有神经元模型的数据
final_df = pd.concat(output_data, ignore_index=True)

# 输出到 Excel 或 CSV 文件
final_df.to_excel('summary_results.xlsx', index=False)  # 可改为 to_csv('summary_results.csv') 输出为 CSV 文件
print("Summary table has been saved as 'summary_results.xlsx'.")
