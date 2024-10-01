import os
import pandas as pd
import json
import matplotlib.pyplot as plt

# 定义 results 文件夹的路径
results_dir = './'  # 当前目录

# 存储所有 test_res.csv 数据的列表
all_data = []

# 遍历 results 文件夹中的所有子文件夹
for subdir, dirs, files in os.walk(results_dir):
    for dir_name in dirs:
        dir_path = os.path.join(subdir, dir_name)

        # 构造 args.json 文件路径
        args_json_path = os.path.join(dir_path, 'args.json')

        # 检查 args.json 文件是否存在
        if os.path.exists(args_json_path):
            # 读取 args.json 文件
            with open(args_json_path, 'r') as f:
                args_data = json.load(f)

            # 检查 network 字段是否为 'cnn'
            if args_data.get('network', '').lower() == 'cnn':
                test_res_path = os.path.join(dir_path, 'test_res.csv')

                # 检查 test_res.csv 是否存在
                if os.path.exists(test_res_path):
                    # 读取 test_res.csv 文件
                    df = pd.read_csv(test_res_path, header=None)
                    if df.empty:
                        print(f"{test_res_path} is empty, skipping.")
                    else:
                        # 第一行为 epoch，第二行为 accuracy
                        epoch = df.iloc[0, :].values
                        accuracy = df.iloc[1, :].values
                        run_data = pd.DataFrame({'epoch': epoch, 'accuracy': accuracy})
                        run_data['run'] = dir_name
                        all_data.append(run_data)

# 检查是否有数据
if not all_data:
    print("No test_res.csv files found or all files are empty.")
else:
    # 合并所有数据
    combined_data = pd.concat(all_data, ignore_index=True)

    # 保存到 Excel 文件
    excel_path = './cnn_test_results.xlsx'
    combined_data.to_excel(excel_path, index=False)

    # 绘图：绘制每个 run 的 accuracy 随 epoch 变化的曲线
    plt.figure(figsize=(10, 6))
    for run in combined_data['run'].unique():
        run_data = combined_data[combined_data['run'] == run]
        plt.plot(run_data['epoch'], run_data['accuracy'], label=run)

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy over Epochs for CNN Runs')
    plt.legend()
    plt.grid(True)

    # 保存绘图
    plt.savefig('./cnn_test_accuracy_plot.png')

    # 展示绘图
    plt.show()

    print(f"结果已保存到 {excel_path}, 绘图已保存到 'cnn_test_accuracy_plot.png'")
