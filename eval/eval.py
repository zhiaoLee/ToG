import argparse
# 导入当前 eval 目录下的工具函数（对齐答案、解析结果、保存指标等）
from utils import *

# 命令行入口：读取 ToG 的输出文件，与标准答案对齐、计算 EM
if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 指定评测使用的数据集名称（需与训练/推理阶段一致）
    parser.add_argument("--dataset", type=str,
                        default="cwq", help="choose the dataset.")
    # 指定模型推理的输出文件（json），里面包含每条样本的 question + results
    parser.add_argument("--output_file", type=str,
                        default="ToG_cwq.json", help="the output file name.")
    # 是否约束/跳过 LLM 拒答样本（如输出中出现道歉等拒绝词）
    parser.add_argument("--constraints_refuse", type=bool,
                        default=True, help="LLM may have refuse erorr, enable this option to skip current sample.")
    # 标记当前评测的方法名，写入最终结果 json 中
    parser.add_argument("--method", type=str,
                        default="ToG", help="")
    # 解析命令行参数
    args = parser.parse_args()

    # 读取标准答案数据、问句字段名，以及模型输出数据
    ground_truth_datas, question_string, output_datas = prepare_dataset_for_eval(args.dataset, args.output_file)

    # 统计预测正确和错误的样本数
    num_right = 0
    num_error = 0
    # 遍历模型输出中的每一条样本
    for data in output_datas:
        # 根据 question 文本，从标准数据中找到对应条目，并解析成 answer 列表
        answers = align(args.dataset, question_string, data, ground_truth_datas)
        # 从模型输出中取出结果字段；不同方法/数据集结构可能不同，但统一假设有 'results'
        results = data['results']      ## 根据不同数据集、算法，此处有差异
        # 若结果字符串中包含 '{'，视为带有结构化信息（如 {answer}）
        if check_string(results):
            # 提取大括号中的主要内容
            response = clean_results(results)
            # clean_results 返回 "NULL" 代表未找到大括号，退化为用原始字符串
            if response=="NULL":
                response = results
            else:
                # 用规范化后的 response 与标准答案列表做精确/包含匹配
                if exact_match(response, answers):
                    num_right+=1
                else:
                    num_error+=1
        else:
            # 不包含 '{' 的情况，直接使用原始字符串作为 response
            response = results
            # 若开启 constraints_refuse，并且检查到字符串中仍有 '{'（理论上不会），则跳过此样本
            if args.constraints_refuse and check_string(response):
                continue
            # 执行 exact_match，比对 response 与答案列表
            if exact_match(response, answers):
                num_right+=1
            else:
                num_error+=1

    # 打印整体 Exact Match 指标（正确样本数 / 总样本数）
    print("Exact Match: {}".format(float(num_right/len(output_datas))))
    # 打印正确样本数和错误样本数
    print("right: {}, error: {}".format(num_right, num_error))

    # 将评测结果（数据集、方法名、EM、right/error 数量）写入 json 文件
    save_result2json(args.dataset, num_right, num_error, len(output_datas), args.method)
    