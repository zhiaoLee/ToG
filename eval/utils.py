import json
# 用于从 LLM 输出或结果中提取大括号包裹的内容
import re


# 读取评测所需的标准数据和模型输出文件
def prepare_dataset_for_eval(dataset_name, output_file):
    # 针对 ComplexWebQuestions (cwq) 数据集
    if dataset_name == 'cwq':
        # 读取数据集中所有条目
        with open('../data/cwq.json',encoding='utf-8') as f:
            datas = json.load(f)
        # cwq 中问句字段名为 "question"
        question_string = 'question'
    # 针对 WebQSP 数据集
    elif dataset_name == 'webqsp':
        with open('../data/WebQSP.json',encoding='utf-8') as f:
            datas = json.load(f)
        # WebQSP 问句字段名为 "RawQuestion"
        question_string = 'RawQuestion'
    # 针对 GrailQA 数据集
    elif dataset_name == 'grailqa':
        with open('../data/grailqa.json',encoding='utf-8') as f:
            datas = json.load(f)
        # GrailQA 问句字段名为 "question"
        question_string = 'question'
    # 针对 SimpleQA 数据集
    elif dataset_name == 'simpleqa':
        with open('../data/SimpleQA.json',encoding='utf-8') as f:
            datas = json.load(f)    
        # SimpleQA 问句字段名为 "question"
        question_string = 'question'
    # 针对 QALD-10 英文数据集
    elif dataset_name == 'qald':
        with open('../data/qald_10-en.json',encoding='utf-8') as f:
            datas = json.load(f) 
        # QALD 问句字段名为 "question"
        question_string = 'question'   
    # 针对 WebQuestions 数据集
    elif dataset_name == 'webquestions':
        with open('../data/WebQuestions.json',encoding='utf-8') as f:
            datas = json.load(f)
        # WebQuestions 问句字段名也为 "question"
        question_string = 'question'
    # 针对 T-REX 数据集
    elif dataset_name == 'trex':
        with open('../data/T-REX.json',encoding='utf-8') as f:
            datas = json.load(f)
        # T-REX 用 "input" 字段存放句子
        question_string = 'input'    
    # 针对 Zero_Shot_RE 数据集
    elif dataset_name == 'zeroshotre':
        with open('../data/Zero_Shot_RE.json',encoding='utf-8') as f:
            datas = json.load(f)
        # 同样使用 "input" 为输入字段
        question_string = 'input'    
    # 针对 Creak 数据集
    elif dataset_name == 'creak':
        with open('../data/creak.json',encoding='utf-8') as f:
            datas = json.load(f)
        # Creak 中句子字段名为 "sentence"
        question_string = 'sentence'
    else:
        # 不支持的数据集名称时，打印可选项并退出
        print("dataset not found, you should pick from {cwq, webqsp, grailqa, simpleqa, qald, webquestions, trex, zeroshotre, creak}.")
        exit(-1)
    # 读取模型推理输出文件（通常由 ToG 主程序生成）
    with open(output_file, encoding='utf-8') as f:
        output_datas = json.load(f)
    # 返回：标准数据列表、问句字段名、模型输出列表
    return datas, question_string, output_datas


# 将标准答案与当前样本对齐，并抽取成统一的 answer_list
def align(dataset_name, question_string, data, ground_truth_datas):
    # 用于收集该样本所有可接受的答案字符串（去重前）
    answer_list= []
    # 从 ground_truth_datas 中找到与当前 data 具有相同问句的原始数据条目
    origin_data = [j for j in ground_truth_datas if j[question_string] == data[question_string]][0]
    # 不同数据集的答案字段格式各不相同，这里逐一处理
    if dataset_name == 'cwq':
        # cwq 可能存在 "answers" 或 "answer" 两种键，做兼容
        if 'answers' in origin_data:
            answers = origin_data["answers"]
        else:
            answers = origin_data["answer"]
        # 每个 answer 条目中既有 aliases，也有主答案
        for answer in answers:
            alias = answer['aliases']
            ans = answer['answer']
            # 将主答案加入别名列表
            alias.append(ans)
            # extend 将整组 alias 添加到 answer_list
            answer_list.extend(alias)

    elif dataset_name == 'webqsp':
        # WebQSP 中的答案在 "Parses" 列表中
        answers = origin_data["Parses"]
        for answer in answers:
            # 每个 parse 下有 "Answers" 列表，包含多个候选答案
            for name in answer['Answers']:
                # 若 EntityName 为 None，则使用 AnswerArgument（可能是字符串值）
                if name['EntityName'] == None:
                    answer_list.append(name['AnswerArgument'])
                else:
                    # 否则使用实体名称（更人类可读）
                    answer_list.append(name['EntityName'])

    elif dataset_name == 'grailqa':
        # GrailQA 中 "answer" 字段本身就是列表
        answers = origin_data["answer"]
        for answer in answers:
            # 若有 "entity_name"，优先使用实体名称
            if "entity_name" in answer:
                answer_list.append(answer['entity_name'])
            else:
                # 否则使用 answer_argument 作为答案文本
                answer_list.append(answer['answer_argument'])

    elif dataset_name == 'simpleqa':
        # SimpleQA 的 "answer" 字段通常为一个字符串
        answers = origin_data["answer"]
        answer_list.append(answers)

    elif dataset_name == 'qald':
        # QALD 中 "answer" 通常为一个 dict，键为变量名
        answers = origin_data["answer"]
        for answer in answers:
            # 取出每个变量对应的答案字符串
            answer_list.append(answers[answer])
        
    elif dataset_name == 'webquestions':
        # WebQuestions 中 "answers" 直接为列表
        answer_list = origin_data["answers"]

    elif dataset_name == 'trex' or dataset_name == 'zeroshotre':
        # T-REX / Zero_Shot_RE 中 "answer" 通常是一个字符串
        answers = origin_data["answer"]
        answer_list.append(answers)

    elif dataset_name == 'creak':
        # Creak 中 'label' 为 gold label（如 True/False 或 yes/no）
        answer = origin_data['label']
        answer_list.append(answer)

    # 使用 set 去重后再转回 list，避免重复答案
    return list(set(answer_list))
    
# 检查字符串中是否包含 '{'，用于粗略判断是否带有结构化内容
def check_string(string):
    return "{" in string

# 从 LLM 输出的字符串中提取首个大括号内的内容
def clean_results(string):
    # 若包含 '{'，则取第一个大括号之间的文本
    if "{" in string:
        # 找到 '{' 的起始位置，加 1 跳过 '{'
        start = string.find("{") + 1
        # 找到与之匹配的第一个 '}'
        end = string.find("}")
        # 截取大括号之间的内容
        content = string[start:end]
        return content
    else:
        # 若不含 '{'，则返回 "NULL" 作为占位
        return "NULL"
    

# 检查模型是否可能在回答中出现“拒答”模式（如道歉、however 等）
def check_refuse(string):
    # 定义一些典型拒答关键词
    refuse_words = ["however", "sorry"]
    # 若任意关键词出现在字符串（小写）中，则认为存在拒答风险
    return any(word in string.lower() for word in refuse_words)


# 对模型预测结果和标准答案执行“宽松”的精确匹配
def exact_match(response, answers):
    # 归一化 response：去掉首尾空格、所有空格，并转小写
    clean_result = response.strip().replace(" ","").lower()
    # 遍历所有标准答案
    for answer in answers:
        # 同样对 answer 做归一化
        clean_answer = answer.strip().replace(" ","").lower()
        # 如果完全相等，或者一方包含另一方，都视为匹配成功
        if clean_result == clean_answer or clean_result in clean_answer or clean_answer in clean_result:
            return True
    # 若所有答案都未匹配，则返回 False
    return False

# 将评测结果写入 json 文件，方便后续统计与可视化
def save_result2json(dataset_name, num_right, num_error, total_nums, method):
    # 构造结果字典，包含数据集名、方法名、EM 值以及样本统计
    results_data = {
        'dataset': dataset_name,
        'method': method,
        'Exact Match': float(num_right/total_nums),
        'Right Samples': num_right,
        'Error Sampels': num_error
    }
    # 使用数据集名构造结果文件名，例如 ToG_cwq_results.json
    with open('ToG_{}_results.json'.format(dataset_name), 'w', encoding='utf-8') as f:
        # 将结果以缩进格式写入 json，保留中文字符
        json.dump(results_data, f, ensure_ascii=False, indent=4)
                     
# 针对多轮 {yes}{answer} 格式的输出，从字符串中提取真正的内容部分
def extract_content(s):
    # 使用正则匹配所有大括号中的内容
    matches = re.findall(r'\{(.*?)\}', s)
    # 若至少有两个匹配且第一个为 "yes"，则第二个为真正答案
    if len(matches) >= 2 and matches[0].lower() == 'yes':
        return matches[1]
    # 否则若至少有一个匹配，则返回第一个
    elif len(matches) >= 1:
        return matches[0]
    else:
        # 未匹配到任何大括号内容时，返回 "NULL"
        return 'NULL'