from prompt_list import *
# 用于读写数据集与结果文件
import json
# 用于实现简单重试和等待
import time
# OpenAI 兼容的 LLM SDK
import openai
# 正则表达式，用于解析 LLM 输出
import re
from prompt_list import *
# BM25 稀疏检索库
from rank_bm25 import BM25Okapi
# SentenceTransformer 提供的相似度工具
from sentence_transformers import util
# Sentence-BERT 模型类，用于编码文本
from sentence_transformers import SentenceTransformer

# 基于 Sentence-BERT 的 dense 检索，返回最相关的 doc 及得分
def retrieve_top_docs(query, docs, model, width=3):
    """
    Retrieve the topn most relevant documents for the given query.

    Parameters:
    - query (str): The input query.
    - docs (list of str): The list of documents to search from.
    - model_name (str): The name of the SentenceTransformer model to use.
    - width (int): The number of top documents to return.

    Returns:
    - list of float: A list of scores for the topn documents.
    - list of str: A list of the topn documents.
    """

    # 获取查询和文档的语义向量（均为高维向量）
    query_emb = model.encode(query)
    doc_emb = model.encode(docs)

    # 计算点积相似度并转成 python list（scores 长度等于 docs 数量）
    scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()

    # 绑定 doc 与 score，按得分降序排列，形成 (doc, score) 对
    doc_score_pairs = sorted(list(zip(docs, scores)), key=lambda x: x[1], reverse=True)

    # 截断出前 width 个结果，对应 top_docs 与 top_scores
    top_docs = [pair[0] for pair in doc_score_pairs[:width]]
    top_scores = [pair[1] for pair in doc_score_pairs[:width]]

    return top_docs, top_scores


# 经典 BM25 稀疏检索，衡量问句与候选关系之间的匹配度
def compute_bm25_similarity(query, corpus, width=3):
    """
    Computes the BM25 similarity between a question and a list of relations,
    and returns the topn relations with the highest similarity along with their scores.

    Args:
    - question (str): Input question.
    - relations_list (list): List of relations.
    - width (int): Number of top relations to return.

    Returns:
    - list, list: topn relations with the highest similarity and their respective scores.
    """

    # BM25 需要预先分词的语料，这里简单按照空格切分
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    # 基于分词后的语料构建 BM25 索引
    bm25 = BM25Okapi(tokenized_corpus)
    # 将 query 也按空格切分
    tokenized_query = query.split(" ")

    # 针对每个候选计算得分，doc_scores 长度为 len(corpus)
    doc_scores = bm25.get_scores(tokenized_query)
    
    # 取最高分的 relation 以及对应分值（按相关性降序）
    relations = bm25.get_top_n(tokenized_query, corpus, n=width)
    doc_scores = sorted(doc_scores, reverse=True)[:width]

    return relations, doc_scores


# 解析 LLM 返回的 relation+score 文本
def clean_relations(string, entity_id, head_relations):
    # 使用命名捕获组匹配 { relation (Score: 0.8) } 的结构
    pattern = r"{\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)}"
    # relations 用于存放解析出的字典结果
    relations=[]
    # 遍历 LLM 输出中的每个 {relation (Score: x)} 片段
    for match in re.finditer(pattern, string):
        # relation 文本内容，去掉首尾空格
        relation = match.group("relation").strip()
        # 过滤可能包含多个 relation 的拼接
        if ';' in relation:
            continue
        # 得分字符串
        score = match.group("score")
        # 若 relation 或 score 为空，说明 LLM 输出被截断
        if not relation or not score:
            return False, "output uncompleted.."
        try:
            # 尝试将 score 转浮点数
            score = float(score)
        except ValueError:
            # 转换失败则认为输出格式非法
            return False, "Invalid score"
        # 标记 relation 是否来自 head 方向，便于后续实体扩展
        if relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": True})
        else:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": False})
    # 若未能解析出任何 relation，则返回失败
    if not relations:
        return False, "No relations found"
    # 正常返回成功标志和 relations 列表
    return True, relations


# 判断列表是否全部为 0
def if_all_zero(topn_scores):
    # 使用 all() 逐个检查元素是否等于 0
    return all(score == 0 for score in topn_scores)


# 将 BM25/SBERT 输出包装成统一结构
def clean_relations_bm25_sent(topn_relations, topn_scores, entity_id, head_relations):
    # relations 存储结构化的关系信息
    relations = []
    # 若所有得分为 0，则退化为平均分（避免全部为 0 的退化情况）
    if if_all_zero(topn_scores):
        topn_scores = [float(1/len(topn_scores))] * len(topn_scores)
    # i 记录当前处理的下标
    i=0
    for relation in topn_relations:
        # 根据 relation 所属方向给出 head 标记
        if relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "score": topn_scores[i], "head": True})
        else:
            relations.append({"entity": entity_id, "relation": relation, "score": topn_scores[i], "head": False})
        # 累加下标
        i+=1
    # 返回 True（BM25/SBERT 输出一般不会失败）和 relations
    return True, relations


# 统一的 LLM ChatCompletion 调用封装，支持替换不同推理后端
def run_llm(prompt, temperature, max_tokens, opeani_api_keys, engine="gpt-3.5-turbo"):
    # if "llama" in engine.lower():
    #     openai.api_key = "EMPTY"
    #     openai.api_base = "http://localhost:8000/v1"  # your local llama server port
    #     engine = openai.Model.list()["data"][0]["id"]
    # else:
    #     openai.api_key = opeani_api_keys

    ## siliconflow.
    # 设置 siliconflow 提供的 API Key（这里写死在代码中，实际使用时应通过配置注入）
    openai.api_key = "sk-zqatabasduftncuukvbrjgmkinkinxjzpagidisgjefsuzmf"
    # 设置 siliconflow 的 API Base 地址
    openai.api_base = "https://api.siliconflow.cn/v1"  # your local llama server port
    # 选择具体使用的模型名称
    engine = "Qwen/Qwen2.5-72B-Instruct"

    ## openrouter
    # 以下为使用 openrouter 作为后端时的配置示例（当前被注释掉）
    # openai.api_key = "sk-or-v1-..."
    # openai.api_base = "https://openrouter.ai/api/v1"  # your local llama server port
    # engine = "openai/gpt-3.5-turbo"

    # 构造 ChatCompletion 所需的 system 与 user 消息
    messages = [{"role":"system","content":"You are an AI assistant that helps people find information."}]
    # user 消息中放入拼接好的 prompt 文本
    message_prompt = {"role":"user","content":prompt}
    # 将 user 消息追加到列表中
    messages.append(message_prompt)
    # f 作为重试标记，0 表示尚未成功
    f = 0
    # 在 LLM 调用失败时进行简单重试
    while(f == 0):
        try:
            # 通过 ChatCompletion.create 发送请求
            response = openai.ChatCompletion.create(
                    model=engine,
                    messages = messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    frequency_penalty=0,
                    presence_penalty=0)
            # 读取第一条回答内容
            result = response["choices"][0]['message']['content']
            # 标记调用成功，跳出循环
            f = 1
        except:
            # 捕获异常（如网络错误），打印提示并等待 2 秒重试
            print("openai error, retry")
            time.sleep(2)
    # 返回最终得到的回答文本
    return result

    
# 检查候选实体是否全部为未知占位符
def all_unknown_entity(entity_candidates):
    # 所有候选都为 "UnName_Entity" 时返回 True
    return all(candidate == "UnName_Entity" for candidate in entity_candidates)


# 删除 "UnName_Entity"，保留真实实体
def del_unknown_entity(entity_candidates):
    # 当列表长度为 1 且为 Unknown 时，直接返回，避免丢失唯一实体
    if len(entity_candidates)==1 and entity_candidates[0]=="UnName_Entity":
        return entity_candidates
    # 否则过滤掉所有 "UnName_Entity"
    entity_candidates = [candidate for candidate in entity_candidates if candidate != "UnName_Entity"]
    return entity_candidates


# 从 LLM 文本中抽取浮点分数，与候选实体长度对齐
def clean_scores(string, entity_candidates):
    # 抽取所有浮点数形式的得分（如 0.1、0.23 等）
    scores = re.findall(r'\d+\.\d+', string)
    # 将字符串列表转换成浮点数列表
    scores = [float(number) for number in scores]
    # 若分数个数与实体个数相同，说明解析正常
    if len(scores) == len(entity_candidates):
        return scores
    else:
        # 否则说明 LLM 返回数量不匹配，退化为“所有实体平等”
        print("All entities are created equal.")
        return [1/len(entity_candidates)] * len(entity_candidates)
    

# 将问答结果和推理链条写入 jsonl 以便评测
def save_2_jsonl(question, answer, cluster_chain_of_entities, file_name):
    # 将当前样本的问句、答案和推理链组装成字典
    dict = {"question":question, "results": answer, "reasoning_chains": cluster_chain_of_entities}
    # 以追加模式打开 ToG_数据集名.jsonl
    with open("ToG_{}.jsonl".format(file_name), "a") as outfile:
        # 将字典序列化成字符串
        json_str = json.dumps(dict)
        # 每条记录一行写入文件
        outfile.write(json_str + "\n")

    
# 从 LLM 返回的字符串中提取 {answer} 包裹的内容
def extract_answer(text):
    # 找到第一个左花括号的位置
    start_index = text.find("{")
    # 找到第一个右花括号的位置
    end_index = text.find("}")
    # 若都存在，则截取中间部分作为答案
    if start_index != -1 and end_index != -1:
        return text[start_index+1:end_index].strip()
    else:
        # 否则返回空字符串，表示未匹配到
        return ""
    

# 将 LLM 的 yes/no 判断转布尔量
def if_true(prompt):
    # 转小写、去空格后与 "yes" 比较
    if prompt.lower().strip().replace(" ","")=="yes":
        return True
    # 默认为 False
    return False


# 当未能扩展知识图时，直接调用 CoT Prompt 获得答案
def generate_without_explored_paths(question, args):
    # 拼接 Chain-of-Thought 的 prompt 模板与问句
    prompt = cot_prompt + "\n\nQ: " + question + "\nA:"
    # 调用 run_llm 获取答案文本
    response = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type)
    # 直接返回 LLM 输出
    return response


# 判断是否所有实体均为终止节点
def if_finish_list(lst):
    # 若列表中每个元素都是 "[FINISH_ID]"，说明所有路径均已终止
    if all(elem == "[FINISH_ID]" for elem in lst):
        return True, []
    else:
        # 否则过滤掉 FINISH 标记，返回仍可扩展的实体 id
        new_lst = [elem for elem in lst if elem != "[FINISH_ID]"]
        return False, new_lst


# 根据数据集名称读取对应 json 并返回问句字段名
def prepare_dataset(dataset_name):
    # 处理 ComplexWebQuestions (cwq) 数据集
    if dataset_name == 'cwq':
        # 打开对应 json 文件，注意使用 utf-8 编码
        with open('../data/cwq.json',encoding='utf-8') as f:
            # 读取全部样本列表
            datas = json.load(f)
        # cwq 中问句字段名为 "question"
        question_string = 'question'
    # 处理 WebQSP 数据集
    elif dataset_name == 'webqsp':
        with open('../data/WebQSP.json',encoding='utf-8') as f:
            datas = json.load(f)
        # WebQSP 中原始问句字段为 "RawQuestion"
        question_string = 'RawQuestion'
    # 处理 GrailQA 数据集
    elif dataset_name == 'grailqa':
        with open('../data/grailqa.json',encoding='utf-8') as f:
            datas = json.load(f)
        # GrailQA 问句字段为 "question"
        question_string = 'question'
    # 处理 SimpleQA 数据集
    elif dataset_name == 'simpleqa':
        with open('../data/SimpleQA.json',encoding='utf-8') as f:
            datas = json.load(f)    
        # SimpleQA 问句字段为 "question"
        question_string = 'question'
    # 处理 QALD 10 英文数据集
    elif dataset_name == 'qald':
        with open('../data/qald_10-en.json',encoding='utf-8') as f:
            datas = json.load(f) 
        # QALD 中问句字段名为 "question"
        question_string = 'question'   
    # 处理 WebQuestions 数据集
    elif dataset_name == 'webquestions':
        with open('../data/WebQuestions.json',encoding='utf-8') as f:
            datas = json.load(f)
        # WebQuestions 问句字段名也为 "question"
        question_string = 'question'
    # 处理 T-REX 数据集（用于关系抽取场景）
    elif dataset_name == 'trex':
        with open('../data/T-REX.json',encoding='utf-8') as f:
            datas = json.load(f)
        # T-REX 中输入字段名为 "input"
        question_string = 'input'    
    # 处理 Zero_Shot_RE 数据集
    elif dataset_name == 'zeroshotre':
        with open('../data/Zero_Shot_RE.json',encoding='utf-8') as f:
            datas = json.load(f)
        # 同样使用 "input" 作为问句字段
        question_string = 'input'    
    # 处理 Creak 数据集（常识推理句子）
    elif dataset_name == 'creak':
        with open('../data/creak.json',encoding='utf-8') as f:
            datas = json.load(f)
        # Creak 中句子字段名为 "sentence"
        question_string = 'sentence'
    else:
        # 若传入未支持的数据集名称，则打印可选列表并退出
        print("dataset not found, you should pick from {cwq, webqsp, grailqa, simpleqa, qald, webquestions, trex, zeroshotre, creak}.")
        exit(-1)
    # 返回样本列表和问句字段名字符串
    return datas, question_string