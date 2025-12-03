from SPARQLWrapper import SPARQLWrapper, JSON
# 引入本目录下的工具函数（LLM 调用、检索、数据处理等）
from utils import *

# Freebase triple store 暴露的 SPARQL 端点，需根据本地服务修改
SPARQLPATH = "http://localhost:8890/sparql"  # depend on your own internal address and port, shown in Freebase folder's readme.md

# 查询实体出边关系的模板（实体作为三元组头部）
sparql_head_relations = """\nPREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?relation\nWHERE {\n  ns:%s ?relation ?x .\n}"""
# 查询实体入边关系的模板（实体作为三元组尾部）
sparql_tail_relations = """\nPREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?relation\nWHERE {\n  ?x ?relation ns:%s .\n}"""
# 以 head 为已知，查询 tail 实体的模板（ns:head ns:relation ?tail）
sparql_tail_entities_extract = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?tailEntity\nWHERE {\nns:%s ns:%s ?tailEntity .\n}""" 
# 以 tail 为已知，查询 head 实体的模板（?head ns:relation ns:tail）
sparql_head_entities_extract = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?tailEntity\nWHERE {\n?tailEntity ns:%s ns:%s  .\n}"""
# 查询实体名称或 sameAs 标识的模板，合并 type.object.name 与 owl:sameAs 两种来源
sparql_id = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?tailEntity\nWHERE {\n  {\n    ?entity ns:type.object.name ?tailEntity .\n    FILTER(?entity = ns:%s)\n  }\n  UNION\n  {\n    ?entity <http://www.w3.org/2002/07/owl#sameAs> ?tailEntity .\n    FILTER(?entity = ns:%s)\n  }\n}"""
    
# 检测 relation 文本结尾是否为过于模板化的词，后续可用于过滤
def check_end_word(s):
    # words 是需要过滤的结尾候选列表
    words = [" ID", " code", " number", "instance of", "website", "URL", "inception", "image", " rate", " count"]
    # 若字符串以任一给定后缀结束，则返回 True
    return any(s.endswith(word) for word in words)

# 丢弃 Freebase 中的内部类型或无语义关系，避免污染推理
def abandon_rels(relation):
    # 对常见的无信息关系统一返回 True（表示应当被过滤掉）
    if relation == "type.object.type" or relation == "type.object.name" or relation.startswith("common.") or relation.startswith("freebase.") or "sameAs" in relation:
        return True


# 发送 SPARQL 请求并返回 bindings，所有查询均复用该方法
def execurte_sparql(sparql_query):
    # 构建 SPARQLWrapper 客户端，指定端点地址
    sparql = SPARQLWrapper(SPARQLPATH)
    # 将拼好的查询语句设置到请求中
    sparql.setQuery(sparql_query)
    # 指定返回格式为 JSON
    sparql.setReturnFormat(JSON)
    # 发送请求并转成 Python 字典
    results = sparql.query().convert()
    # 返回 "results"->"bindings" 中的主体内容
    return results["results"]["bindings"]


# 将关系 URI 去掉 Freebase 前缀，便于在 prompt 中呈现
def replace_relation_prefix(relations):
    # 遍历 JSON bindings，取出 relation 字段并替换掉固定前缀
    return [relation['relation']['value'].replace("http://rdf.freebase.com/ns/","") for relation in relations]

# 将实体 URI 去掉 Freebase 前缀，仅保留 m.xxxxx 简写
def replace_entities_prefix(entities):
    # 遍历 JSON bindings，取出 tailEntity 字段并替换掉固定前缀
    return [entity['tailEntity']['value'].replace("http://rdf.freebase.com/ns/","") for entity in entities]


# 把实体 ID 映射成可读名称或 sameAs URI，若不存在则返回占位符
def id2entity_name_or_type(entity_id):
    # 将实体 id 替换到 sparql_id 模板中，生成完整查询
    sparql_query = sparql_id % (entity_id, entity_id)
    # 构建 SPARQLWrapper 客户端
    sparql = SPARQLWrapper(SPARQLPATH)
    # 配置查询语句
    sparql.setQuery(sparql_query)
    # 设置返回格式为 JSON
    sparql.setReturnFormat(JSON)
    # 执行查询并转换结果
    results = sparql.query().convert()
    # 如果没有任何绑定结果，返回占位字符串
    if len(results["results"]["bindings"])==0:
        return "UnName_Entity"
    else:
        # 否则返回第一个 tailEntity 值（名称或 sameAs URI）
        return results["results"]["bindings"][0]['tailEntity']['value']
    
# 下面引用 prompt、LLM、BM25、Sentence-BERT 等依赖，支撑后续检索与打分
from freebase_func import *
from prompt_list import *
import json
import time
import openai
import re
from prompt_list import *
from rank_bm25 import BM25Okapi
from sentence_transformers import util
from sentence_transformers import SentenceTransformer


# 解析 LLM 返回的 relation+score 文本，转成结构化列表
def clean_relations(string, entity_id, head_relations):
    # pattern 匹配形如 { relation (Score: 0.8) } 的片段
    pattern = r"{\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)}"
    # relations 保存解析出的结构化结果
    relations=[]
    # 遍历 LLM 输出字符串中所有匹配的片段
    for match in re.finditer(pattern, string):
        # 提取 relation 文本并去掉前后空格
        relation = match.group("relation").strip()
        # 如果 relation 中包含 ';' 说明格式不标准，直接跳过
        if ';' in relation:
            continue
        # 提取得分字符串
        score = match.group("score")
        # 如果 relation 或 score 为空，则认为输出不完整
        if not relation or not score:
            return False, "output uncompleted.."
        try:
            # 尝试将 score 转为浮点数
            score = float(score)
        except ValueError:
            # 转换失败说明格式错误
            return False, "Invalid score"
        # 判断该 relation 是否属于 head 方向
        if relation in head_relations:
            # head=True 表示是从当前实体发出的边
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": True})
        else:
            # head=False 表示是指向当前实体的边
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": False})
    # 如果一个 relation 都没有解析到，也视作失败
    if not relations:
        return False, "No relations found"
    # 正常返回成功标记和解析结果
    return True, relations


# 判断打分列表是否全为 0，便于 fallback
def if_all_zero(topn_scores):
    # all() 检查所有元素是否等于 0
    return all(score == 0 for score in topn_scores)


# 将 BM25 / Sentence-BERT 检索结果统一成与 clean_relations 相同的数据结构
def clean_relations_bm25_sent(topn_relations, topn_scores, entity_id, head_relations):
    # relations 用于存储结构化后的关系信息
    relations = []
    # 如果所有得分为 0，则退化为平均分
    if if_all_zero(topn_scores):
        topn_scores = [float(1/len(topn_scores))] * len(topn_scores)
    # i 记录当前处理到第几个 relation
    i=0
    # 顺序遍历所有 topn 关系
    for relation in topn_relations:
        # 若当前 relation 属于 head_relations，则 head=True
        if relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "score": topn_scores[i], "head": True})
        else:
            # 否则标记 head=False
            relations.append({"entity": entity_id, "relation": relation, "score": topn_scores[i], "head": False})
        # 下标自增
        i+=1
    # 这里总是返回 True（BM25/SBERT 输出格式稳定）
    return True, relations


# 构造提示词，告知 LLM 从 relation 列表中选出前 args.width 条
def construct_relation_prune_prompt(question, entity_name, total_relations, args):
    # extract_relation_prompt 定义在 prompt_list 中，带有占位符 %d
    # args.width 控制要求 LLM 选出多少条关系
    return extract_relation_prompt % (args.width, args.width) + question + '\nTopic Entity: ' + entity_name + '\nRelations: '+ '; '.join(total_relations) + "\nA: "
        

# 让 LLM 对候选实体打分，返回归一化权重
def construct_entity_score_prompt(question, relation, entity_candidates):
    # score_entity_candidates_prompt 是一个包含 {} 占位符的模板
    # 此处填入问句和关系，再拼接所有候选实体名称
    return score_entity_candidates_prompt.format(question, relation) + "; ".join(entity_candidates) + '\nScore: '


# 面向单个实体，搜集其 head/tail 关系并根据策略裁剪
def relation_search_prune(entity_id, entity_name, pre_relations, pre_head, question, args):
    # 查询实体向外发出的所有关系（当前实体作为 head）
    sparql_relations_extract_head = sparql_head_relations % (entity_id)
    # 执行 SPARQL 得到原始 JSON 绑定
    # head_relations[0]:  {'relation': {'type': 'uri', 'value': 'http://rdf.freebase.com/ns/book.author.works_written'}}
    head_relations = execurte_sparql(sparql_relations_extract_head)
    # 去掉关系 URI 的前缀，保留简写
    # head_relations[0]:  'book.author.works_written'
    head_relations = replace_relation_prefix(head_relations)
    
    # 查询实体指向它的所有关系（当前实体作为 tail）
    sparql_relations_extract_tail= sparql_tail_relations % (entity_id)
    # 执行 SPARQL 得到原始 JSON 绑定
    tail_relations = execurte_sparql(sparql_relations_extract_tail)
    # 去掉 URI 前缀，保留简写
    tail_relations = replace_relation_prefix(tail_relations)

    # 根据开关过滤无信息关系
    if args.remove_unnecessary_rel:
        # 使用 abandon_rels 判定是否为内部/无用关系
        head_relations = [relation for relation in head_relations if not abandon_rels(relation)]
        tail_relations = [relation for relation in tail_relations if not abandon_rels(relation)]
    
    # 避免重复探索历史关系，head/tail 分别排除
    if pre_head:
        # 如果上一层是从 head 方向来，这一层就从 tail 方向排除历史关系
        tail_relations = list(set(tail_relations) - set(pre_relations))
    else:
        # 否则从 head 方向排除历史关系
        head_relations = list(set(head_relations) - set(pre_relations))

    # 去重并合并 head 与 tail 列表
    head_relations = list(set(head_relations))
    tail_relations = list(set(tail_relations))
    # 将 head 与 tail 两个列表拼在一起，作为候选全集
    total_relations = head_relations+tail_relations
    # 对关系名排序，保证 prompt 中各关系的顺序稳定
    total_relations.sort()  # make sure the order in prompt is always equal
    
    # 根据配置选择 LLM 或语义检索方式进行关系裁剪
    if args.prune_tools == "llm":
        # 利用 LLM 判断哪些关系最相关
        prompt = construct_relation_prune_prompt(question, entity_name, total_relations, args)

        # 调用 LLM 获得字符串形式的候选关系及得分
        result = run_llm(prompt, args.temperature_exploration, args.max_length, args.opeani_api_keys, args.LLM_type)
        # 将 LLM 输出解析为结构化列表
        flag, retrieve_relations_with_scores = clean_relations(result, entity_id, head_relations) 

    elif args.prune_tools == "bm25":
        # 使用 BM25 对 total_relations 做语义检索，返回 topn 关系及其得分
        topn_relations, topn_scores = compute_bm25_similarity(question, total_relations, args.width)
        # 转换为统一的数据结构
        flag, retrieve_relations_with_scores = clean_relations_bm25_sent(topn_relations, topn_scores, entity_id, head_relations) 
    else:
        # 使用 Sentence-BERT 模型进行语义检索
        model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b')
        # 检索问句和所有关系之间的相似度
        topn_relations, topn_scores = retrieve_top_docs(question, total_relations, model, args.width)
        # 转为统一的结构化形式
        flag, retrieve_relations_with_scores = clean_relations_bm25_sent(topn_relations, topn_scores, entity_id, head_relations) 

    # flag 为 True 表示成功解析到了至少一个关系
    if flag:
        # 返回带得分和方向信息的关系列表
        return retrieve_relations_with_scores
    else:
        # 若格式错误或 LLM 输出过短，返回空列表
        return [] # format error or too small max_length
    
    
# 根据 relation 方向查找相邻实体 ID
def entity_search(entity, relation, head=True):
    # head=True：实体在三元组中作为 head，根据 (entity, relation, ?) 查找 tail
    if head:
        # 使用 tail_entities_extract 模板填入实体 id 和 relation
        tail_entities_extract = sparql_tail_entities_extract% (entity, relation)
        # 执行查询，得到所有 tail 实体
        entities = execurte_sparql(tail_entities_extract)
    else:
        # head=False：实体在三元组中作为 tail，根据 (?, relation, entity) 查找 head
        head_entities_extract = sparql_head_entities_extract% (entity, relation)
        # 执行查询，得到所有 head 实体
        entities = execurte_sparql(head_entities_extract)

    # 仅保留 Freebase 主命名空间（m. 开头）的实体
    entity_ids = replace_entities_prefix(entities)
    # 过滤掉非 m. 开头的 id（如 type 等）
    new_entity = [entity for entity in entity_ids if entity.startswith("m.")]
    # 返回相邻实体 id 列表
    return new_entity


# 对 relation 下的候选实体按问句相关性打分
def entity_score(question, entity_candidates_id, score, relation, args):
    # 先将实体 ID 转成可读名称方便 prompt
    entity_candidates = [id2entity_name_or_type(entity_id) for entity_id in entity_candidates_id]
    # 如果所有实体都为 "UnName_Entity"，则平均分摊 relation 得分
    if all_unknown_entity(entity_candidates):
        return [1/len(entity_candidates) * score] * len(entity_candidates), entity_candidates, entity_candidates_id
    # 否则删去未知实体，仅保留命名实体
    entity_candidates = del_unknown_entity(entity_candidates)
    # 若只剩一个实体，则直接将整个 relation 分数赋给该实体
    if len(entity_candidates) == 1:
        return [score], entity_candidates, entity_candidates_id
    # 若没有任何实体，则返回 0 分
    if len(entity_candidates) == 0:
        return [0.0], entity_candidates, entity_candidates_id
    
    # 将实体名称和 ID 排序绑定，防止两者顺序不一致
    zipped_lists = sorted(zip(entity_candidates, entity_candidates_id))
    # 解压缩出排序后的实体名称列表和 id 列表
    entity_candidates, entity_candidates_id = zip(*zipped_lists)
    # 转为普通 list 便于后续处理
    entity_candidates = list(entity_candidates)
    entity_candidates_id = list(entity_candidates_id)
    # 若采用 LLM 来给实体打分
    if args.prune_tools == "llm":
        # 构造包含问句、关系和候选实体的打分 prompt
        prompt = construct_entity_score_prompt(question, relation, entity_candidates)

        # 调用 LLM 获取每个实体的得分
        result = run_llm(prompt, args.temperature_exploration, args.max_length, args.opeani_api_keys, args.LLM_type)
        # clean_scores 负责从文本中抽取得分并与实体列表长度对齐
        return [float(x) * score for x in clean_scores(result, entity_candidates)], entity_candidates, entity_candidates_id

    elif args.prune_tools == "bm25":
        # 使用 BM25 衡量问句与候选实体名称之间的相关性
        topn_entities, topn_scores = compute_bm25_similarity(question, entity_candidates, args.width)
    else:
        # 使用 Sentence-BERT 检索相关实体
        model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b')
        # 根据语义相似度选择 topn 实体
        topn_entities, topn_scores = retrieve_top_docs(question, entity_candidates, model, args.width)
    # 若所有检索得分为 0，则退化为平均分
    if if_all_zero(topn_scores):
        topn_scores = [float(1/len(topn_scores))] * len(topn_scores)
    # 将检索得分与 relation 自身得分相乘，得到实体的综合分数
    return [float(x) * score for x in topn_scores], topn_entities, entity_candidates_id

    
# 将本轮探索得到的候选实体信息写入累计队列，方便 pruning
def update_history(entity_candidates, entity, scores, entity_candidates_id, total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head):
    # 若当前关系下没有扩展出任何实体，则打上 FINISH 标记
    if len(entity_candidates) == 0:
        entity_candidates.append("[FINISH]")
        entity_candidates_id = ["[FINISH_ID]"]
    # candidates_relation：当前所有候选实体对应的 relation 名称
    candidates_relation = [entity['relation']] * len(entity_candidates)
    # topic_entities：当前所有候选实体对应的“上一层实体 id”
    topic_entities = [entity['entity']] * len(entity_candidates)
    # head_num：记录这些候选实体是 head 方向还是 tail 方向扩展出来
    head_num = [entity['head']] * len(entity_candidates)
    # 将当前批次的实体名称添加到总的实体候选列表
    total_candidates.extend(entity_candidates)
    # 将当前批次的得分添加到总得分列表
    total_scores.extend(scores)
    # 将当前批次的关系名称添加到总关系列表
    total_relations.extend(candidates_relation)
    # 将当前批次的实体 id 添加到总实体 id 列表
    total_entities_id.extend(entity_candidates_id)
    # 将上一层实体 id 添加到总 topic 实体列表
    total_topic_entities.extend(topic_entities)
    # 将 head/tail 标记添加到总 head 列表
    total_head.extend(head_num)
    # 返回更新后的所有累计容器
    return total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head


# 搜索无增量信息时提前结束，并直接依据已有路径作答
def half_stop(question, cluster_chain_of_entities, depth, args):
    # 打印停止搜索的深度，方便调试和日志分析
    print("No new knowledge added during search depth %d, stop searching." % depth)
    # 调用 generate_answer 使用已有三元组生成最终答案
    answer = generate_answer(question, cluster_chain_of_entities, args)
    # 将该样本的问句、答案和推理链保存到 jsonl 文件中
    save_2_jsonl(question, answer, cluster_chain_of_entities, file_name=args.dataset)


# 将累积的知识三元组串联进 prompt，请求 LLM 生成最终答案
def generate_answer(question, cluster_chain_of_entities, args): 
    # answer_prompt 定义了整体问答的前置说明
    prompt = answer_prompt + question + '\n'
    # chain_prompt 将所有子链条拆平，并将每个三元组转成字符串后用逗号连接
    chain_prompt = '\n'.join([', '.join([str(x) for x in chain]) for sublist in cluster_chain_of_entities for chain in sublist])
    # 将知识三元组拼进 prompt，作为 LLM 的“外部知识”
    prompt += "\nKnowledge Triplets: " + chain_prompt + 'A: '
    # 调用 LLM，在推理阶段使用 temperature_reasoning 和 max_length 等参数
    result = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type)
    # 返回原始 LLM 文本输出
    return result


# 基于累计得分排序并截断候选实体，形成新的探索前沿
def entity_prune(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores, args):
    # 将所有信息绑定成一个列表，方便按照得分统一排序
    zipped = list(zip(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores))
    # 以得分（下标 5）为 key 进行降序排序
    sorted_zipped = sorted(zipped, key=lambda x: x[5], reverse=True)
    # 将排序后的各字段拆分为独立列表
    sorted_entities_id, sorted_relations, sorted_candidates, sorted_topic_entities, sorted_head, sorted_scores = [x[0] for x in sorted_zipped], [x[1] for x in sorted_zipped], [x[2] for x in sorted_zipped], [x[3] for x in sorted_zipped], [x[4] for x in sorted_zipped], [x[5] for x in sorted_zipped]

    # 按 args.width 截断，只保留前 width 个候选
    entities_id, relations, candidates, topics, heads, scores = sorted_entities_id[:args.width], sorted_relations[:args.width], sorted_candidates[:args.width], sorted_topic_entities[:args.width], sorted_head[:args.width], sorted_scores[:args.width]
    # 将六个列表重新 zip 成元组列表，方便过滤
    merged_list = list(zip(entities_id, relations, candidates, topics, heads, scores))
    # 过滤掉得分为 0 的候选（认为其无贡献）
    filtered_list = [(id, rel, ent, top, hea, score) for id, rel, ent, top, hea, score in merged_list if score != 0]
    # 如果过滤后空列表，说明没有有效候选，返回 False
    if len(filtered_list) ==0:
        return False, [], [], [], []
    # 非空时将各字段解包为独立列表
    entities_id, relations, candidates, tops, heads, scores = map(list, zip(*filtered_list))

    # 将上一层实体 id（tops）映射成实体名称，便于后续展示
    tops = [id2entity_name_or_type(entity_id) for entity_id in tops]
    # 组织成 [[(topic_name, relation, candidate_name), ...]] 的嵌套结构
    cluster_chain_of_entities = [[(tops[i], relations[i], candidates[i]) for i in range(len(candidates))]]
    # 返回 True 说明仍有可扩展实体，并返回新一层的实体 id / 关系 / head 标记
    return True, cluster_chain_of_entities, entities_id, relations, heads


# 在每个深度节点调用 LLM 评估当前链路是否可直接回答
def reasoning(question, cluster_chain_of_entities, args):
    # prompt_evaluate 用于说明“请判断是否已经足够回答问题”
    prompt = prompt_evaluate + question
    # 将当前所有深度的三元组展开为链条字符串
    chain_prompt = '\n'.join([', '.join([str(x) for x in chain]) for sublist in cluster_chain_of_entities for chain in sublist])
    # 将知识三元组拼接到 prompt 中
    prompt += "\nKnowledge Triplets: " + chain_prompt + 'A: '

    # 调用 LLM 获取“是否可以回答”的文本结果
    response = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type)
    
    # 从 LLM 的整体回答中抽取大括号内的简短答案（例如 {yes}/{no}）
    result = extract_answer(response)
    # 如果答案语义上为“yes”，则认为可以停止扩展
    if if_true(result):
        return True, response
    else:
        # 否则继续扩展下一个深度
        return False, response
    



