from tqdm import tqdm
# 用于解析命令行参数
import argparse
# 工具函数覆盖 LLM 调用、检索、数据集处理等
from utils import *
# Freebase 相关 SPARQL 与推理函数
from freebase_func import *
# 用于随机采样候选实体
import random
# HTTP 客户端，用于与知识库或 LLM 服务交互
from client import *


# 命令行入口，负责读取数据集并调用 ToG 搜索主流程
if __name__ == '__main__':
    # 创建 ArgumentParser 对象，承载所有命令行参数配置
    parser = argparse.ArgumentParser()
    # 指定使用的数据集名称，如 webqsp / grailqa 等
    parser.add_argument("--dataset", type=str,
                        default="webqsp", help="choose the dataset.")
    # 控制 LLM 输出的最大 token 数
    parser.add_argument("--max_length", type=int,
                        default=256, help="the max length of LLMs output.")
    # 探索阶段的温度：越高越随机，利于发现多样路径
    parser.add_argument("--temperature_exploration", type=float,
                        default=0.4, help="the temperature in exploration stage.")
    # 推理阶段温度：一般设为 0，使回答更稳定
    parser.add_argument("--temperature_reasoning", type=float,
                        default=0, help="the temperature in reasoning stage.")
    # ToG 每一层保留的宽度（实体/关系数）
    parser.add_argument("--width", type=int,
                        default=3, help="choose the search width of ToG.")
    # ToG 最大搜索深度（最多扩展几跳实体）
    parser.add_argument("--depth", type=int,
                        default=3, help="choose the search depth of ToG.")
    # 是否去掉一些“无用”的 Freebase 关系
    parser.add_argument("--remove_unnecessary_rel", type=bool,
                        default=True, help="whether removing unnecessary relations.")
    # 逻辑上希望使用的 LLM 名称（实际 run_llm 内部可以被覆盖）
    parser.add_argument("--LLM_type", type=str,
                        default="gpt-3.5-turbo", help="base LLM model.")
    # OpenAI 风格接口的 API Key（当前代码中未直接使用，而是写死 siliconflow）
    parser.add_argument("--opeani_api_keys", type=str,
                        default="sk-zqatabasduftncuukvbrjgmkinkinxjzpagidisgjefsuzmf", help="if the LLM_type is gpt-3.5-turbo or gpt-4, you need add your own openai api keys.")
    # 当候选实体太多时，最多保留多少个用于后续打分
    parser.add_argument("--num_retain_entity", type=int,
                        default=5, help="Number of entities retained during entities search.")
    # 剪枝工具选择：llm / bm25 / sentencebert
    parser.add_argument("--prune_tools", type=str,
                        default="llm", help="prune tools for ToG, can be llm (same as LLM_type), bm25 or sentencebert.")
    # 解析命令行参数，得到 args 对象
    args = parser.parse_args()

    # 根据数据集名称加载样本列表和问句字段名
    datas, question_string = prepare_dataset(args.dataset)
    # 在控制台打印当前正在运行的数据集
    print("Start Running ToG on %s dataset." % args.dataset)
    # tqdm 用于显示数据集遍历的进度条
    for data in tqdm(datas):
        # 从当前样本字典中取出问句文本
        question = data[question_string]
        # topic_entity 是一个 dict，key 为实体 id，value 为实体名称
        topic_entity = data['topic_entity']
        # 用于存储每一层的实体-关系-实体 推理链（嵌套列表）
        cluster_chain_of_entities = []
        # 如果样本没有识别到主题实体，直接走纯 CoT 流程
        if len(topic_entity) == 0:
            # 没有主题实体时直接调用 CoT 推理
            results = generate_without_explored_paths(question, args)
            # 保存当前问答结果（reasoning_chains 为空列表）
            save_2_jsonl(question, results, [], file_name=args.dataset)
            # 跳到下一个样本
            continue
        # pre_relations 用于记录上一层被选中的关系，防止重复扩展同一关系
        pre_relations = []
        # pre_heads 记录每个 topic_entity 上一层是从 head 还是 tail 方向扩展
        pre_heads= [-1] * len(topic_entity)
        # 标记当前样本是否已经输出过结果（避免重复输出）
        flag_printed = False
        # 逐层扩展 ToG 搜索树（深度从 1 到最大 depth）
        for depth in range(1, args.depth+1):
            # 存放当前这一层所有实体对应的候选关系（含打分等信息）
            current_entity_relations_list = []
            # i 用来索引 pre_heads 中对应的 head/tail 标记
            i=0
            # 遍历当前层的所有主题实体（字典的 key 是实体 id）
            for entity in topic_entity:
                # "[FINISH_ID]" 表示该路径已经终止，不再扩展
                if entity!="[FINISH_ID]":
                    # 针对每个实体搜索关系并做裁剪，返回打分后的 relation
                    # entity 是实体 id，topic_entity[entity] 是实体名称
                    retrieve_relations_with_scores = relation_search_prune(entity, topic_entity[entity], pre_relations, pre_heads[i], question, args)  # best entity triplet, entitiy_id
                    # 将该实体的候选关系列表加入当前层的总列表
                    current_entity_relations_list.extend(retrieve_relations_with_scores)
                # 移动到下一个实体，对应更新 pre_heads 索引
                i+=1
            # total_candidates：累计所有关系扩展后得到的候选“新实体名”
            total_candidates = []
            # total_scores：对应候选实体的综合分数
            total_scores = []
            # total_relations：每个候选实体是通过哪条 relation 得到的
            total_relations = []
            # total_entities_id：候选实体的 ID（m.xxxxx）
            total_entities_id = []
            # total_topic_entities：每个候选实体对应的上一层“主题实体 id”
            total_topic_entities = []
            # total_head：每个候选是从 head 还是 tail 方向扩展出来
            total_head = []

            # 遍历本层中所有“实体-关系”的组合，继续搜索下一层实体
            for entity in current_entity_relations_list:
                # 根据关系方向决定查询 head 还是 tail 实体
                if entity['head']:
                    # head 为 True，表示当前实体在三元组中作为 head，去找 tail
                    entity_candidates_id = entity_search(entity['entity'], entity['relation'], True)
                else:
                    # head 为 False，表示当前实体在三元组中作为 tail，去找 head
                    entity_candidates_id = entity_search(entity['entity'], entity['relation'], False)
                
                # 如果用 LLM 做实体剪枝，在候选实体过多时随机下采样
                if args.prune_tools == "llm":
                    if len(entity_candidates_id) >=20:
                        # 从全部候选实体 id 中随机采样固定数量
                        entity_candidates_id = random.sample(entity_candidates_id, args.num_retain_entity)

                # 无候选实体则跳过当前关系
                if len(entity_candidates_id) ==0:
                    continue
                # 对所有候选实体做打分（结合关系得分和问句相关度）
                scores, entity_candidates, entity_candidates_id = entity_score(question, entity_candidates_id, entity['score'], entity['relation'], args)
                
                # 将当前关系下的所有候选实体信息写入全局累积队列
                total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head = update_history(entity_candidates, entity, scores, entity_candidates_id, total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head)
            
            # 没有新的实体加入，触发半截停止
            if len(total_candidates) ==0:
                # 使用当前已经构建的推理链直接生成答案
                half_stop(question, cluster_chain_of_entities, depth, args)
                # 表示本样本已经输出结果
                flag_printed = True
                # 结束当前样本的深度扩展循环
                break
                
            # 根据累计得分筛选下一批实体
            flag, chain_of_entities, entities_id, pre_relations, pre_heads = entity_prune(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores, args)
            # 将当前深度的链条（若干三元组）加入整体链条列表
            cluster_chain_of_entities.append(chain_of_entities)
            # flag 为 True 表示本层筛选到了非零得分的实体
            if flag:
                # 调用 reasoning 判断当前链条是否已经足够回答问题
                stop, results = reasoning(question, cluster_chain_of_entities, args)
                if stop:
                    # 若 LLM 判断可以回答，则在此深度停止 ToG 扩展
                    print("ToG stoped at depth %d." % depth)
                    # 将包含推理链的最终回答保存到 jsonl
                    save_2_jsonl(question, results, cluster_chain_of_entities, file_name=args.dataset)
                    # 标记当前样本已经输出结果
                    flag_printed = True
                    # 跳出深度循环，处理下一个样本
                    break
                else:
                    # 若 LLM 判断仍无法回答，则继续向下一层扩展
                    print("depth %d still not find the answer." % depth)
                    # 检查本层得到的实体 id 列表是否全部为 FINISH 标记
                    flag_finish, entities_id = if_finish_list(entities_id)
                    if flag_finish:
                        # 所有路径都终止，不再有新实体可扩展，执行半截停止
                        half_stop(question, cluster_chain_of_entities, depth, args)
                        flag_printed = True
                    else:
                        # 将当前层保留下来的实体 id 映射成 {id: name} 形式，下层作为新的 topic_entity
                        topic_entity = {entity: id2entity_name_or_type(entity) for entity in entities_id}
                        # 继续下一层的 for depth 循环
                        continue
            else:
                # flag 为 False 说明有效实体得分全为 0，无法继续扩展，执行半截停止
                half_stop(question, cluster_chain_of_entities, depth, args)
                flag_printed = True
        
        # 如果在所有深度内都没能产生带链条的答案，则退化为纯 CoT
        if not flag_printed:
            # 直接用不依赖图搜索的 CoT 方式生成答案
            results = generate_without_explored_paths(question, args)
            # 保存结果（reasoning_chains 为空）
            save_2_jsonl(question, results, [], file_name=args.dataset)
