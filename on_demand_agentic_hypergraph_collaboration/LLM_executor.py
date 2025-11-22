import os
import queue
import threading

from tqdm import tqdm
import httpx
from openai import OpenAI
from collections import defaultdict
import json
import random
import sys

sys.path.append('/home/dex/Desktop/entity_sy/MTKGA-Wild')
from ThreadPoolExecutor import ThreadPoolExecutor


import tokens_cal

def load_descriptions(file_path):
    """Load entity description information"""
    with open(file_path, 'r', encoding='utf-8') as f:
        descriptions = json.load(f)
    return {int(k): v[0]['description'] if v else "" for k, v in descriptions.items()}

def get_entity_context_m3(entity_id, entity_names, descriptions):
    """Get descriptive information about the entity (used for is_first_time=True)"""
    return f"Entity Name: {entity_names.get(entity_id, 'Unknown')}\nDescription:\n{descriptions.get(entity_id, 'No description available')}"

def get_random_rules(data_dir, n=5):
    """Randomly select n rules from the rules file"""
    LLM_executor_MESSAGE_POOL = {
        'important_entities': os.path.join(data_dir, "message_pool", "important_entities.txt"),
        'ucon_similarity_results': os.path.join(data_dir, "message_pool", "ucon_similarity_results.txt"),
        'alignment_rules': os.path.join(data_dir, "message_pool", "alignment_rules.txt"),
        'execution_alignment_pairs': os.path.join(data_dir, "message_pool", "execution_alignment_pairs.txt"),
    }
    rules = []
    rules_file = LLM_executor_MESSAGE_POOL['alignment_rules']


    if not os.path.exists(rules_file):
        os.makedirs(os.path.dirname(rules_file), exist_ok=True)
        with open(rules_file, 'w', encoding='utf-8') as f:
            pass  # 创建空文件
        print(f"Created empty rules file: {rules_file}")
        return rules


    with open(rules_file, 'r', encoding='utf-8') as f:
        all_rules = f.readlines()

    if all_rules:
        rules = random.sample(all_rules, min(n, len(all_rules)))

    return rules


def load_multimodal_info(file_path):
    """Load multimodal auxiliary information (video/txt/audio/image links)
    Returns empty dict if file doesn't exist or has errors"""
    if not os.path.exists(file_path):
        print(f"[INFO] Multimodal file not found: {file_path}, skipping...")
        return {}

    multimodal_info = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # 跳过空行
                    continue

                parts = line.split('\t')
                if len(parts) >= 2:
                    try:
                        entity_id = int(parts[0])
                        link = parts[1]
                        multimodal_info[entity_id] = link
                    except ValueError:
                        print(f"[WARNING] Invalid entity ID in {file_path} line {line_num}: {parts[0]}")
                        continue
                else:
                    print(f"[WARNING] Invalid format in {file_path} line {line_num}: {line}")
    except Exception as e:
        print(f"[ERROR] Failed to load {file_path}: {str(e)}")
        return {}

    if multimodal_info:
        print(f"[INFO] Loaded {len(multimodal_info)} entries from {file_path}")

    return multimodal_info


def format_multimodal_info(entity_id, video_info, txt_info, audio_info, image_info):
    """Format multimodal auxiliary information
    Returns empty string if no information available"""
    info_parts = []

    # 安全地检查每种类型的信息
    if video_info and entity_id in video_info:
        info_parts.append(f"Video Link: {video_info[entity_id]}")
    if txt_info and entity_id in txt_info:
        info_parts.append(f"Text Link: {txt_info[entity_id]}")
    if audio_info and entity_id in audio_info:
        info_parts.append(f"Audio Link: {audio_info[entity_id]}")
    if image_info and entity_id in image_info:
        info_parts.append(f"Image Link: {image_info[entity_id]}")

    if info_parts:
        return "\nAdditional Information:\n" + "\n".join(info_parts)
    return ""



def load_entity_names(file_path):
    """Load mapping of entity ids and names"""
    entity_names = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                entity_names[int(parts[0])] = parts[1]
    return entity_names

def load_triples(file_path):
    """Load ternary data"""
    triples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            triples.append([int(x) for x in parts[:3]])
    return triples


def get_entity_context(entity_id, entity_names, triples, rel_names,
                       video_info=None, txt_info=None, audio_info=None, image_info=None, n=3):
    """Get the first n relationships of the entity with multimodal info"""
    # 设置默认空字典
    video_info = video_info or {}
    txt_info = txt_info or {}
    audio_info = audio_info or {}
    image_info = image_info or {}

    relations = []
    for h, r, t in triples:
        if h == entity_id:
            rel_str = rel_names.get(r, str(r))
            tail_str = entity_names.get(t, str(t))
            relations.append(f"- Has relation '{rel_str}' with {tail_str}")
        elif t == entity_id:
            rel_str = rel_names.get(r, str(r))
            head_str = entity_names.get(h, str(h))
            relations.append(f"- Is {rel_str} of {head_str}")
        if len(relations) >= n:
            break

    context = f"Entity Name: {entity_names.get(entity_id, 'Unknown')}\n"
    context += "Relationships:\n" + "\n".join(relations[:n]) if relations else "Relationships:\nNo relationships found"

    multimodal_str = format_multimodal_info(entity_id, video_info, txt_info, audio_info, image_info)
    if multimodal_str:
        context += multimodal_str

    return context


def group_candidates(input_file):
    """Group candidate entity pairs in the input file by KG1 entities"""
    groups = defaultdict(list)
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            e1, e2 = map(int, line.strip().split('\t'))
            groups[e1].append(e2)
    return groups

def align_entities(data_dir, is_first_time = False, ablation_config = None, no_optimization_tool = False):


    LLM_executor_MESSAGE_POOL = {
        'important_entities': os.path.join(data_dir, "message_pool", "conflict.txt"),
        'ucon_similarity_results': os.path.join(data_dir, "message_pool", "conflict.txt"),
        'alignment_rules': os.path.join(data_dir, "message_pool", "alignment_rules.txt"),
        'execution_alignment_pairs': os.path.join(data_dir, "message_pool", "execution_alignment_pairs.txt"),
        'video_link_trans_1': os.path.join(data_dir, "video_link_trans_1"),
        'video_link_trans_2': os.path.join(data_dir, "video_link_trans_2"),
        'txt_link_trans_1': os.path.join(data_dir, "txt_link_trans_1"),
        'txt_link_trans_2': os.path.join(data_dir, "txt_link_trans_2"),
        'audio_link_trans_1': os.path.join(data_dir, "audio_link_trans_1"),
        'audio_link_trans_2': os.path.join(data_dir, "audio_link_trans_2"),
        'image_link_trans_1': os.path.join(data_dir, "image_link_trans_1"),
        'image_link_trans_2': os.path.join(data_dir, "image_link_trans_2")
    }

    LLM1_Agent_Profile = '''
Goal: As a knowledge graph alignment expert, determine if the first entity represents the same object as one of the entities in the candidate entity list.
Constraint: if there is a match, return only the ID of the matching candidate entity number; if none of them matches (is not the same object), return "No"; if none of them matches (is not the same object), return "No"; if none of them matches (is not the same object), return "No".
    '''

    input_file = LLM_executor_MESSAGE_POOL['ucon_similarity_results'] if is_first_time else LLM_executor_MESSAGE_POOL['important_entities']
    output_file = LLM_executor_MESSAGE_POOL['execution_alignment_pairs']

    # Setting up the OpenAI client
    client = OpenAI(
        base_url="xxx",
        api_key="xxx",
        http_client=httpx.Client(
            base_url="xxx",
            follow_redirects=True,
        ),
    )


    ent_names_1 = load_entity_names(os.path.join(data_dir, 'ent_ids_1'))
    ent_names_2 = load_entity_names(os.path.join(data_dir, 'ent_ids_2'))


    # Load multimodal auxiliary information for both KGs (with error handling)
    print("\n=== Loading Multimodal Information ===")
    video_info_1 = load_multimodal_info(LLM_executor_MESSAGE_POOL.get('video_link_trans_1', ''))
    video_info_2 = load_multimodal_info(LLM_executor_MESSAGE_POOL.get('video_link_trans_2', ''))
    txt_info_1 = load_multimodal_info(LLM_executor_MESSAGE_POOL.get('txt_link_trans_1', ''))
    txt_info_2 = load_multimodal_info(LLM_executor_MESSAGE_POOL.get('txt_link_trans_2', ''))
    audio_info_1 = load_multimodal_info(LLM_executor_MESSAGE_POOL.get('audio_link_trans_1', ''))
    audio_info_2 = load_multimodal_info(LLM_executor_MESSAGE_POOL.get('audio_link_trans_2', ''))
    image_info_1 = load_multimodal_info(LLM_executor_MESSAGE_POOL.get('image_link_trans_1', ''))
    image_info_2 = load_multimodal_info(LLM_executor_MESSAGE_POOL.get('image_link_trans_2', ''))

    # 统计加载的多模态信息
    total_multimodal = sum([
        len(video_info_1), len(video_info_2),
        len(txt_info_1), len(txt_info_2),
        len(audio_info_1), len(audio_info_2),
        len(image_info_1), len(image_info_2)
    ])
    print(f"Total multimodal entries loaded: {total_multimodal}")
    if total_multimodal == 0:
        print("[WARNING] No multimodal information available, proceeding with text-only alignment")
    print("=" * 40 + "\n")


    if is_first_time:
        # 移除了KG对比描述的加载
        rules = get_random_rules(data_dir)
    else:
        rel_names_1 = load_entity_names(os.path.join(data_dir, 'rel_ids_1'))
        rel_names_2 = load_entity_names(os.path.join(data_dir, 'rel_ids_2'))
        triples_1 = load_triples(os.path.join(data_dir, 'triples_1'))
        triples_2 = load_triples(os.path.join(data_dir, 'triples_2'))


    # Grouping of candidate entities
    candidate_groups = group_candidates(input_file)
    aligned_pairs = []

    lock = threading.Lock()
    executor = ThreadPoolExecutor(max_workers=30)
    # Create a queue to store writes to the file
    result_queue = queue.Queue()

    def openai_task(kg1_entity, kg2_candidates):
        try:
            if is_first_time:
                context1 = f"Entity Name: {ent_names_1.get(kg1_entity, 'Unknown')}"
                # �� 添加这3行
                multimodal_str_1 = format_multimodal_info(kg1_entity, video_info_1, txt_info_1, audio_info_1,
                                                          image_info_1)
                if multimodal_str_1:
                    context1 += multimodal_str_1

                candidates_contexts = []
                for kg2_entity in kg2_candidates:
                    context2 = f"Entity Name: {ent_names_2.get(kg2_entity, 'Unknown')}"
                    # �� 添加这3行
                    multimodal_str_2 = format_multimodal_info(kg2_entity, video_info_2, txt_info_2, audio_info_2,
                                                              image_info_2)
                    if multimodal_str_2:
                        context2 += multimodal_str_2

                    candidates_contexts.append({
                        'entity_id': kg2_entity,
                        'context': context2
                    })


            else:
                context1 = get_entity_context(kg1_entity, ent_names_1, triples_1, rel_names_1,

                                              video_info_1, txt_info_1, audio_info_1, image_info_1)
                candidates_contexts = []
                for kg2_entity in kg2_candidates:
                    # �� 添加多模态参数
                    context = get_entity_context(kg2_entity, ent_names_2, triples_2, rel_names_2,
                                                 video_info_2, txt_info_2, audio_info_2, image_info_2)
                    candidates_contexts.append({
                        'entity_id': kg2_entity,
                        'context': context
                    })
            # Building the Prompt
            prompt = LLM1_Agent_Profile + f"""
                                Entity 1 (ID: {kg1_entity}):
                                {context1}

                                the candidate entity list:"""




            for i, candidate in enumerate(candidates_contexts, 1):
                prompt += f"\n\ncandidate entity{i} (ID: {candidate['entity_id']}):\n{candidate['context']}"

            prompt += """\n\nDo any of these candidate entities represent the same object as entity 1? If so, only the corresponding entity ID is returned; if none of them match (is not the same object), only "No" is returned; if none of them match (is not the same object), only "No" is returned; if none of them match (is not the same object), only "No" is returned:"""

            if is_first_time and rules:
                prompt += "\n\nReference Rules:\n" + "".join(rules)

            response = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[{'role': 'user', 'content': prompt}]
            )

            answer = response.choices[0].message.content.strip()

            tokens_cal.update_add_var(response.usage.total_tokens)  # update tokens

            # The kg1_entity and kg2_candidates are printed here for each execution.
            print(f"Processing entity {kg1_entity} with candidates {kg2_candidates}")

            print(prompt, answer)
            # an analytic response
            if answer.lower() != "no":
                # Trying to extract the entity ID from the answer
                for kg2_id in kg2_candidates:
                    if str(kg2_id) in answer:
                        with lock:
                            result_queue.put((kg1_entity, kg2_id))
                            aligned_pairs.append((kg1_entity, kg2_id))
                        break

        except Exception as e:
            print(f"Error processing entity {kg1_entity}: {str(e)}")

    # Processing each group of candidate entities
    for kg1_entity_c, kg2_candidates_c in tqdm(candidate_groups.items()):
        executor.submit(openai_task,kg1_entity_c, kg2_candidates_c)

    # Save results
    # if os.path.exists(output_file):
    #     with open(output_file, 'a', encoding='utf-8') as f:
    #         for e1, e2 in aligned_pairs:
    #             f.write(f"{e1}\t{e2}\n")
    # else:
    #     with open(output_file, 'w', encoding='utf-8') as f:
    #         for e1, e2 in aligned_pairs:
    #             f.write(f"{e1}\t{e2}\n")

    executor.shutdown(wait=True)

    # Write the results from the queue to a file
    with open(output_file, 'a+', encoding='utf-8') as output_f:
        while not result_queue.empty():
            kg1_entity, kg2_id = result_queue.get()
            output_f.write(f"{kg1_entity}\t{kg2_id}\n")
            output_f.flush()  # Flush the buffer immediately to ensure that it is written to disk

    deduplicate_output_file(output_file)

    return aligned_pairs

def deduplicate_output_file(file_path):
    """De-duplication of the output file"""
    if not os.path.exists(file_path):
        return

    # Reads all rows and de-duplicates them
    unique_pairs = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            e1, e2 = map(int, line.strip().split('\t'))
            unique_pairs.add((e1, e2))

    # Rewrite the result after de-duplication
    with open(file_path, 'w', encoding='utf-8') as f:
        for e1, e2 in sorted(unique_pairs):  # Sorting to maintain stable output
            f.write(f"{e1}\t{e2}\n")

    print(f"Deduplicated file {file_path}: {len(unique_pairs)} unique pairs")

if __name__ == "__main__":
    data_dir = "/home/dex/Desktop/entity_sy/AdaCoAgent_backup/data/icews_wiki/"
    aligned_pairs = align_entities(data_dir)
    print(f"Found {len(aligned_pairs)} aligned entity pairs.")