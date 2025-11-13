"""
meta eva
将execution_alignment_pairs加入训练集并重新运行Simple-HHEA
"""

from typing import List, Tuple, Set  # Make sure Tuple is imported
import subprocess
import os
import queue
import threading
from concurrent.futures import as_completed

from tqdm import tqdm
import httpx
from openai import OpenAI
from collections import defaultdict
import random
import sys
sys.path.append('/home/dex/Desktop/entity_sy/MTKGA-Wild')
from ThreadPoolExecutor import ThreadPoolExecutor

import tokens_cal

def load_entity_names(file_path):
    """Load entity ID to name mapping"""
    entity_names = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                entity_names[int(parts[0])] = parts[1]
    return entity_names


def load_triples(file_path):
    """Load triple data"""
    triples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            triples.append([int(x) for x in parts[:3]])
    return triples


def load_true_pairs(file_path):
    """Load true aligned entity pairs"""
    pairs = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            e1, e2 = map(int, line.strip().split('\t'))
            pairs.add((e1, e2))
    return pairs


def get_negative_pairs(true_pairs, ent_names_1, ent_names_2, sample_size=100):
    """Generate negative pairs from entities that are not aligned"""
    kg1_entities = set(e1 for e1, _ in true_pairs)
    kg2_entities = set(e2 for _, e2 in true_pairs)

    negative_pairs = set()
    kg1_list = list(ent_names_1.keys())
    kg2_list = list(ent_names_2.keys())

    while len(negative_pairs) < sample_size:
        e1 = random.choice(kg1_list)
        e2 = random.choice(kg2_list)
        if (e1, e2) not in true_pairs:
            negative_pairs.add((e1, e2))

    return negative_pairs


def get_entity_context(entity_id, entity_names, triples, rel_names):
    """Get entity context with 3 random relations"""
    relations = []
    entity_triples = []

    for h, r, t in triples:
        if h == entity_id:
            entity_triples.append((h, r, t, True))  # True for head position
        elif t == entity_id:
            entity_triples.append((h, r, t, False))  # False for tail position

    if entity_triples:
        sampled_triples = random.sample(entity_triples, min(3, len(entity_triples)))
        for h, r, t, is_head in sampled_triples:
            rel_str = rel_names.get(r, str(r))
            if is_head:
                tail_str = entity_names.get(t, str(t))
                relations.append(f"Has relation '{rel_str}' with {tail_str}")
            else:
                head_str = entity_names.get(h, str(h))
                relations.append(f"Is {rel_str} of {head_str}")

    context = {
        "name": entity_names.get(entity_id, f"Entity_{entity_id}"),
        "relations": relations
    }
    return context


def run_full_process_llm2(data_dir, batch_size=500, ablation_config=None, no_optimization_tool=False):
    """Generate alignment rules using LLM analysis"""
    LLM2_PRIVATE_MESSAGE_POOL = {
        'alignment_rules': os.path.join(data_dir, "message_pool", "alignment_rules.txt"),
        'aligned_entities': os.path.join(data_dir, "message_pool", "aligned_entities.txt"),
        'aligned_entities_history': os.path.join(data_dir, "message_pool", "aligned_entities_history.txt"),
        'ucon_similarity_results': os.path.join(data_dir, "message_pool", "ucon_similarity_results.txt"),
    }

    # Read the last round of archived data
    previous_pairs = set()
    if os.path.exists(LLM2_PRIVATE_MESSAGE_POOL['aligned_entities_history']):
        previous_pairs = load_true_pairs(LLM2_PRIVATE_MESSAGE_POOL['aligned_entities_history'])

    true_pairs = load_true_pairs(LLM2_PRIVATE_MESSAGE_POOL['aligned_entities'])

    # Create an archive of the current data
    with open(LLM2_PRIVATE_MESSAGE_POOL['aligned_entities_history'], 'w', encoding='utf-8') as f:
        for e1, e2 in true_pairs:
            f.write(f"{e1}\t{e2}\n")

    # Filtering out added data
    new_pairs = true_pairs - previous_pairs
    if not new_pairs:
        print("No new entity pairs to process.")  # Clear entity pairs that are not sure if they are aligned or not
        if os.path.exists(LLM2_PRIVATE_MESSAGE_POOL['ucon_similarity_results']):
            with open(LLM2_PRIVATE_MESSAGE_POOL['ucon_similarity_results'], 'w') as f:
                f.write('')
        return []

    # Initialize OpenAI client
    client = OpenAI(
        base_url="xxxx",
        api_key="xxx",
        http_client=httpx.Client(
            base_url="xxx",
            follow_redirects=True,
        ),
    )

    # Load data
    ent_names_1 = load_entity_names(os.path.join(data_dir, 'ent_ids_1'))
    ent_names_2 = load_entity_names(os.path.join(data_dir, 'ent_ids_2'))
    rel_names_1 = load_entity_names(os.path.join(data_dir, 'rel_ids_1'))
    rel_names_2 = load_entity_names(os.path.join(data_dir, 'rel_ids_2'))
    triples_1 = load_triples(os.path.join(data_dir, 'triples_1'))
    triples_2 = load_triples(os.path.join(data_dir, 'triples_2'))

    # Load true pairs and generate negative pairs

    negative_pairs = get_negative_pairs(new_pairs, ent_names_1, ent_names_2)

    all_rules = []

    executor = ThreadPoolExecutor(max_workers=10)

    def process_positive_pair(e1, e2):
        context1 = get_entity_context(e1, ent_names_1, triples_1, rel_names_1)
        context2 = get_entity_context(e2, ent_names_2, triples_2, rel_names_2)
        return {'type': 'positive', 'kg1': context1, 'kg2': context2}

    def process_negative_pair(e1, e2):
        context1 = get_entity_context(e1, ent_names_1, triples_1, rel_names_1)
        context2 = get_entity_context(e2, ent_names_2, triples_2, rel_names_2)
        return {'type': 'negative', 'kg1': context1, 'kg2': context2}

    def openai_task(batch_start, batch_index):
        batch_pairs = list(new_pairs)[batch_start:batch_start + batch_size]
        batch_examples = []

        # Displaying a progress bar with tqdm
        with ThreadPoolExecutor(max_workers=100) as executor_results:
            # Submit the task and return the Future object
            futures = [executor_results.submit(process_positive_pair, pair[0], pair[1]) for pair in batch_pairs]

            # Manually updating the progress bar
            with tqdm(total=len(batch_pairs), desc=f"Processing Positive Pairs - Batch {batch_index}") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    batch_examples.append(result)
                    pbar.update(1)  # Updates the progress bar every time a task is completed

        # Process negative pairs with multi-threading
        negative_batch = list(negative_pairs)[batch_start:batch_start + batch_size]
        with ThreadPoolExecutor(max_workers=30) as executor_results:
            # Submit the task and return the Future object
            futures = [executor_results.submit(process_negative_pair, pair[0], pair[1]) for pair in negative_batch]

            # Manually updating the progress bar
            with tqdm(total=len(negative_batch), desc=f"Processing Negative Pairs - Batch {batch_index}") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    batch_examples.append(result)
                    pbar.update(1)  # Updates the progress bar every time a task is completed

        # Randomly select 3 key examples from the batch
        selected_examples = random.sample(batch_examples, min(3, len(batch_examples)))

        LLM2_Agent_Profile = '''
        Goal: As a knowledge graph alignment expert, analyze these entity pairs and generate logical rules that capture the patterns of alignment and non-alignment.
        Constraint: Focus on extracting generalizable patterns.
        '''

        if ablation_config or no_optimization_tool:
            if ablation_config[0] == 'ablation5' and ablation_config[1] == 'Multi-Granularity':
                prompt = """\nBased on this information, some experience insights are generated. one per line："""
                for example in selected_examples:
                    prompt += f"\n{'Aligned' if example['type'] == 'positive' else 'Non-aligned'} pair:\n"
                    prompt += f"KG1 Entity: {example['kg1']['name']}\n"
                    prompt += "Relations:\n" + "\n".join(f"- {r}" for r in example['kg1']['relations']) + "\n"
                    prompt += f"KG2 Entity: {example['kg2']['name']}\n"
                    prompt += "Relations:\n" + "\n".join(f"- {r}" for r in example['kg2']['relations']) + "\n"

            elif ablation_config[0] == 'ablation5' and ablation_config[1] == 'Communication':
                prompt = """\nBased on this information, some experience insights are generated："""
                for example in selected_examples:
                    prompt += f"\n{'Aligned' if example['type'] == 'positive' else 'Non-aligned'} pair:\n"
                    prompt += f"KG1 Entity: {example['kg1']['name']}\n"
                    prompt += "Relations:\n" + "\n".join(f"- {r}" for r in example['kg1']['relations']) + "\n"
                    prompt += f"KG2 Entity: {example['kg2']['name']}\n"
                    prompt += "Relations:\n" + "\n".join(f"- {r}" for r in example['kg2']['relations']) + "\n"

                prompt += """\nBased on these examples:
                                            ∃x,y(name(x, "Barack Obama") ∧ name(y, "Barack Obama") ∧ role(x, "President") ∧ role(y, "President") ∧ time_period(x, "2009-2017") ∧ time_period(y, "2009-2017")) ⟹ x = y

                                            ∃x,y(name(x, "John Smith") ∧ name(y, "John Smith") ∧ organization(x, "Company A") ∧ organization(y, "Company B")) ⟹ x ≠ y

                                          """
            elif no_optimization_tool:
                prompt = """\nSome experience insights are generated. As much as possible："""
                for example in selected_examples:
                    prompt += f"\n{'Aligned' if example['type'] == 'positive' else 'Non-aligned'} pair:\n"
                    prompt += f"KG1 Entity: {example['kg1']['name']}\n"
                    prompt += "Relations:\n" + "\n".join(f"- {r}" for r in example['kg1']['relations']) + "\n"
                    prompt += f"KG2 Entity: {example['kg2']['name']}\n"
                    prompt += "Relations:\n" + "\n".join(f"- {r}" for r in example['kg2']['relations']) + "\n"

            else:
                prompt = LLM2_Agent_Profile + """
                            Examples to analyze:

                            """
                for example in selected_examples:
                    prompt += f"\n{'Aligned' if example['type'] == 'positive' else 'Non-aligned'} pair:\n"
                    prompt += f"KG1 Entity: {example['kg1']['name']}\n"
                    prompt += "Relations:\n" + "\n".join(f"- {r}" for r in example['kg1']['relations']) + "\n"
                    prompt += f"KG2 Entity: {example['kg2']['name']}\n"
                    prompt += "Relations:\n" + "\n".join(f"- {r}" for r in example['kg2']['relations']) + "\n"

                prompt += """\nBased on these examples, generate positive/negative logical rules in the following format:


                            ∃x,y(name(x, "Barack Obama") ∧ name(y, "Barack Obama") ∧ role(x, "President") ∧ role(y, "President") ∧ time_period(x, "2009-2017") ∧ time_period(y, "2009-2017")) ⟹ x = y

                            ∃x,y(name(x, "John Smith") ∧ name(y, "John Smith") ∧ organization(x, "Company A") ∧ organization(y, "Company B")) ⟹ x ≠ y


                            Generate only the rules, one per line:"""

        else:
            prompt = LLM2_Agent_Profile + """
            Examples to analyze:

            """
            for example in selected_examples:
                prompt += f"\n{'Aligned' if example['type'] == 'positive' else 'Non-aligned'} pair:\n"
                prompt += f"KG1 Entity: {example['kg1']['name']}\n"
                prompt += "Relations:\n" + "\n".join(f"- {r}" for r in example['kg1']['relations']) + "\n"
                prompt += f"KG2 Entity: {example['kg2']['name']}\n"
                prompt += "Relations:\n" + "\n".join(f"- {r}" for r in example['kg2']['relations']) + "\n"

            prompt += """\nBased on these examples, generate positive/negative logical rules in the following format:


            ∃x,y(name(x, "Barack Obama") ∧ name(y, "Barack Obama") ∧ role(x, "President") ∧ role(y, "President") ∧ time_period(x, "2009-2017") ∧ time_period(y, "2009-2017")) ⟹ x = y

            ∃x,y(name(x, "John Smith") ∧ name(y, "John Smith") ∧ organization(x, "Company A") ∧ organization(y, "Company B")) ⟹ x ≠ y


            Generate only the rules, one per line:"""

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",  # "gpt-4-0125-preview"
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.7
            )

            # print(prompt)
            rules = response.choices[0].message.content.strip().split('\n')
            print(rules)
            tokens_cal.update_add_var(response.usage.total_tokens)  # 更新tokens
            all_rules.extend([r for r in rules if r.strip()])
        except Exception as e:
            print(f"Error processing batch: {str(e)}")

    # Process positive pairs
    for batch_index, batch_start in tqdm(enumerate(range(0, len(new_pairs), batch_size)), desc="Processing Batches"):
        executor.submit(openai_task, batch_start, batch_index)

    executor.shutdown(wait=True)

    if all_rules:
        output_file = LLM2_PRIVATE_MESSAGE_POOL['alignment_rules']
        with open(output_file, 'a+', encoding='utf-8') as f:
            for rule in all_rules:
                f.write(rule + '\n')

    return all_rules


def copy_aligned_pairs(data_dir, S4_PRIVATE_MESSAGE_POOL={}):
    """
    Copy aligned entity pairs from execution_alignment_pairs.txt to sup_pairs file

    Args:
        data_dir (str): Base directory containing the data folders

    Returns:
        bool: True if successful, False otherwise
    """

    try:
        # Get dataset name from data_dir (last folder name)
        dataset = os.path.basename(data_dir)

        # Construct full paths
        input_file = S4_PRIVATE_MESSAGE_POOL['execution_alignment_pairs']
        output_file = os.path.join(data_dir, 'sup_pairs')

        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"Error: Input file not found at {input_file}")
            return False

        # Create parent directory for output file if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Copy the content
        with open(input_file, 'r', encoding='utf-8') as infile, \
                open(output_file, 'w', encoding='utf-8') as outfile:
            outfile.write(infile.read())

        print(f"Successfully copied aligned pairs from {input_file} to {output_file}")
        return True

    except Exception as e:
        print(f"Error copying aligned pairs: {str(e)}")
        return False

def run_full_process_s4(data_dir, model_name):

    S4_PRIVATE_MESSAGE_POOL = {'sup_pairs': os.path.join(data_dir, "message_pool", "sup_pairs"),
                               'hypergraph_neural_top_pairs': os.path.join(data_dir, "message_pool",
                                                                           "hypergraph_neural_top_pairs.txt"),
                               'execution_alignment_pairs': os.path.join(data_dir, "message_pool", "execution_alignment_pairs.txt"),
                               }
    # copy_aligned_pairs(data_dir, S4_PRIVATE_MESSAGE_POOL)
    if model_name == "Simple-HHEA":
        return run_simple_hhea(data_dir, S4_PRIVATE_MESSAGE_POOL)
    elif model_name == "MGTEA":
        return run_other_model1(data_dir, S4_PRIVATE_MESSAGE_POOL)
    elif model_name == "BERT-INT":
        return run_other_model2(data_dir, S4_PRIVATE_MESSAGE_POOL)
    else:
        print(f"Unknown model: {model_name}")
        return False


def run_simple_hhea(data_dir, S4_PRIVATE_MESSAGE_POOL={}):
    cuda = 1
    lr = 0.01
    wd = 0.001
    gamma = 1.0
    epochs = 500  # 1500
    lang = os.path.basename(data_dir)

    commands = [
        f"python ./on_demand_agentic_hypergraph_collaboration/s4/Simple-HHEA/process_name_embedding.py --data {lang}",
        f"python ./on_demand_agentic_hypergraph_collaboration/s4/Simple-HHEA/feature_perprocessing/preprocess.py --l {lang}",
        f"python ./on_demand_agentic_hypergraph_collaboration/s4/Simple-HHEA/feature_perprocessing/longterm/main.py --input 'data/{lang}/deepwalk.data' --output 'data/{lang}/longterm.vec' --node2rel 'data/{lang}/node2rel' --q 0.7",
        f"python ./on_demand_agentic_hypergraph_collaboration/s4/Simple-HHEA/feature_perprocessing/get_deep_emb.py --path 'data/{lang}/'",
        f"python ./on_demand_agentic_hypergraph_collaboration/s4/Simple-HHEA/main_SimpleHHEA.py --data {lang} --cuda {cuda} --lr {lr} --wd {wd} --gamma {gamma} --epochs {epochs} --noise_ratio 0.0"
    ]

    try:
        for cmd in commands:
            print(f"\nExecuting: {cmd}")
            result = subprocess.run(cmd, shell=True, check=True, text=True)
            if result.returncode != 0:
                print(f"Error executing command: {cmd}")
                return False
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e.cmd}")
        print(f"Error output: {e.output}")
        return False


def run_other_model1(data_dir, S4_PRIVATE_MESSAGE_POOL={}):
    # MGTEA

    # dataset: BETA
    lang = os.path.basename(data_dir)

    ds = 1  # dataset
    if lang == 'BETA':
        ds = 1
    commands = [
        f"python ./s4/MGTEA/main.py --ds {ds}",
    ]

    try:
        for cmd in commands:
            print(f"\nExecuting: {cmd}")
            result = subprocess.run(cmd, shell=True, check=True, text=True)
            if result.returncode != 0:
                print(f"Error executing command: {cmd}")
                return False
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e.cmd}")
        print(f"Error output: {e.output}")
        return False


def run_other_model2(data_dir, S4_PRIVATE_MESSAGE_POOL={}):
    # BERT-INT

    # Note that basic_bert_unit/Param.py and interaction_model/Param.py is the config file. including the dataset
    commands = [
        f"python ./on_demand_agentic_hypergraph_collaboration/s4/BERT-INT/basic_bert_unit/main.py",
        f"python ./on_demand_agentic_hypergraph_collaboration/s4/BERT-INT/basic_bert_unit/interaction_model/clean_attribute_data.py",
        f"python ./on_demand_agentic_hypergraph_collaboration/s4/BERT-INT/basic_bert_unit/get_entity_embedding.py",
        f"python ./on_demand_agentic_hypergraph_collaboration/s4/BERT-INT/basic_bert_unit/get_attributeValue_embedding.py",
        f"python ./on_demand_agentic_hypergraph_collaboration/s4/BERT-INT/basic_bert_unit/get_neighView_and_desView_interaction_feature.py",
        f"python ./on_demand_agentic_hypergraph_collaboration/s4/BERT-INT/basic_bert_unit/get_attributeView_interaction_feature.py",
        f"python ./on_demand_agentic_hypergraph_collaboration/s4/BERT-INT/basic_bert_unit/interaction_model.py",
    ]

    # Note that interaction_model/Param.py is the config file. including the dataset
    try:
        for cmd in commands:
            print(f"\nExecuting: {cmd}")
            result = subprocess.run(cmd, shell=True, check=True, text=True)
            if result.returncode != 0:
                print(f"Error executing command: {cmd}")
                return False
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e.cmd}")
        print(f"Error output: {e.output}")
        return False
    pass


def run_other_model3(data_dir, S4_PRIVATE_MESSAGE_POOL={}):
    # any others
    pass

class MetaEvaluation:
    def __init__(self, data_dir: str = "./"):
        """
        Implements Step ⑥: Meta Evaluation (Section III-C.4)
        - Input: s' (new state from Step ⑤), g (goal)
        - Output: r (reward), f_fb (feedback), F_meta (meta-fusion signal)
        - 公式: r = R(s', g) (Eq. 9), F_meta = φ(H_evo, r, f_fb) (Eq. 10)
        Key Function: Evaluate if new state s' aligns with goal g, and decide
        whether to continue iteration or terminate.
        """
        self.data_dir = data_dir
        self.execution_alignment_path = 'message_pool/execution_alignment_pairs.txt'
        self.train_alignment_path = 'sup_pairs'
        self.updated_train_path = 'sup_pairs'

    def _load_alignment_pairs(self, filepath: str) -> Set[Tuple[int, int]]:
        pairs = set()

        if not os.path.exists(filepath):
            print(f"err: {filepath} not exit")
            return pairs

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    kg1_id = int(parts[0])
                    kg2_id = int(parts[1])
                    pairs.add((kg1_id, kg2_id))

        return pairs

    def _save_alignment_pairs(self, filepath: str, pairs: Set[Tuple[int, int]]):
        """save alignment to file (save after sorting)"""
        sorted_pairs = sorted(list(pairs))

        with open(filepath, 'w', encoding='utf-8') as f:
            for kg1, kg2 in sorted_pairs:
                f.write(f"{kg1}\t{kg2}\n")

    def _merge_alignment_pairs(self,
                               train_pairs: Set[Tuple[int, int]],
                               execution_pairs: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        """
        Merge training set alignment and execution alignment
        """
        train_dict = {}
        for kg1, kg2 in train_pairs:
            train_dict[kg1] = kg2

        execution_dict = {}
        for kg1, kg2 in execution_pairs:
            execution_dict[kg1] = kg2


        merged_dict = train_dict.copy()
        merged_dict.update(execution_dict)

        merged_pairs = set([(kg1, kg2) for kg1, kg2 in merged_dict.items()])

        return merged_pairs

    def _run_simple_hhea(self, train_file: str):
        """
        run Simple-HHEA
        """
        print(f"{train_file} Run Simple-HHEA...")
        run_full_process_s4(self.data_dir, "Simple-HHEA")

    def evaluate(self) -> Tuple[Set[Tuple[int, int]], bool]:  # Change this line
        """
        Evaluate alignment quality and determine if iteration should continue.
        Args:
            state: New state s' after Step ⑤ execution
            goal: Target goal g
        Returns:
            (reward, should_continue):
            - reward r ∈ {0, 1}: r=1 if goal achieved, r=0 otherwise
            - should_continue: True if need to return to Step ③
        """
        print("\n" + "=" * 50)
        print("Start meta assessment...")

        # Step 1: load execution_alignment_pairs
        execution_pairs = self._load_alignment_pairs(
            os.path.join(self.data_dir, self.execution_alignment_path)
        )
        print(f"Load execution_alignment_pairs: {len(execution_pairs)} 对")

        # Key judgment: terminate if there is no new execution pairs
        if len(execution_pairs) == 0:
            print("ERR: execution_alignment_pairs Null, terminate iteration")
            print("=" * 50)
            return set(), False  # Returns an empty set and false to indicate that you do not want to continue

        # Step 2: load the original training set
        train_pairs = self._load_alignment_pairs(
            os.path.join(self.data_dir, self.train_alignment_path)
        )
        print(f"Load original training set: {len(train_pairs)} Pairs")

        # Step 3: merge and de duplicate
        merged_pairs = self._merge_alignment_pairs(train_pairs, execution_pairs)
        print(f"Combined training set: {len(merged_pairs)} Pairs")

        # Statistics new and updated alignment
        new_pairs = execution_pairs - train_pairs
        print(f"  New alignment: {len(new_pairs)} Pairs")

        #Statistics of replaced alignments
        execution_entities = set([kg1 for kg1, _ in execution_pairs])
        replaced_count = 0
        for kg1, kg2 in train_pairs:
            if kg1 in execution_entities:
                exec_kg2 = None
                for e_kg1, e_kg2 in execution_pairs:
                    if e_kg1 == kg1:
                        exec_kg2 = e_kg2
                        break
                if exec_kg2 is not None and exec_kg2 != kg2:
                    replaced_count += 1

        print(f"  Replace alignment: {replaced_count} Pairs")

        # Step 4: save the updated training set
        updated_train_file = os.path.join(self.data_dir, self.updated_train_path)
        self._save_alignment_pairs(updated_train_file, merged_pairs)
        print(f"The updated training set has been saved to: {updated_train_file}")

        original_train_file = os.path.join(self.data_dir, self.train_alignment_path)
        self._save_alignment_pairs(original_train_file, merged_pairs)
        print(f"Original training set updated: {original_train_file}")


        print("\nReady to run simple HHEA...")
        self._run_simple_hhea(original_train_file)

        print("\nFinish！")
        print("=" * 50)

        # Returns the merged pairs and true, indicating that the iteration can continue
        return merged_pairs, True


def main(data_dir = "/home/dex/Desktop/entity_sy/MTKGA-Wild/data/icews_wiki/"):
    "" "sample usage" ""
    # Initialize meta evaluation module
    evaluator = MetaEvaluation(data_dir="./")


    updated_train_pairs, is_continue = evaluator.evaluate()

    if is_continue:
        rules = run_full_process_llm2(data_dir)
        print(f"Generated {len(rules)} alignment rules.")
    print(f"\nthe final training set contains {len(updated_train_pairs)} alignment")

    return updated_train_pairs


if __name__ == "__main__":
    updated_train_pairs = main()