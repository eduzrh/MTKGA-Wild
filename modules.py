import os
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
import subprocess


from collections import defaultdict
from Area2.LLM1_label_selector import align_entities
from Area2.LLM2_rule_generator import run_full_process_llm2
from Area3.LLM3_instruction_generator import run_full_process_llm3
from Area3.LLM4_5_expert_system import run_full_process_llm4_5



def copy_aligned_pairs(data_dir, S4_PRIVATE_MESSAGE_POOL = {}):
    """
    Copy aligned entity pairs from aligned_entities.txt to sup_pairs file
    
    Args:
        data_dir (str): Base directory containing the data folders
        
    Returns:
        bool: True if successful, False otherwise
    """

    try:
        # Get dataset name from data_dir (last folder name)
        dataset = os.path.basename(data_dir)
        
        # Construct full paths
        input_file = S4_PRIVATE_MESSAGE_POOL['aligned_entities']
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
    S4_Agent_Profile = '''
    Goal: S4 quickly computes the similarity between entities using information about the current aligned entity pairs, the series structure of the knowledge graph, and other semantic information.
    Constraint: Output the matrix of similarity between entities in the knowledge graph.
    '''

    S4_PRIVATE_MESSAGE_POOL = {'sup_pairs': os.path.join(data_dir, "message_pool", "sup_pairs"),
                               'hypergraph_neural_top_pairs': os.path.join(data_dir, "message_pool", "hypergraph_neural_top_pairs.txt"),
                               'aligned_entities': os.path.join(data_dir, "message_pool", "aligned_entities.txt"),
                               }
    copy_aligned_pairs(data_dir, S4_PRIVATE_MESSAGE_POOL)
    if model_name == "Simple-HHEA":
        return run_simple_hhea(data_dir, S4_PRIVATE_MESSAGE_POOL)
    elif model_name == "MGTEA":
        return run_other_model1(data_dir, S4_PRIVATE_MESSAGE_POOL)
    elif model_name == "BERT-INT":
        return run_other_model2(data_dir, S4_PRIVATE_MESSAGE_POOL)
    else:
        print(f"Unknown model: {model_name}")
        return False

def run_simple_hhea(data_dir, S4_PRIVATE_MESSAGE_POOL = {}):

    cuda = 1
    lr = 0.01
    wd = 0.001
    gamma = 1.0
    epochs = 500#1500
    lang = os.path.basename(data_dir)
    
    commands = [
        f"python ./s4/Simple-HHEA/process_name_embedding.py --data {lang}",
        f"python ./s4/Simple-HHEA/feature_perprocessing/preprocess.py --l {lang}",
        f"python ./s4/Simple-HHEA/feature_perprocessing/longterm/main.py --input 'data/{lang}/deepwalk.data' --output 'data/{lang}/longterm.vec' --node2rel 'data/{lang}/node2rel' --q 0.7",
        f"python ./s4/Simple-HHEA/feature_perprocessing/get_deep_emb.py --path 'data/{lang}/'",
        f"python ./s4/Simple-HHEA/main_SimpleHHEA.py --data {lang} --cuda {cuda} --lr {lr} --wd {wd} --gamma {gamma} --epochs {epochs} --noise_ratio 0.0"
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

def run_other_model1(data_dir, S4_PRIVATE_MESSAGE_POOL = {}):
    # MGTEA

    # dataset: BETA
    lang = os.path.basename(data_dir)

    ds = 1 #dataset
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


def run_other_model2(data_dir, S4_PRIVATE_MESSAGE_POOL = {}):
    # BERT-INT

    # Note that basic_bert_unit/Param.py and interaction_model/Param.py is the config file. including the dataset
    commands = [
        f"python ./s4/BERT-INT/basic_bert_unit/main.py",
        f"python ./s4/BERT-INT/basic_bert_unit/interaction_model/clean_attribute_data.py",
        f"python ./s4/BERT-INT/basic_bert_unit/get_entity_embedding.py",
        f"python ./s4/BERT-INT/basic_bert_unit/get_attributeValue_embedding.py",
        f"python ./s4/BERT-INT/basic_bert_unit/get_neighView_and_desView_interaction_feature.py",
        f"python ./s4/BERT-INT/basic_bert_unit/get_attributeView_interaction_feature.py",
        f"python ./s4/BERT-INT/basic_bert_unit/interaction_model.py",
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

def run_other_model3(data_dir, S4_PRIVATE_MESSAGE_POOL = {}):
    # any others
    pass





def role_assignment_and_knowledge_enhancement(data_dir, is_activation_m3 = True, ablation_config = None, no_optimization_tool = False):
    """
    Perform role assignment and knowledge enhancement
    - Based on alignment results, assign roles and enhance knowledge through expert models
    """
    print("Starting role assignment and knowledge enhancement...")


    run_full_process_llm3(data_dir, ablation_config = ablation_config, no_optimization_tool = no_optimization_tool)

    role_assignments_dir = os.path.join(data_dir, "role_assignments")

    run_full_process_llm4_5(data_dir, role_assignments_dir, ablation_config = ablation_config, no_optimization_tool = no_optimization_tool)

    print("Role assignment and knowledge enhancement process complete.")
    


def calculate_abs_hits_at_1(ref_path: str, pred_path: str) -> float:
    """
    Calculate the Hits@1 metric between two entity alignment files

    Parameters.
        ref_path: reference standard file path
        pred_path: path to the prediction result file

    Returns.
        Hits@1 evaluation value (floating point number)
    """

    def _load_pairs(file_path: str) -> set:
        """Internal function: load set of alignment pairs"""
        pairs = set()
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) == 2:
                    pairs.add((parts[0].strip(), parts[1].strip()))
        return pairs

    try:
        ref_pairs = _load_pairs(ref_path)
        pred_pairs = _load_pairs(pred_path)

        correct = len(ref_pairs & pred_pairs)
        total = len(ref_pairs)
        result = correct / total if total > 0 else 0.0
        print(f"Hits@1 = {result:.4f}")
        return correct / total if total > 0 else 0.0

    except FileNotFoundError as e:
        print(f"File Read Error: {e}")
        return 0.0