"""
Core Module Selection Module
Function: Detects TOP-1 pre-alignment conflicts between four modality layers (text, image, audio, video) and selects Agent layers to activate
"""
from dataclasses import dataclass
import os
from typing import List, Dict, Set, Tuple
import subprocess

@dataclass
class HypergraphState:
    """
    Represents the state s ∈ S in the MDP framework (Eq. 4).
    In practice, state s is encoded as:
    - Alignment files for each modality layer (text_H, image_H, audio_H, video_H)
    - Current TOP-1 candidates stored in projection_pairs_merged.txt
    "state s represents the current alignment status"
    """
    data_dir: str  # Root directory containing modality layers
    modality_files: Dict[str, str]  # Mapping: modality -> file path
    def __post_init__(self):
        self.modality_files = {
            'text': f"{self.data_dir}/text_H/projection_pairs_merged.txt",
            'image': f"{self.data_dir}/image_H/projection_pairs_merged.txt",
            'audio': f"{self.data_dir}/audio_H/projection_pairs_merged.txt",
            'video': f"{self.data_dir}/video_H/projection_pairs_merged.txt"
        }

@dataclass
class Goal:
    """
    Represents the goal g ∈ O in the MDP framework (Eq. 4).
    Goal: Achieve conflict-free alignment across all modality layers.
    "goal g is to resolve all conflicts in H_evo"
    """
    description: str = "Conflict-free multi-modal alignment"
    def is_achieved(self, conflict_count: int) -> bool:
        """Check if goal is achieved (no conflicts remaining)"""
        return conflict_count == 0


def run_full_process_s4(data_dir, model_name):
    '''
    Goal: S4 quickly computes the similarity between entities using information about the current aligned entity pairs, the series structure of the knowledge graph, and other semantic information.
    Constraint: Output the matrix of similarity between entities in the knowledge graph.
    '''

    S4_PRIVATE_MESSAGE_POOL = {'hypergraph_neural_top_pairs': os.path.join(data_dir, "message_pool", "hypergraph_neural_top_pairs.txt")}

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


class CoreModuleSelector:
    def __init__(self, base_dir: str = "."):
        """
        Implements Step ③: Core Block Selection (Section III-C.1)
        - Input: Current state s and goal g
        - Output: B(s,g) = {B₁, B₂, ..., Bₙ} (activated agentic blocks)
        - B(s,g) = {Bᵢ | ∃e^s ∈ E^s: inconsistency(e^s, L) > 0}
        Implementation: Analyzes TOP-1 conflicts across modality layers to determine
        which agents (modality-specific blocks) need to be activated.
        """



        self.base_dir = base_dir
        self.modalities = {
            'text_H': 'message_pool/text_temporal_H/projection_pairs_merged.txt',
            'image_H': 'message_pool/image_temporal_H/projection_pairs_merged.txt',
            'audio_H': 'message_pool/audio_temporal_H/projection_pairs_merged.txt',
            'video_H': 'message_pool/video_temporal_H/projection_pairs_merged.txt'
        }
        self.hypergraph_file = 'message_pool/hypergraph_neural_top_pairs.txt'

    def load_alignment_pairs(self, file_path: str) -> Dict[str, str]:
        """
        Load alignment entity pair file (only read TOP-1)

        Args:
            file_path: File path

        Returns:
            Dict[KG1 entity id, KG2 entity id]: TOP-1 alignment for each KG1 entity
        """
        alignments = {}
        full_path = os.path.join(self.base_dir, file_path)

        if not os.path.exists(full_path):
            print(f"Warning: File does not exist {full_path}")
            return alignments

        with open(full_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    kg1_id, kg2_id = parts[0], parts[1]
                    # Only keep the first alignment (TOP-1) for each KG1 entity
                    if kg1_id not in alignments:
                        alignments[kg1_id] = kg2_id

        return alignments

    def load_all_alignment_pairs(self, file_path: str) -> List[Tuple[str, str]]:
        """
        Load alignment entity pair file (read all alignment pairs)

        Args:
            file_path: File path

        Returns:
            List[(KG1 entity id, KG2 entity id)]: All alignment pairs list (maintain original order)
        """
        alignments = []
        full_path = os.path.join(self.base_dir, file_path)

        if not os.path.exists(full_path):
            print(f"Warning: File does not exist {full_path}")
            return alignments

        with open(full_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    kg1_id, kg2_id = parts[0], parts[1]
                    alignments.append((kg1_id, kg2_id))

        return alignments

    def initialize_modality_files(self):
        """
        Initialize four modality layer files

        Steps:
        1. Call Simple-HHEA to generate hypergraph_neural_top_pairs.txt
        2. For Text layer only: Place pre-aligned entity pairs at TOP-1 position for the same source entity
        3. For all layers: Remove all alignment pairs of source entities not appearing in pre-alignment
        """
        print("=" * 60)
        print("Starting initialization process...")
        print("=" * 60)

        # Step 1: Call Simple-HHEA (example code, needs to be replaced with actual call)

        print("\nStep 1: Calling Simple-HHEA for fast computation...")
        run_full_process_s4(self.base_dir, "Simple-HHEA")
        print("Simple-HHEA computation completed, generated hypergraph_neural_top_pairs.txt")

        # Step 2: Load pre-aligned entity pairs
        print("\nStep 2: Loading pre-aligned entity pairs...")
        hypergraph_pairs = self.load_alignment_pairs(self.hypergraph_file)
        print(f"Loaded {len(hypergraph_pairs)} pre-aligned entity pairs")

        valid_kg1_entities = set(hypergraph_pairs.keys())

        # Step 3: Process four modality layer files
        print("\nStep 3: Processing four modality layer files...")
        for modality_name, file_path in self.modalities.items():
            full_path = os.path.join(self.base_dir, file_path)

            if not os.path.exists(full_path):
                print(f"  {modality_name} layer file does not exist, skipping")
                continue

            # Read all alignment pairs from the original file
            all_pairs = self.load_all_alignment_pairs(file_path)

            # Group by KG1 entity
            entity_groups = {}  # {KG1 entity id: [(KG1 entity id, KG2 entity id), ...]}
            for kg1_id, kg2_id in all_pairs:
                if kg1_id not in entity_groups:
                    entity_groups[kg1_id] = []
                entity_groups[kg1_id].append((kg1_id, kg2_id))

            # Reorganize alignment pairs
            new_pairs = []
            removed_entities = 0
            modified_entities = 0

            for kg1_id in entity_groups:
                # Only keep source entities that appear in pre-alignment
                if kg1_id not in valid_kg1_entities:
                    removed_entities += 1
                    continue

                # Get all alignment pairs for this entity
                pairs = entity_groups[kg1_id]

                # Only adjust TOP-1 for Text layer
                if modality_name == 'text_H':
                    # Text layer: Place pre-aligned KG2 entity at first position (TOP-1 position)
                    if kg1_id in hypergraph_pairs:
                        pre_aligned_kg2 = hypergraph_pairs[kg1_id]

                        # Check if pre-aligned KG2 entity already exists
                        existing_kg2_ids = [kg2 for _, kg2 in pairs]

                        if pre_aligned_kg2 not in existing_kg2_ids:
                            # If not exists, insert at first position
                            new_pairs.append((kg1_id, pre_aligned_kg2))
                            modified_entities += 1
                        else:
                            # If exists, remove it and insert at first position
                            pairs = [(k1, k2) for k1, k2 in pairs if k2 != pre_aligned_kg2]
                            new_pairs.append((kg1_id, pre_aligned_kg2))
                            modified_entities += 1

                        # Add other alignment pairs
                        new_pairs.extend(pairs)
                    else:
                        # No pre-alignment, keep original alignment pairs
                        new_pairs.extend(pairs)
                else:
                    # Other layers (image_H, audio_H, video_H): Don't adjust TOP-1, maintain original order
                    new_pairs.extend(pairs)

            # Overwrite and save
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w', encoding='utf-8') as f:
                for kg1_id, kg2_id in new_pairs:
                    f.write(f"{kg1_id}\t{kg2_id}\n")

            print(f"  {modality_name} layer:")
            print(f"    - Original alignment pairs: {len(all_pairs)}")
            print(f"    - Retained alignment pairs: {len(new_pairs)}")
            print(f"    - Removed source entities: {removed_entities}")
            if modality_name == 'text_H':
                print(f"    - Modified TOP-1 entities: {modified_entities}")
            else:
                print(f"    - Maintained original TOP-1 order")

        print("\nInitialization completed!")
        print("=" * 60)

    def detect_conflicts(self) -> Tuple[List[int], Dict[str, Set[str]]]:
        """
        Implements inconsistency detection (Eq. 6):
        inconsistency(e^s, L) = Σ_{l∈L} 1_{top1_l(e^s) ≠ top1_mode(e^s)}
        Returns:
            (activated_agents, conflict_count)
        """
        print("\nDetecting TOP-1 alignment conflicts across four modality layers...")

        # Load TOP-1 alignments for all modalities
        modality_alignments = {}
        for modality_name, file_path in self.modalities.items():
            alignments = self.load_alignment_pairs(file_path)
            modality_alignments[modality_name] = alignments
            print(f"  {modality_name} layer: {len(alignments)} TOP-1 alignments")

        # Collect all KG1 entities
        all_kg1_entities = set()
        for alignments in modality_alignments.values():
            all_kg1_entities.update(alignments.keys())

        # Detect conflicts
        conflicts = {}  # {KG1 entity id: {modality layer name}}
        conflict_modalities = set()  # Modality layers with conflicts

        for kg1_id in all_kg1_entities:
            kg2_mappings = {}  # {modality layer: KG2 entity id}

            for modality_name, alignments in modality_alignments.items():
                if kg1_id in alignments:
                    kg2_mappings[modality_name] = alignments[kg1_id]

            # Check for conflicts (different layers map to different KG2 entities)
            if len(kg2_mappings) > 1:
                kg2_values = set(kg2_mappings.values())
                if len(kg2_values) > 1:  # Conflict exists
                    conflicts[kg1_id] = set(kg2_mappings.keys())
                    conflict_modalities.update(kg2_mappings.keys())

        # Generate activated layer list
        modality_order = ['text_H', 'image_H', 'audio_H', 'video_H']
        activated_agents = [1 if m in conflict_modalities else 0 for m in modality_order]

        print(f"\nDetection results:")  # Meta-agent hypergraph is a judgment model
        print(f"  Found {len(conflicts)} conflicting entities")
        print(f"  Activated Agent layers: {modality_order} -> {activated_agents}")

        return activated_agents, conflicts

    def run(self, do_initialization: bool = False) -> List[int]:
        """
               Run core module selection

               Args:
                   do_initialization: Whether to execute initialization process

               Returns:
                   Activated Agent layer list: [text_H, image_H, audio_H, video_H] -> [1/0, 1/0, 1/0, 1/0]
               """
        if do_initialization:
            self.initialize_modality_files()

        activated_agents, conflicts = self.detect_conflicts()
        return activated_agents


if __name__ == "__main__":
    # Test code
    selector = CoreModuleSelector(base_dir=".")

    # Set to True on first run for initialization
    activated = selector.run(do_initialization=False)

    print(f"\nFinal output: {activated}")