"""
Core Module Selection Module
Function: Detects TOP-1 pre-alignment conflicts between four modality layers (text, image, audio, video) and selects Agent layers to activate
"""
from dataclasses import dataclass
import os
from typing import List, Dict, Set, Tuple

class CollaborativeDecision:
    """
        DecisionPlan
        Represents the decision plan D(s) in Eq. (7).
        D(s) = {(e^s, a_es) | e^s ∈ E^s}
        - e^s: Source entity with conflict
        - a_es: Alignment action (resolve conflict via LLM judgment)
        Implementation: Stored as conflict.txt, where each line is a candidate pair
        (e^s, e^t) requiring LLM verification.

    Implements Step ④: Collaboration Decision-Making

    - Input: B(s,g) (activated agents from Step ③)
    - Output: D(s) (decision plan containing conflict resolution actions)
    Key Function: Detect conflicts across modality layers and generate
    a decision plan for LLM-based conflict resolution.
    """
    def __init__(self, data_dir: str = "./"):

        self.data_dir = data_dir
        self.modalities = ['text_H', 'image_H', 'audio_H', 'video_H']
        self.modality_paths = {
            'text_H': 'message_pool/text_temporal_H/projection_pairs_merged.txt',
            'image_H': 'message_pool/image_temporal_H/projection_pairs_merged.txt',
            'audio_H': 'message_pool/audio_temporal_H/projection_pairs_merged.txt',
            'video_H': 'message_pool/video_temporal_H/projection_pairs_merged.txt'
        }
        self.conflict_output_path = 'message_pool/conflict.txt'


        self.ent_ids_1 = self._load_entity_info('ent_ids_1')
        self.ent_ids_2 = self._load_entity_info('ent_ids_2')

    def _load_entity_info(self, filename: str) -> Dict[int, str]:
        entity_info = {}
        filepath = os.path.join(self.data_dir, filename)

        if not os.path.exists(filepath):
            return entity_info

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    entity_id = int(parts[0])
                    entity_text = parts[1]
                    entity_info[entity_id] = entity_text

        return entity_info

    def _load_alignment_pairs_top1(self, filepath: str) -> Dict[int, int]:
        pairs = {}
        if not os.path.exists(filepath):
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

                    if kg1_id not in pairs:
                        pairs[kg1_id] = kg2_id

        return pairs

    def _load_all_alignment_pairs(self, filepath: str) -> Dict[int, List[int]]:
        pairs_dict = {}
        if not os.path.exists(filepath):
            return pairs_dict

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    kg1_id = int(parts[0])
                    kg2_id = int(parts[1])

                    if kg1_id not in pairs_dict:
                        pairs_dict[kg1_id] = []
                    pairs_dict[kg1_id].append(kg2_id)

        return pairs_dict

    def detect_and_record_conflicts(self, activated_agents: List[int]) -> Set[int]:
        """
        Detect conflicts and generate decision plan D(s).
        Args:
            activated_agents: B(s,g) from Step ③
            state: Current hypergraph state s
        Returns:
            D(s): Decision plan containing (entity, action) pairs
        """
        print("\n=" * 50)
        print("Starting collaborative decision-making...")

        # Find activated modalities
        activated_modalities = [
            self.modalities[i] for i in range(len(self.modalities))
            if activated_agents[i] == 1
        ]

        if len(activated_modalities) == 0:
            print("No activated agents, no need for collaborative decision-making")
            return set()

        print(f"Activated modalities: {activated_modalities}")

        # Load TOP-1 alignments of activated modalities (for conflict detection)
        modality_top1 = {}
        # Load all alignment pairs of activated modalities (for saving conflict details)
        modality_all_pairs = {}

        for modality in activated_modalities:
            filepath = os.path.join(self.data_dir, self.modality_paths[modality])
            if os.path.exists(filepath):
                modality_top1[modality] = self._load_alignment_pairs_top1(filepath)
                modality_all_pairs[modality] = self._load_all_alignment_pairs(filepath)
                print(f"Loaded {modality} layer: TOP-1 alignments for {len(modality_top1[modality])} entities")
            else:
                modality_top1[modality] = {}
                modality_all_pairs[modality] = {}
                print(f"Warning: {filepath} does not exist")

        # Find all KG1 entities
        all_kg1_entities = set()
        for pairs in modality_top1.values():
            all_kg1_entities.update(pairs.keys())

        print(f"Total of {len(all_kg1_entities)} KG1 entities to check")

        # Detect conflicts and collect all alignment pairs
        conflict_entities = set()
        conflict_pairs = []  # Store all alignment pairs of conflict entities
        conflict_details = []

        for kg1_entity in sorted(all_kg1_entities):
            # Get TOP-1 alignments of this entity in each activated modality
            kg2_top1_alignments = {}
            for modality in activated_modalities:
                if kg1_entity in modality_top1[modality]:
                    kg2_top1_alignments[modality] = modality_top1[modality][kg1_entity]

            # If only one or zero modalities have this entity, no conflict
            if len(kg2_top1_alignments) <= 1:
                continue

            # Check if all modalities align to the same KG2 entity
            kg2_values = list(kg2_top1_alignments.values())
            if len(set(kg2_values)) > 1:
                # Conflict detected! Collect all alignment pairs of this entity in all activated modalities
                conflict_entities.add(kg1_entity)

                # Get entity text information (if available)
                kg1_text = self.ent_ids_1.get(kg1_entity, "Unknown")

                detail = {
                    'kg1_entity': kg1_entity,
                    'kg1_text': kg1_text,
                    'alignments': {}
                }

                # Collect all alignment pairs of this conflict entity in all activated modalities
                all_kg2_for_this_entity = set()
                for modality in activated_modalities:
                    if kg1_entity in modality_all_pairs[modality]:
                        kg2_list = modality_all_pairs[modality][kg1_entity]
                        detail['alignments'][modality] = []

                        for kg2_id in kg2_list:
                            all_kg2_for_this_entity.add(kg2_id)
                            kg2_text = self.ent_ids_2.get(kg2_id, "Unknown")
                            detail['alignments'][modality].append({
                                'kg2_id': kg2_id,
                                'kg2_text': kg2_text
                            })

                # Add all alignment pairs of this entity to conflict_pairs
                for kg2_id in sorted(all_kg2_for_this_entity):
                    conflict_pairs.append((kg1_entity, kg2_id))

                conflict_details.append(detail)

        # Print conflict details
        print(f"\nFound {len(conflict_entities)} conflict entities:")
        for detail in conflict_details:
            print(f"\nKG1 entity {detail['kg1_entity']} ({detail['kg1_text']}):")
            for modality, align_list in detail['alignments'].items():
                print(f"  {modality} layer ({len(align_list)} alignments):")
                for align_info in align_list:
                    print(f"    -> KG2 entity {align_info['kg2_id']} ({align_info['kg2_text']})")

        # Save all conflict alignment pairs to file
        self._save_conflict_pairs(conflict_pairs)

        print(f"\nTotal of {len(conflict_pairs)} conflict alignment pairs saved to {self.conflict_output_path}")
        print("=" * 50)

        return conflict_entities

    def _save_conflict_pairs(self, conflict_pairs: List[Tuple[int, int]]):

        output_path = os.path.join(self.data_dir, self.conflict_output_path)

        with open(output_path, 'w', encoding='utf-8') as f:
            for kg1_entity, kg2_entity in conflict_pairs:
                f.write(f"{kg1_entity}\t{kg2_entity}\n")

    def load_conflict_pairs(self) -> List[Tuple[int, int]]:
        conflict_pairs = []
        filepath = os.path.join(self.data_dir, self.conflict_output_path)

        if not os.path.exists(filepath):
            return conflict_pairs

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    kg1_id = int(parts[0])
                    kg2_id = int(parts[1])
                    conflict_pairs.append((kg1_id, kg2_id))

        return conflict_pairs

    def get_conflict_entities(self) -> Set[int]:
        """Get all conflict KG1 entity IDs from conflict.txt"""
        conflict_pairs = self.load_conflict_pairs()
        conflict_entities = set([kg1 for kg1, kg2 in conflict_pairs])
        return conflict_entities



def run_collaborative_decision():
    """Example usage"""
    # Initialize collaborative decision module
    decision = CollaborativeDecision(data_dir="./")

    # Assume the activated Agent list obtained from the core module selector
    # For example: [1, 0, 0, 1] means text and video are activated
    activated_agents = [1, 0, 0, 1]

    # Detect and record conflicts (save all alignment pairs)
    conflict_entities = decision.detect_and_record_conflicts(activated_agents)

    print(f"\nNumber of conflict entities returned: {len(conflict_entities)}")

    # Can reload saved conflict pairs
    print("\nReloading conflict pairs:")
    loaded_pairs = decision.load_conflict_pairs()
    print(f"Loaded {len(loaded_pairs)} conflict alignment pairs")
    if len(loaded_pairs) > 0:
        print("First 5 pairs as examples:")
        for kg1, kg2 in loaded_pairs[:5]:
            print(f"  {kg1}\t{kg2}")

    return conflict_entities

if __name__ == "__main__":
    conflict_entities = run_collaborative_decision()