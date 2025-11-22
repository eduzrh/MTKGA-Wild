"""
Agent Hypergraph Execution Module
Based on LLM judgment results, execute conflict resolution at each activation layer
"""
from dataclasses import dataclass
import os
from typing import List, Dict, Set, Tuple
from .LLM_executor import align_entities



import os
from typing import List, Dict, Set, Tuple


class HypergraphAgentExecution:
    def __init__(self, data_dir: str = "./"):
        """
        Implements Step ⑤: Agentic Hypergraph Execution (Section III-C.3)
        - Input: D(s) (decision plan from Step ④)
        - Output: s' (new state), execution_alignment_pairs
        -  s' = T(s, D(s)) (Eq. 8)
        Key Function: Execute conflict resolution via LLM judgment and update
        hypergraph state accordingly.
        """
        self.data_dir = data_dir
        self.modalities = ['text_H', 'image_H', 'audio_H', 'video_H']
        self.modality_paths = {
            'text_H': 'message_pool/text_temporal_H/projection_pairs_merged.txt',
            'image_H': 'message_pool/image_temporal_H/projection_pairs_merged.txt',
            'audio_H': 'message_pool/audio_temporal_H/projection_pairs_merged.txt',
            'video_H': 'message_pool/video_temporal_H/projection_pairs_merged.txt'
        }
        self.conflict_path = 'message_pool/conflict.txt'
        self.execution_alignment_path = 'message_pool/execution_alignment_pairs.txt'

    def _load_conflict_pairs(self) -> List[Tuple[int, int]]:
        conflict_pairs = []
        filepath = os.path.join(self.data_dir, self.conflict_path)

        if not os.path.exists(filepath):
            print(f"Warning: {filepath} does not exist")
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

    def _call_llm_for_alignment(self) -> Dict[int, int]:
        """
        Call LLM for alignment judgment (placeholder function, replace with actual implementation)

        The LLM will read required files from self.data_dir:
        - conflict.txt: conflict alignment pairs
        - ent_ids_1, ent_ids_2: entity information
        - link files for each modality, etc.

        Returns:
            Dict[kg1_id, kg2_id]: alignment results judged by LLM
        """
        print("Calling LLM for alignment judgment...")
        print(f"  Passing data directory: {self.data_dir}")

        # Pass data_dir to LLM module, let it read conflict.txt and other required files

        execution_pair = align_entities(self.data_dir, is_first_time = True)

        # 示例：从execution_alignment_pairs.txt读取（假设已由LLM生成）
        execution_pairs = self._load_execution_alignment_pairs()
        print(f"LLM judgment completed, obtained {len(execution_pairs)} alignment pairs")

        return execution_pairs

    def _load_execution_alignment_pairs(self) -> Dict[int, int]:

        execution_pairs = {}
        filepath = os.path.join(self.data_dir, self.execution_alignment_path)

        if not os.path.exists(filepath):
            print(f"Warning: {filepath} does not exist")
            return execution_pairs

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    kg1_id = int(parts[0])
                    kg2_id = int(parts[1])
                    execution_pairs[kg1_id] = kg2_id

        return execution_pairs

    def _load_modality_pairs(self, filepath: str) -> List[Tuple[int, int]]:
        pairs = []

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
                    pairs.append((kg1_id, kg2_id))

        return pairs

    def _update_modality_pairs(self,
                               original_pairs: List[Tuple[int, int]],
                               execution_pairs: Dict[int, int]) -> List[Tuple[int, int]]:
        """
        Update modality layer alignment pairs: place alignments from execution_pairs at TOP-1 position

        Args:
            original_pairs: original alignment pair list
            execution_pairs: alignment results judged by LLM

        Returns:
            updated alignment pair list
        """
        # Group by KG1 entities
        grouped = {}
        for kg1, kg2 in original_pairs:
            if kg1 not in grouped:
                grouped[kg1] = []
            grouped[kg1].append(kg2)

        # Reorder and deduplicate. The overall logic here is an operation on the hypergraph,
        # essentially moving towards eliminating conflicts. Specifically, it is achieved by
        # replacing core entity pairs within the hypergraph to adjust hyperedges, update
        # projections, or reweight connections to modify the hypergraph structure
        updated_pairs = []
        processed_entities = set()

        for kg1_entity in sorted(set([kg1 for kg1, _ in original_pairs])):
            if kg1_entity in processed_entities:
                continue

            processed_entities.add(kg1_entity)
            kg2_list = grouped.get(kg1_entity, [])

            # If this entity is in execution_pairs, place it first
            if kg1_entity in execution_pairs:
                exec_kg2 = execution_pairs[kg1_entity]

                # First add execution alignment (TOP-1)
                updated_pairs.append((kg1_entity, exec_kg2))

                # Then add other alignments (remove duplicate exec_kg2)
                seen_kg2 = {exec_kg2}
                for kg2 in kg2_list:
                    if kg2 not in seen_kg2:
                        updated_pairs.append((kg1_entity, kg2))
                        seen_kg2.add(kg2)
            else:
                # No execution alignment, keep original order but deduplicate
                seen_kg2 = set()
                for kg2 in kg2_list:
                    if kg2 not in seen_kg2:
                        updated_pairs.append((kg1_entity, kg2))
                        seen_kg2.add(kg2)

        return updated_pairs

    def _save_modality_pairs(self, filepath: str, pairs: List[Tuple[int, int]]):
        with open(filepath, 'w', encoding='utf-8') as f:
            for kg1, kg2 in pairs:
                f.write(f"{kg1}\t{kg2}\n")

    def execute_conflict_resolution(self, activated_agents: List[int]) -> Dict[int, int]:
        """
        Execute decision plan D(s) to transition state s → s'.
        Args:
            decision_plan: D(s) from Step ④
            state: Current state s
            activated_agents: B(s,g) from Step ③
        Returns:
            execution_pairs: Dict mapping source_entity → aligned_target
            (saved to execution_alignment_pairs.txt)

        Implement state transition function T(s, D(s)) → s' (Eq. 8).
        Updates TOP-1 positions in modality-specific alignment files.

        """
        print("\n" + "=" * 50)
        print("Starting agent hypergraph execution...")

        # Find activated modalities
        activated_modalities = [
            self.modalities[i] for i in range(len(self.modalities))
            if activated_agents[i] == 1
        ]

        print(f"Activated modalities: {activated_modalities}")

        # Step 1: Read conflict alignment pairs
        conflict_pairs = self._load_conflict_pairs()
        print(f"Loaded {len(conflict_pairs)} conflict alignment pairs")

        if len(conflict_pairs) == 0:
            print("No conflicts to resolve")
            return {}

            # Step 2: Call LLM for judgment (pass data_dir instead of conflict_pairs)
        execution_pairs = self._call_llm_for_alignment()

        if len(execution_pairs) == 0:
            print("LLM did not return alignment results")
            return {}

        # Get source entities involved in execution_pairs
        execution_entities = set(execution_pairs.keys())
        print(f"Need to update alignments for {len(execution_entities)} source entities")

        # Step 3: Update alignment pairs for each activated modality
        for modality in activated_modalities:
            filepath = os.path.join(self.data_dir, self.modality_paths[modality])

            if not os.path.exists(filepath):
                print(f"Warning: {filepath} does not exist, skipping")
                continue

            # Load original alignment pairs
            original_pairs = self._load_modality_pairs(filepath)
            print(f"\nProcessing {modality} layer:")
            print(f"  Original alignment pair count: {len(original_pairs)}")

            # Update alignment pairs (place execution_pairs at TOP-1)
            updated_pairs = self._update_modality_pairs(original_pairs, execution_pairs)
            print(f"  Updated alignment pair count: {len(updated_pairs)}")

            # Count affected entities
            affected_entities = set([kg1 for kg1, _ in original_pairs]) & execution_entities
            print(f"  Number of affected entities: {len(affected_entities)}")

            # Save updated alignment pairs
            self._save_modality_pairs(filepath, updated_pairs)
            print(f"  Saved to {filepath}")

        print("\nAgent hypergraph execution completed!")
        print("=" * 50)

        return execution_pairs


def main():
    """Example usage"""
    # Initialize agent hypergraph execution module
    executor = HypergraphAgentExecution(data_dir="./")

    # Assume these agents are activated
    activated_agents = [1, 0, 0, 1]  # text and video are activated

    # Execute conflict resolution
    execution_pairs = executor.execute_conflict_resolution(activated_agents)

    print(f"\nExecution result: {len(execution_pairs)} alignment pairs confirmed")

    # Demonstrate how to actually call LLM module externally
    print("\nIn actual use, you should call the LLM module like this:")
    print(f"execution_pairs = llm_alignment_module.judge('{executor.data_dir}')")

    return execution_pairs


if __name__ == "__main__":
    execution_pairs = main()