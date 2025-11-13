
import os
import time
import sys
import argparse
from collections import defaultdict


from hypergraph_representation.neural_retrieval import neural_retrieval
from hypergraph_representation.projection_and_hypergraph import AdaptiveEvolutionProjection, merge_batch_files
from on_demand_agentic_hypergraph_collaboration.core_selector import CoreModuleSelector
from on_demand_agentic_hypergraph_collaboration.collaborative_decision import CollaborativeDecision
from on_demand_agentic_hypergraph_collaboration.hypergraph_agent_execution import HypergraphAgentExecution
from on_demand_agentic_hypergraph_collaboration.meta_evaluation import MetaEvaluation, run_full_process_llm2
from on_demand_agentic_hypergraph_collaboration.LLM_executor import align_entities

import openai
import tokens_cal


os.environ["OPENAI_API_BASE"] = 'xxx'
os.environ["OPENAI_API_KEY"] = "xxx"
openai.api_key = os.getenv("OPENAI_API_KEY")

# ==================== Ablation Experiment Configuration ====================
ABLATION_CONFIGS = {
    'full': {  # Complete model
        'neural_retrieval': True,
        'adaptive_decoupling': True,
        'evolution_hypergraph': True,
        'core_block_selection': True,
        'collaborative_decision': True,
        'agentic_execution': True,
        'meta_evaluation': True,
        'description': 'Complete EvoWildAlign Framework'
    },
    'w/o_neural_retrieval': {
        'neural_retrieval': False,
        'adaptive_decoupling': True,
        'evolution_hypergraph': True,
        'core_block_selection': True,
        'collaborative_decision': True,
        'agentic_execution': True,
        'meta_evaluation': True,
        'description': 'Without Neural Retrieval'
    },
    'w/o_adaptive_decoupling': {
        'neural_retrieval': True,
        'adaptive_decoupling': False,
        'evolution_hypergraph': True,
        'core_block_selection': True,
        'collaborative_decision': True,
        'agentic_execution': True,
        'meta_evaluation': True,
        'description': 'Without Adaptive Decoupling'
    },
    'w/o_evolution_hypergraph': {
        'neural_retrieval': True,
        'adaptive_decoupling': True,
        'evolution_hypergraph': False,
        'core_block_selection': True,
        'collaborative_decision': True,
        'agentic_execution': True,
        'meta_evaluation': True,
        'description': 'Without Evolution Hypergraph Construction'
    },
    'w/o_core_block_selection': {
        'neural_retrieval': True,
        'adaptive_decoupling': True,
        'evolution_hypergraph': True,
        'core_block_selection': False,
        'collaborative_decision': True,
        'agentic_execution': True,
        'meta_evaluation': True,
        'description': 'Without Core Block Selection'
    },
    'w/o_collaborative_decision': {
        'neural_retrieval': True,
        'adaptive_decoupling': True,
        'evolution_hypergraph': True,
        'core_block_selection': True,
        'collaborative_decision': False,
        'agentic_execution': True,
        'meta_evaluation': True,
        'description': 'Without Collaborative Decision-Making'
    },
    'w/o_agentic_execution': {
        'neural_retrieval': True,
        'adaptive_decoupling': True,
        'evolution_hypergraph': True,
        'core_block_selection': True,
        'collaborative_decision': True,
        'agentic_execution': False,
        'meta_evaluation': True,
        'description': 'Without Agentic Hypergraph Execution'
    },
    'w/o_meta_evaluation': {
        'neural_retrieval': True,
        'adaptive_decoupling': True,
        'evolution_hypergraph': True,
        'core_block_selection': True,
        'collaborative_decision': True,
        'agentic_execution': True,
        'meta_evaluation': False,
        'description': 'Without Meta Evaluation'
    },
    'w/o_stage1': {  # Group 1: Remove entire Stage 1
        'neural_retrieval': False,
        'adaptive_decoupling': False,
        'evolution_hypergraph': False,
        'core_block_selection': True,
        'collaborative_decision': True,
        'agentic_execution': True,
        'meta_evaluation': True,
        'description': 'Without Neuro-symbolic Evolution Hypergraph Representation (Stage 1)'
    },
    'w/o_stage2': {  # Group 1: Remove entire Stage 2
        'neural_retrieval': True,
        'adaptive_decoupling': True,
        'evolution_hypergraph': True,
        'core_block_selection': False,
        'collaborative_decision': False,
        'agentic_execution': False,
        'meta_evaluation': False,
        'description': 'Without On-demand Agentic Hypergraph Collaboration (Stage 2)'
    }
}

def hypergraph_representation(data_dir, ablation_config):
    """
    Stage 1: Hypergraph Representation (with ablation support)
    """
    print("=" * 80)
    print("Stage 1: Neuro-symbolic Evolution Hypergraph Representation")
    print(f"Config: {ablation_config['description']}")
    print("=" * 80)

    # Step 1.1: Neural Retrieval
    if ablation_config['neural_retrieval']:
        print("\n[Step 1.1] Neural Retrieval")
        neural_retrieval(data_dir)
    else:
        print("\n[Step 1.1] Neural Retrieval - SKIPPED (Ablation)")

        create_dummy_retrieval_output(data_dir)

    # Step 1.2: Adaptive Decoupling & Evolution Hypergraph
    if ablation_config['adaptive_decoupling'] and ablation_config['evolution_hypergraph']:
        print("\n[Step 1.2] Adaptive Symbolic Decoupling")
        processor = AdaptiveEvolutionProjection(data_dir, data_dir+"/message_pool")
        processor.load_data()
        processor.project_and_save(batch_size=5000)

        print("\n[Step 2] Evolution Hypergraph Construction")
        merge_batch_files(data_dir+"/message_pool")
    elif ablation_config['adaptive_decoupling']:
        print("\n[Step 1.2] Adaptive Decoupling Only - Without Evolution Hypergraph")
        # Only perform decoupling, don't construct hypergraph
        processor = AdaptiveEvolutionProjection(data_dir, data_dir+"/message_pool")
        processor.load_data()
        # Simplified processing
    elif ablation_config['evolution_hypergraph']:
        print("\n[Step 2] Evolution Hypergraph Only - Without Adaptive Decoupling")
        # Directly construct hypergraph, skip decoupling
        merge_batch_files(data_dir+"/message_pool")
    else:
        print("\n[Stage 1] SKIPPED - Both components disabled")

def run_collaboration_stage(data_dir, ablation_config, current_iteration):
    """
    Stage 2: Collaboration Stage (with ablation support)
    """
    print(f"\n{'=' * 80}")
    print(f"Iteration {current_iteration + 1}")
    print(f"Stage 2: On-demand Agentic Hypergraph Collaboration")
    print(f"Config: {ablation_config['description']}")
    print(f"{'=' * 80}\n")

    # Step 3: Core Block Selection
    if ablation_config['core_block_selection']:
        print(f"[Step 3] Core Block Selection")
        selector = CoreModuleSelector(base_dir=data_dir)
        activated_agents = selector.run(do_initialization=(current_iteration == 0))
        print(f"  ✓ Activated agents: {activated_agents}")
    else:
        print(f"[Step 3] Core Block Selection - SKIPPED (Ablation)")

        activated_agents = [1, 1, 1, 1]
        print(f"  ✓ Default activated agents: {activated_agents}")

    if sum(activated_agents) == 0:
        print("  ✓ Goal achieved! No conflicts remaining")
        return None, True, 0

    # Step 4: Collaboration Decision-Making
    if ablation_config['collaborative_decision']:
        print(f"\n[Step 4] Collaboration Decision-Making")
        decision = CollaborativeDecision(data_dir=data_dir)
        conflict_entities = decision.detect_and_record_conflicts(activated_agents)
        print(f"  ✓ Detected {len(conflict_entities)} conflicts")
    else:
        print(f"\n[Step 4] Collaboration Decision-Making - SKIPPED (Ablation)")

        conflict_entities = set(range(100))  # 示例
        print(f"  ✓ Assuming all entities have conflicts")

    if len(conflict_entities) == 0:
        return None, True, 0

    # Step 5: Agentic Hypergraph Execution
    if ablation_config['agentic_execution']:
        print(f"\n[Step 5] Agentic Hypergraph Execution")
        executor = HypergraphAgentExecution(data_dir=data_dir)
        execution_pairs = executor.execute_conflict_resolution(activated_agents)
        print(f"  ✓ Generated {len(execution_pairs)} alignment pairs")
    else:
        print(f"\n[Step 5] Agentic Hypergraph Execution - SKIPPED (Ablation)")
        execution_pairs = {}
        print(f"  ✓ No execution performed")

    # Step 6: Meta Evaluation
    if ablation_config['meta_evaluation']:
        print(f"\n[Step 6] Meta Evaluation")
        evaluator = MetaEvaluation(data_dir=data_dir)
        updated_train_pairs, should_continue = evaluator.evaluate()

        if should_continue:
            rules = run_full_process_llm2(data_dir)
            print(f"Generated {len(rules)} alignment rules.")
    else:
        print(f"\n[Step 6] Meta Evaluation - SKIPPED (Ablation)")

        should_continue = len(execution_pairs) > 0
        updated_train_pairs = set()

    return execution_pairs, should_continue, len(conflict_entities)

def create_dummy_retrieval_output(data_dir):
    """
    Create placeholder output when Neural Retrieval is disabled
    """
    output_file = os.path.join(data_dir, "message_pool", "retriever_outputs.txt")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    ## Create simple 1:1 mapping as fallback
    ent_ids_1 = os.path.join(data_dir, 'ent_ids_1')
    if os.path.exists(ent_ids_1):
        with open(ent_ids_1, 'r') as f, open(output_file, 'w') as out:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    entity_id = parts[0]
                    # Simple mapping to itself
                    out.write(f"{entity_id}\t{entity_id}\n")
    print(f"  Created dummy retrieval output at {output_file}")

def run_full_process_with_ablation(data_dir, ablation_config):
    """
    Full process (with ablation experiment support)
    """
    start_time = time.time()

    print(f"\n{'#' * 100}")
    print(f"# Running Ablation Experiment: {ablation_config['description']}")
    print(f"{'#' * 100}\n")

    # Stage 1:
    # hypergraph_representation(data_dir, ablation_config)
    print("-" * 80)

    # Stage 2:
    max_iterations = 3
    current_iteration = 0
    should_continue = True

    results = {
        'iterations': 0,
        'total_conflicts': 0,
        'total_alignments': 0
    }

    while should_continue and current_iteration < max_iterations:
        execution_pairs, should_continue, conflict_count = run_collaboration_stage(
            data_dir, ablation_config, current_iteration
        )

        if execution_pairs is None:
            break

        results['iterations'] = current_iteration + 1
        results['total_conflicts'] += conflict_count
        results['total_alignments'] += len(execution_pairs)

        current_iteration += 1

        if current_iteration >= max_iterations:
            print(f"\n达到最大迭代次数 ({max_iterations}),终止迭代。")
            break

    end_time = time.time()
    total_seconds = end_time - start_time


    print(f"\n{'=' * 80}")
    print(f"Ablation Experiment Complete: {ablation_config['description']}")
    print(f"{'=' * 80}")
    print(f"Total Iterations: {results['iterations']}")
    print(f"Total Conflicts Detected: {results['total_conflicts']}")
    print(f"Total Alignments Generated: {results['total_alignments']}")
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)

    print(f"Final Process completed in: {end_time - start_time:.2f} seconds")
    print(f'Time Cost : {hours}hour, {minutes:02d}min, {seconds:02d}sec')
    print(f"Tokens Cost: {tokens_cal.global_tokens}")
    print(f"{'=' * 80}\n")




    return results, total_seconds, tokens_cal.global_tokens

def save_ablation_results(all_results, output_file):
    """
    Save ablation experiment results
    """
    with open(output_file, 'w') as f:
        f.write("Ablation Study Results\n")
        f.write("=" * 100 + "\n\n")

        for config_name, result_data in all_results.items():
            results, time_cost, token_cost = result_data
            f.write(f"Configuration: {config_name}\n")
            f.write(f"Description: {ABLATION_CONFIGS[config_name]['description']}\n")
            f.write(f"  Iterations: {results['iterations']}\n")
            f.write(f"  Total Conflicts: {results['total_conflicts']}\n")
            f.write(f"  Total Alignments: {results['total_alignments']}\n")
            f.write(f"  Time Cost: {time_cost:.2f} seconds\n")
            f.write(f"  Token Cost: {token_cost}\n")
            f.write("-" * 100 + "\n\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="icews_wiki",
                       choices=["icews_wiki", "icews_yago", "fr_en", "dbp_wd_100",
                               "BETA", "MTKGA_W_I", "MTKGA_Y_I", "FB15K_DB15K"],
                       help="Dataset name. Available options: icews_wiki, icews_yago, "
                            "fr_en, dbp_wd_100, BETA, MTKGA_W_I, MTKGA_Y_I, FB15K_DB15K")
    parser.add_argument("--ablation", type=str, default=None,
                       help="Ablation config name (e.g., 'w/o_neural_retrieval'). Use 'all' for all configs.")
    parser.add_argument("--output", type=str, default="ablation_results.txt",
                       help="Output file for ablation results")

    args = parser.parse_args()
    data_dir = os.path.join("/home/dex/Desktop/entity_sy/AdaCoAgent_backup/data", args.data)

    if args.ablation is None or args.ablation == 'full':
        #Run full model
        print("Running full model (no ablation)")
        run_full_process_with_ablation(data_dir, ABLATION_CONFIGS['full'])

    elif args.ablation == 'all':
        # Run all ablation experiments
        print("Running ALL ablation experiments")
        all_results = {}

        for config_name, config in ABLATION_CONFIGS.items():

            tokens_cal.global_tokens = 0

            results, time_cost, token_cost = run_full_process_with_ablation(
                data_dir, config
            )
            all_results[config_name] = (results, time_cost, token_cost)

        output_path = os.path.join(data_dir, args.output)
        save_ablation_results(all_results, output_path)
        print(f"\nAll ablation experiments completed! Results saved to: {output_path}")

    else:
        # Run specified ablation configuration
        if args.ablation not in ABLATION_CONFIGS:
            print(f"Error: Unknown ablation configuration '{args.ablation}'")
            print(f"Available configurations: {list(ABLATION_CONFIGS.keys())}")
            sys.exit(1)

        print(f"Running ablation: {args.ablation}")
        run_full_process_with_ablation(data_dir, ABLATION_CONFIGS[args.ablation])