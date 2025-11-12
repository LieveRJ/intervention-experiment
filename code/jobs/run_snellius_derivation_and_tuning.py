import os
import sys
import json
import math
from datetime import datetime
import subprocess
import argparse
import random
import torch

def _format_activations(dataset):
    """Format the activations for the probe model."""
    activations = dataset['residual_activations']
    model_layers_num = 32
    mlp_dim_num = 4096
    layer_activation_dict={i: torch.zeros(len(activations), mlp_dim_num) for i in range(model_layers_num)}
    for i in range(len(activations)):
        for j in range(model_layers_num):
            layer_activation_dict[j][i] = torch.tensor(activations[i][j])
    return layer_activation_dict

        
def _format_intervention_vectors(layer_diff_means):
    """
    Format the intervention vectors

    Args:
        layer_diff_means: torch.Tensor
    """
    # Prepare the intervention vectors
    intervention_vectors = [layer_diff_means] * 32
    # Replace the first 3 vectors with 0 
    intervention_vectors[:3] = torch.zeros(3, 4096)
    return intervention_vectors


def _calculate_diff_means_directions(layer_activation_dict, first_indices, second_indices):
    """Calculate the diff means direction"""
    # Get the activations for the correct and incorrect predictions
    model_layers_num = 32
    mlp_dim_num = 4096

    candidate_directions = torch.zeros((model_layers_num, mlp_dim_num), dtype=torch.float64, device='cuda')
    
    # calculating candidate reasoning features
    for layer in range(model_layers_num):
        activations = layer_activation_dict[layer]
        #  we store the mean activations in high-precision to avoid numerical issues
        correct_activations = activations[first_indices, :].to(torch.float64)
        incorrect_activations = activations[second_indices, :].to(torch.float64)

        mean_correct_activations = correct_activations.mean(dim=0)
        mean_incorrect_activations = incorrect_activations.mean(dim=0)

        mean_diff = mean_correct_activations - mean_incorrect_activations 
        candidate_directions[layer] = mean_diff

    return candidate_directions


def ensure_poetry_env(project_root: str):
    """Ensure Poetry venv is created and dependencies installed; returns venv site-packages path."""
    print("Setting up Poetry environment")
    cwd = os.getcwd()
    os.chdir(project_root)
    try:
        subprocess.run(['poetry', 'config', 'virtualenvs.in-project', 'true', '--local'], check=True)
        print("Installing dependencies...")
        subprocess.run(['poetry', 'install', '--no-interaction', '--no-root'], check=True)
        # Prefer in-project venv
        venv_path = os.path.join(project_root, '.venv')
        venv_python = os.path.join(venv_path, 'bin', 'python')
        if os.path.isfile(venv_python):
            if os.path.realpath(sys.executables if hasattr(sys, 'executables') else sys.executable) != os.path.realpath(venv_python):
                print(f"Re-executing with venv interpreter: {venv_python}")
                script_path = os.path.abspath(sys.argv[0])
                os.execv(venv_python, [venv_python, script_path] + sys.argv[1:])
        else:
            # Fallback to Poetry-reported path
            result = subprocess.run(['poetry', 'env', 'info', '--path'], capture_output=True, text=True, check=True)
            venv_path = result.stdout.strip()
            venv_python = os.path.join(venv_path, 'bin', 'python')
            if os.path.isfile(venv_python) and os.path.realpath(sys.executable) != os.path.realpath(venv_python):
                print(f"Re-executing with Poetry venv interpreter: {venv_python}")
                script_path = os.path.abspath(sys.argv[0])
                os.execv(venv_python, [venv_python, script_path] + sys.argv[1:])
        # Already running under venv interpreter; add site-packages just in case
        pyver = f'python{sys.version_info.major}.{sys.version_info.minor}'
        for base in ('lib', 'lib64'):
            sp = os.path.join(venv_path, base, pyver, 'site-packages')
            if os.path.isdir(sp) and sp not in sys.path:
                sys.path.insert(0, sp)
        print(f"Using Poetry environment at: {venv_path}")
        return venv_path
    finally:
        os.chdir(cwd)


def compute_diff_means(task: str, mode: str = 'counter_factual'):
    """Compute and store diff-means directions for a task using existing utility."""
    from experiments.chess import ChessExperiment
    from experiments.arithmetic import ArithmeticExperiment
    from experiments.programming import ProgrammingExperiment


    if task == 'chess':
        results_path = f'./outputs/chess/tuning/{mode}/'
        os.makedirs(results_path, exist_ok=True)
        directions_path = os.path.join(results_path, 'diff_means_directions.pth')
        if os.path.exists(directions_path):
            print(f"Found existing diff-means directions: {directions_path}. Skipping recompute.")
            return directions_path
        experiment = ChessExperiment(
            input_path='./inputs/chess/data/',
            output_path=results_path,
            chunk_size=800,
            chunk_id=0,
        )
        dataset = experiment.run_experiment(
            collect_activations=True,
            attach_control_prompts=False,
            save_activations=False,
        )
        results, _, _ = experiment.evaluate_llm_responses(dataset)
        correct_prediction_indices = results[mode]['correct']['yes'] + results[mode]['correct']['no']
        incorrect_prediction_indices = (
            results[mode]['incorrect']['yes'] + results[mode]['incorrect']['no'] + results[mode]['incorrect']['invalid']
        )
    elif task == 'arithmetic-base8':
        base = 8 if mode == 'counter_factual' else 10
        results_path = f'./outputs/arithmetic/tuning/{mode}/'
        os.makedirs(results_path, exist_ok=True)
        directions_path = os.path.join(results_path, 'diff_means_directions.pth')
        if os.path.exists(directions_path):
            print(f"Found existing diff-means directions: {directions_path}. Skipping recompute.")
            return directions_path
        experiment = ArithmeticExperiment(
            input_path=f'./inputs/arithmetic/data/base{base}.txt',
            output_path=results_path,
            chunk_size=1000,
            chunk_id=0,
            base=base,
        )
        dataset = experiment.run_experiment(
            collect_activations=True,
            mode=mode,
            save_activations=False,
        )
        if mode == 'counter_factual':
            _, correct_prediction_indices, real_world_prediction_indices, unparsable_prediction_indices = experiment.evaluate_llm_responses(dataset)
            incorrect_prediction_indices = unparsable_prediction_indices + real_world_prediction_indices
        else:
            correct_prediction_indices = [i for i in range(1000)]
            incorrect_prediction_indices = [i for i in range(1000, 2000)]
    elif task == 'programming':
        results_path = f'./outputs/programming/tuning/{mode}/'
        os.makedirs(results_path, exist_ok=True)
        directions_path = os.path.join(results_path, 'diff_means_directions.pth')
        if os.path.exists(directions_path):
            print(f"Found existing diff-means directions: {directions_path}. Skipping recompute.")
            return directions_path
        experiment = ProgrammingExperiment(
            input_path='./inputs/programming/',
            output_path=results_path,
            chunk_size=800,
            chunk_id=0,
            mode=mode,
        )
        dataset = experiment.run_experiment(
            collect_activations=True,
        )
        _, correct_prediction_indices, incorrect_prediction_indices = experiment.evaluate_llm_responses(dataset)
    else:
        raise ValueError(f"Unknown task: {task}")

    layer_activation_dict = _format_activations(dataset)
    print('Calculating diff-means directions...')
    diff_means = _calculate_diff_means_directions(layer_activation_dict, correct_prediction_indices, incorrect_prediction_indices)
    diff_means_cpu = diff_means.cpu().float()
    torch_path = os.path.join(results_path, 'diff_means_directions.pth')
    print(f'Saving diff-means to {torch_path}')
    torch.save(diff_means_cpu, torch_path)
    return torch_path


def tune_on_gsm(task: str, directions_path: str, sample_size: int = 100, mode: str = 'counter_factual'):
    """Tune layer and alpha on GSM-symbolic; return best config and save optimal direction tensor."""
    from experiments.gsm_symbolic import GSMSymbolicExperiment

    out_root = f'./outputs/{task}/tuning/{mode}/'
    os.makedirs(out_root, exist_ok=True)
    alphas = [0.10] # Only one alpha value (Grid search takes too long)

    print(f'Loading directions from {directions_path}')
    layer_wise = torch.load(directions_path)

    experiment = GSMSymbolicExperiment(
        input_path='./inputs/liref/gsm-symbolic_data/',
        output_path=f'./outputs/gsm-symbolic/intervention/tuning_temp/',
        chunk_size=sample_size,
        chunk_id=0,
        model_name='meta-llama/Llama-3.1-8B',
        sample_size=sample_size,
    )

    best = { 'metric': -math.inf, 'alpha': None, 'layer': None }
    metrics = {}

    for layer in range(layer_wise.shape[0]):
        direction = layer_wise[layer]
        intervention_vectors = _format_intervention_vectors(direction)
        for alpha in alphas:
            dataset = experiment.run_experiment(
                intervention_vectors=intervention_vectors,
                alpha=alpha,
                collect_activations=False,
                test_set=False,
            )
            reason_acc, _, _ = experiment.evaluate_llm_responses(dataset, test_set=False)
            metrics.setdefault(str(layer), {})[str(alpha)] = float(reason_acc)
            if reason_acc > best['metric']:
                best.update({'metric': float(reason_acc), 'alpha': float(alpha), 'layer': int(layer)})

    # Save metrics and best
    metrics_path = os.path.join(out_root, 'gsm_tuning_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({'task': task, 'mode': mode, 'metrics': metrics, 'best': best, 'updated_at': datetime.now().isoformat()}, f, indent=2)

    # Save optimal direction
    optimal_dir = layer_wise[best['layer']].cpu().float()
    optimal_out_dir = f'./outputs/optimal_directions/{task}/'
    os.makedirs(optimal_out_dir, exist_ok=True)
    optimal_path = os.path.join(optimal_out_dir, 'optimal_direction.pth')
    torch.save(optimal_dir, optimal_path)

    # Also store a small manifest
    manifest = { 'task': task, 'directions_source': directions_path, 'best': best, 'saved_at': datetime.now().isoformat(), 'path': optimal_path }
    with open(os.path.join(optimal_out_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"Best for {task}: layer={best['layer']}, alpha={best['alpha']}, acc={best['metric']:.4f}")
    return best, optimal_path


def compute_combined_diff_means(tasks=None, mode: str = 'counter_factual'):
    """
    Create a combined layer-wise diff-means tensor by averaging across tasks.
    Returns the saved path to the combined diff_means_directions tensor.
    """
    if tasks is None:
        tasks = ['chess', 'arithmetic-base8', 'programming']
    paths = [compute_diff_means(task, mode=mode) for task in tasks]
    # Force-load to CPU to avoid mixed device tensors when stacking
    tensors = [torch.load(p, map_location='cpu').cpu() for p in paths]
    stacked = torch.stack(tensors, dim=0)
    combined = stacked.mean(dim=0)
    out_root = f'./outputs/combined/tuning/{mode}/'
    os.makedirs(out_root, exist_ok=True)
    combined_path = os.path.join(out_root, 'diff_means_directions.pth')
    torch.save(combined.cpu().float(), combined_path)
    return combined_path


def compute_combined_diff_means_balanced(tasks=None, mode: str = 'counter_factual'):
    """
    Run each task to collect activations, then for each task:
    Finally, compute the weighted mean across tasks for correct and incorrect, where weights are
    proportional to the task dataset sizes (before balancing), and return the saved path.
    """
    from experiments.chess import ChessExperiment
    from experiments.arithmetic import ArithmeticExperiment
    from experiments.programming import ProgrammingExperiment

    # Check if the combined_balanced_diff_means_directions.pth exists
    combined_balanced_path = f'./outputs/combined_balanced/tuning/{mode}/diff_means_directions.pth'
    if os.path.exists(combined_balanced_path):
        print(f"Found existing combined balanced diff-means directions: {combined_balanced_path}. Skipping recompute.")
        return combined_balanced_path

    if tasks is None:
        tasks = ['chess', 'arithmetic-base8', 'programming']

    # Collect per-task activations and indices
    task_entries = []
    for task in tasks:
        if task == 'chess':
            results_path = f'./outputs/chess/tuning/{mode}/'
            os.makedirs(results_path, exist_ok=True)
            experiment = ChessExperiment(
                input_path='./inputs/chess/data/',
                output_path=results_path,
                chunk_size=800,
                chunk_id=0,
            )
            dataset = experiment.run_experiment(
                collect_activations=True,
                attach_control_prompts=False,
                save_activations=False,
            )
            results, _, _ = experiment.evaluate_llm_responses(dataset)
            correct_indices = results[mode]['correct']['yes'] + results[mode]['correct']['no']
            incorrect_indices = (
                results[mode]['incorrect']['yes'] + results[mode]['incorrect']['no'] + results[mode]['incorrect']['invalid']
            )
        elif task == 'arithmetic-base8':
            base = 8 if mode == 'counter_factual' else 10
            results_path = f'./outputs/arithmetic/tuning/{mode}/'
            os.makedirs(results_path, exist_ok=True)
            experiment = ArithmeticExperiment(
                input_path=f'./inputs/arithmetic/data/base{base}.txt',
                output_path=results_path,
                chunk_size=1000,
                chunk_id=0,
                base=base,
            )
            dataset = experiment.run_experiment(
                collect_activations=True,
                mode=mode,
                save_activations=False,
            )
            if mode == 'counter_factual':
                _, correct_indices, real_world_indices, unparsable_indices = experiment.evaluate_llm_responses(dataset)
                incorrect_indices = unparsable_indices + real_world_indices
            else:
                correct_indices = [i for i in range(1000)]
                incorrect_indices = [i for i in range(1000, 2000)]
        elif task == 'programming':
            results_path = f'./outputs/programming/tuning/{mode}/'
            os.makedirs(results_path, exist_ok=True)
            experiment = ProgrammingExperiment(
                input_path='./inputs/programming/',
                output_path=results_path,
                chunk_size=800,
                chunk_id=0,
                mode=mode,
            )
            dataset = experiment.run_experiment(
                collect_activations=True,
            )
            _, correct_indices, incorrect_indices = experiment.evaluate_llm_responses(dataset)
        else:
            raise ValueError(f"Unknown task: {task}")
        
        activations = _format_activations(dataset)
        task_size = len(dataset)
        task_entries.append({
            'task': task,
            'activations': activations,
            'correct': list(correct_indices),
            'incorrect': list(incorrect_indices),
            'size': task_size,
        })

    model_layers_num = 32
    mlp_dim_num = 4096

    # Helper: unit-normalize rows
    def _unit_normalize_rows(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        norms = torch.linalg.norm(x, dim=1, keepdim=True)
        norms = torch.clamp(norms, min=eps)
        return x / norms

    # Task-balanced, class-balanced weighting using all data (no subsampling)
    mu_correct = torch.zeros((model_layers_num, mlp_dim_num), dtype=torch.float32)
    mu_incorrect = torch.zeros((model_layers_num, mlp_dim_num), dtype=torch.float32)
    T = len(task_entries)
    w_task = 1.0 / float(T) if T > 0 else 1.0
    w_class = 0.5

    for t_idx, entry in enumerate(task_entries):
        corr = list(entry['correct'])
        incorr = list(entry['incorrect'])
      
        for layer in range(model_layers_num):
            raw_layer = entry['activations'][layer].to(torch.float32)
            norm_layer = _unit_normalize_rows(raw_layer)
            if len(corr) > 0:
                mu_correct[layer] += w_task*w_class * norm_layer[corr, :].mean(dim=0)
            if len(incorr) > 0:
                mu_incorrect[layer] += w_task*w_class * norm_layer[incorr, :].mean(dim=0)

    combined_direction = (mu_correct - mu_incorrect).to(torch.float32)

    out_root = f'./outputs/combined_balanced_weighted/tuning/{mode}/'
    os.makedirs(out_root, exist_ok=True)
    out_path = os.path.join(out_root, 'diff_means_directions.pth')
    torch.save(combined_direction.cpu().float(), out_path)
    return out_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_root', type=str, default='/gpfs/home5/jholshuijsen/reasoning-reciting-probing')
    parser.add_argument('--tasks', type=str, nargs='+', default=['chess', 'arithmetic-base8', 'programming'])
    parser.add_argument('--gsm_samples', type=int, default=100)
    parser.add_argument('--tune_target', type=str, choices=['tasks', 'combined', 'both', 'combined_balanced'], default='both',
                        help='Tune per task (tasks), the averaged combined directions (combined), or both')
    args = parser.parse_args()

    ensure_poetry_env(args.project_root)
    # Ensure we can import modules from the repository's code directory
    sys.path.insert(0, os.path.join(args.project_root, 'code'))

    torch.manual_seed(8888)
    random.seed(8888)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(8888)
        torch.cuda.manual_seed_all(8888)

    results = {}
    if args.tune_target in ('tasks', 'both'):
        for task in args.tasks:
            print(f'=== Processing task: {task} ===')
            dir_path = compute_diff_means(task, mode='counter_factual')
            best, optimal_path = tune_on_gsm(task, dir_path, sample_size=args.gsm_samples, mode='counter_factual')
            results[task] = {'best': best, 'optimal_path': optimal_path}

    if args.tune_target in ('combined', 'both'):
        # Also tune the combined average of the three direction sets
        print('=== Processing task: combined ===')
        combined_dir_path = compute_combined_diff_means(mode='counter_factual')
        best, optimal_path = tune_on_gsm('combined', combined_dir_path, sample_size=args.gsm_samples, mode='counter_factual')
        results['combined'] = {'best': best, 'optimal_path': optimal_path}
    if args.tune_target in ('combined_balanced', 'both'):
        print('=== Processing task: combined_balanced ===')
        combined_dir_path = compute_combined_diff_means_balanced(mode='counter_factual')
        best, optimal_path = tune_on_gsm('combined_balanced', combined_dir_path, sample_size=args.gsm_samples, mode='counter_factual')
        results['combined_balanced'] = {'best': best, 'optimal_path': optimal_path}

    summary_path = './outputs/optimal_directions/summary.json'
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump({'results': results, 'created_at': datetime.now().isoformat()}, f, indent=2)
    print(f'Summary saved to {summary_path}')


if __name__ == '__main__':
    # sys.path.insert(0, '/gpfs/home5/jholshuijsen/reasoning-reciting-probing/code')
    main()
    # ensure_poetry_env('/gpfs/home5/jholshuijsen/reasoning-reciting-probing')
    # compute_combined_diff_means_balanced(mode='counter_factual')


