import os
import sys
import json
import argparse
import subprocess
import torch
from datetime import datetime


# Hardcoded alpha sweep: test from 0.25 down to -0.25 in steps of 0.05 (including 0.0)
ALPHAS = [0.25, 0.20, 0.15, 0.10, 0.05, 0.00, -0.05, -0.10, -0.15, -0.20, -0.25]


def ensure_poetry_env(project_root: str):
    """Ensure Poetry venv is created and dependencies installed; re-exec under venv."""
    print("Setting up Poetry environment")
    cwd = os.getcwd()
    os.chdir(project_root)
    try:
        subprocess.run(['poetry', 'config', 'virtualenvs.in-project', 'true', '--local'], check=True)
        print("Installing dependencies...")
        subprocess.run(['poetry', 'install', '--no-interaction', '--no-root'], check=True)
        venv_path = os.path.join(project_root, '.venv')
        venv_python = os.path.join(venv_path, 'bin', 'python')
        if os.path.isfile(venv_python):
            if os.path.realpath(sys.executable) != os.path.realpath(venv_python):
                print(f"Re-executing with venv interpreter: {venv_python}")
                script_path = os.path.abspath(sys.argv[0])
                os.execv(venv_python, [venv_python, script_path] + sys.argv[1:])
        else:
            result = subprocess.run(['poetry', 'env', 'info', '--path'], capture_output=True, text=True, check=True)
            venv_path = result.stdout.strip()
            venv_python = os.path.join(venv_path, 'bin', 'python')
            if os.path.isfile(venv_python) and os.path.realpath(sys.executable) != os.path.realpath(venv_python):
                print(f"Re-executing with Poetry venv interpreter: {venv_python}")
                script_path = os.path.abspath(sys.argv[0])
                os.execv(venv_python, [venv_python, script_path] + sys.argv[1:])
        print(f"Using Poetry environment at: {venv_path}")
        return venv_path
    finally:
        os.chdir(cwd)


def _format_intervention_vectors(layer_diff_means: torch.Tensor):
    intervention_vectors = [layer_diff_means] * 32
    intervention_vectors[:3] = torch.zeros(3, 4096)
    return intervention_vectors


def eval_on_gsm(direction: torch.Tensor, alpha: float, sample_size: int = 200, test_set: bool = True):
    from experiments.gsm_symbolic import GSMSymbolicExperiment

    intervention_vectors = _format_intervention_vectors(direction)
    experiment = GSMSymbolicExperiment(
        input_path='./inputs/liref/gsm-symbolic_data/',
        output_path=f'./outputs/gsm-symbolic/intervention/eval_temp/',
        chunk_size=sample_size,
        chunk_id=0,
        model_name='meta-llama/Llama-3.1-8B',
        sample_size=sample_size,
    )
    dataset = experiment.run_experiment(
        intervention_vectors=intervention_vectors,
        alpha=alpha,
        collect_activations=False,
        test_set=test_set,
    )
    acc, _, _ = experiment.evaluate_llm_responses(dataset, test_set=test_set)
    return float(acc)


def eval_on_chess(direction: torch.Tensor, alpha: float):
    from experiments.chess import ChessExperiment
    intervention_vectors = _format_intervention_vectors(direction)
    experiment = ChessExperiment(
        input_path='./inputs/chess/data/',
        output_path='./outputs/chess/intervention/eval_temp/',
        chunk_size=800,
        chunk_id=0,
    )
    dataset = experiment.run_experiment(
        intervention_vectors=intervention_vectors,
        alpha=alpha,
        attach_control_prompts=False,
        collect_activations=False,
    )
    results, _, _ = experiment.evaluate_llm_responses(dataset)
    rw_acc, cf_acc, _, _ = experiment.get_chess_accuracies(results)
    return {'real_world': float(rw_acc), 'counter_factual': float(cf_acc)}


def eval_on_arithmetic(direction: torch.Tensor, alpha: float, mode: str = 'counter_factual'):
    from experiments.arithmetic import ArithmeticExperiment
    base = 8 if mode == 'counter_factual' else 10
    intervention_vectors = _format_intervention_vectors(direction)
    experiment = ArithmeticExperiment(
        input_path=f'./inputs/arithmetic/data/base{base}.txt',
        output_path=f'./outputs/arithmetic/intervention/eval_temp/',
        chunk_size=1000,
        chunk_id=0,
        base=base,
    )
    dataset = experiment.run_experiment(
        intervention_vectors=intervention_vectors,
        alpha=alpha,
        collect_activations=False,
        mode=mode,
    )
    acc, reasoning_acc, memorization_acc, unparsable_acc = experiment.evaluate_llm_responses(dataset)
    return float(acc)


def eval_on_programming(direction: torch.Tensor, alpha: float, mode: str = 'counter_factual'):
    from experiments.programming import ProgrammingExperiment
    intervention_vectors = _format_intervention_vectors(direction)
    experiment = ProgrammingExperiment(
        input_path='./inputs/programming/',
        output_path='./outputs/programming/intervention/eval_temp/',
        chunk_size=800,
        chunk_id=0,
        mode=mode,
    )
    dataset = experiment.run_experiment(
        intervention_vectors=intervention_vectors,
        alpha=alpha,
        collect_activations=False,
    )
    pass_k, _, _ = experiment.evaluate_llm_responses(dataset)
    return {k: float(v) for k, v in pass_k.items()}


# Optional hardcoded layer overrides per task; set to an integer to override, or None to use tuned best
LAYER_OVERRIDES = {
    'chess': 25,               # e.g., 20
    'arithmetic-base8': 17,    # e.g., 9
    'programming': 16,         # e.g., 15
    'combined': None,          # use optimal if available; otherwise layerwise fallback
}

HARD_CODED_INPUTS = {
    'chess': {
        'optimal_path': './outputs/optimal_directions/chess/optimal_direction.pth',
        'layerwise_path': './outputs/chess/tuning/counter_factual/diff_means_directions.pth',
    },
    'arithmetic-base8': {
        'optimal_path': './outputs/optimal_directions/arithmetic-base8/optimal_direction.pth',
        'layerwise_path': './outputs/arithmetic/tuning/counter_factual/diff_means_directions.pth',
    },
    'programming': {
        'optimal_path': './outputs/optimal_directions/programming/optimal_direction.pth',
        'layerwise_path': './outputs/programming/tuning/counter_factual/diff_means_directions.pth',
    },
    'combined': {
        'optimal_path': './outputs/optimal_directions/combined/optimal_direction.pth',
        'layerwise_path': './outputs/combined/tuning/counter_factual/diff_means_directions.pth',
    },
    'combined_balanced': {
        'optimal_path': './outputs/optimal_directions/combined_balanced/optimal_direction.pth',
        'layerwise_path': './outputs/combined_balanced/tuning/counter_factual/diff_means_directions.pth',
    },
}


def load_direction_for_task(task: str) -> tuple[torch.Tensor, int, str]:
    """Return (direction_vector, layer_used, source_kind) using hardcoded paths and overrides."""
    cfg = HARD_CODED_INPUTS.get(task)
    if cfg is None:
        raise KeyError(f"No HARD_CODED_INPUTS entry for task '{task}'.")

    override = LAYER_OVERRIDES.get(task)
    optimal_path = cfg.get('optimal_path')
    layerwise_path = cfg.get('layerwise_path')

    if override is None:
        if optimal_path and os.path.exists(optimal_path):
            vec = torch.load(optimal_path)
            return vec, -1, 'optimal'
        elif layerwise_path and os.path.exists(layerwise_path):
            layer_wise = torch.load(layerwise_path)
            norms = torch.linalg.norm(layer_wise, dim=1)
            best_layer = int(torch.argmax(norms).item())
            vec = layer_wise[best_layer].cpu().float()
            return vec, best_layer, 'layerwise_max_norm_fallback'
        else:
            raise FileNotFoundError(f"Neither optimal_path nor layerwise_path available for task '{task}'.")
    else:
        if not (layerwise_path and os.path.exists(layerwise_path)):
            raise FileNotFoundError(f"Layer override requires layerwise_path for task '{task}'. Missing: {layerwise_path}")
        layer_wise = torch.load(layerwise_path)
        if override < 0 or override >= layer_wise.shape[0]:
            raise ValueError(f"Override layer {override} out of range for {task}; tensor has {layer_wise.shape[0]} layers.")
        vec = layer_wise[override].cpu().float()
        return vec, int(override), 'layer_override'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_root', type=str, default='/gpfs/home5/jholshuijsen/reasoning-reciting-probing')
    parser.add_argument('--tasks', type=str, nargs='+', default=['chess', 'arithmetic-base8', 'programming', 'combined', 'combined_balanced'])
    parser.add_argument('--gsm_samples', type=int, default=200)
    parser.add_argument('--gsm_test_set', action='store_true', default=True)
    # alpha_sweep arg is no longer used; ALPHAS is hardcoded above
    args = parser.parse_args()

    ensure_poetry_env(args.project_root)
    sys.path.insert(0, os.path.join(args.project_root, 'code'))

    results = {}
    # Evaluate each task's direction on all tasks (GSM, chess, arithmetic, programming)
    for source_task in args.tasks:
        direction, used_layer, src_kind = load_direction_for_task(source_task)

        # Store as: task_direction -> tasks -> alphas
        src_res = { 'tasks': { 'gsm_symbolic': {}, 'chess': {}, 'arithmetic': {}, 'programming': {} } }
        for alpha in ALPHAS:
            alpha_key = f"{alpha:.2f}"
            # Always evaluate on GSM-Symbolic
            print(f"Evaluating on GSM-Symbolic for {source_task} with alpha {alpha}")
            gsm_acc = eval_on_gsm(direction, alpha, sample_size=args.gsm_samples, test_set=args.gsm_test_set)
            # Evaluate on all tasks
            print(f"Evaluating on Chess for {source_task} with alpha {alpha}")
            chess_res = eval_on_chess(direction, alpha)
            print(f"Evaluating on Arithmetic for {source_task} with alpha {alpha}")
            arith_acc = eval_on_arithmetic(direction, alpha, mode='counter_factual')
            print(f"Evaluating on Programming for {source_task} with alpha {alpha}")
            prog_res = eval_on_programming(direction, alpha, mode='counter_factual')
            src_res['tasks']['gsm_symbolic'][alpha_key] = gsm_acc
            src_res['tasks']['chess'][alpha_key] = chess_res
            src_res['tasks']['arithmetic'][alpha_key] = arith_acc
            src_res['tasks']['programming'][alpha_key] = prog_res

        results[source_task] = src_res

    out_manifest = {
        'created_at': datetime.now().isoformat(),
        'gsm_samples': args.gsm_samples,
        'results': results,
    }
    optimal_dir_filename = '_'.join(args.tasks)
    out_path = f'./outputs/optimal_directions/cross_task_eval_{optimal_dir_filename}.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out_manifest, f, indent=2)
    print(f"Saved cross-task evaluation to {out_path}")


if __name__ == '__main__':
    main()
