import logging
import os
import sys
from types import SimpleNamespace # To simulate the args object
from typing import Optional, List, Tuple, Dict, Any

# --- Project Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# --- Imports from Project Modules ---
from RLSortingNetworks.sorting_network_rl.utils.config_loader import load_config
from RLSortingNetworks.sorting_network_rl.utils.evaluation import (
    prune_redundant_comparators,
    format_network_visualization,
    is_sorting_network
)
from RLSortingNetworks.sorting_network_rl.core.evaluator import Evaluator

# --- Constants ---
CONFIG_FILENAME = "config.yaml"
MODEL_FILENAME = "model.pt"
BEST_NETWORK_FILENAME = "best_network.csv"

# --- Helper Functions ---

def _get_hardcoded_arguments() -> SimpleNamespace:
    """
    Defines the hardcoded arguments for IDE execution.

    Returns:
        SimpleNamespace: An object containing evaluation arguments (run_dir, prune, eval_best).
    """
    # Specify the path relative to the project root for the run directory
    # Example: If the directory is checkpoints/4w_10s/
    HARDCODED_RUN_DIR = "checkpoints/4w_10s"

    HARDCODED_PRUNE = True  # Set to True to attempt pruning, False otherwise
    HARDCODED_EVAL_BEST = True # Set to True to evaluate best_network.csv, False to evaluate model.pt

    args = SimpleNamespace(
        run_dir=HARDCODED_RUN_DIR,
        prune=HARDCODED_PRUNE,
        eval_best=HARDCODED_EVAL_BEST
    )
    logging.info("--- Running in hardcoded mode ---")
    logging.info(f"Run directory: {args.run_dir}")
    logging.info(f"Prune network: {args.prune}")
    logging.info(f"Evaluate best network (CSV): {args.eval_best}")
    logging.info("---------------------------------")
    return args

def _validate_paths(run_dir_arg: str, project_root: str) -> Dict[str, str]:
    """
    Validates the run directory path and constructs absolute paths for required files.

    Args:
        run_dir_arg (str): The run directory path (can be relative or absolute).
        project_root (str): The absolute path to the project root.

    Returns:
        Dict[str, str]: A dictionary containing absolute paths for 'run_dir',
                        'config_path', 'model_path', 'best_network_path'.

    Raises:
        SystemExit: If the run directory or config file is not found.
    """
    # Construct the absolute path for the run directory
    if os.path.isabs(run_dir_arg):
        effective_run_dir = run_dir_arg
    else:
        # Ensure the path is interpreted relative to the project root
        effective_run_dir = os.path.abspath(os.path.join(project_root, run_dir_arg))

    logging.info(f"Using effective run directory: {effective_run_dir}")

    if not os.path.isdir(effective_run_dir):
        logging.error(f"Run directory not found: {effective_run_dir}")
        sys.exit(1)

    paths = {
        "run_dir": effective_run_dir,
        "config_path": os.path.join(effective_run_dir, CONFIG_FILENAME),
        "model_path": os.path.join(effective_run_dir, MODEL_FILENAME),
        "best_network_path": os.path.join(effective_run_dir, BEST_NETWORK_FILENAME),
    }

    # Essential check: configuration file must exist
    if not os.path.exists(paths["config_path"]):
        logging.error(f"Configuration file ({CONFIG_FILENAME}) not found in: {effective_run_dir}")
        sys.exit(1)

    # Existence checks for model and best_network are handled later based on evaluation mode

    return paths

def _load_or_generate_network(evaluator: Evaluator, args: SimpleNamespace, paths: Dict[str, str]) -> Tuple[Optional[List[Tuple[int, int]]], str]:
    """
    Loads the best network from CSV or generates one using the agent's policy based on args.

    Args:
        evaluator (Evaluator): The initialized Evaluator instance.
        args (SimpleNamespace): Hardcoded arguments containing 'eval_best'.
        paths (Dict[str, str]): Dictionary containing necessary file paths.

    Returns:
        Tuple[Optional[List[Tuple[int, int]]], str]:
            - The list of comparators, or None if loading/generation fails.
            - A string describing the source of the network.
    """
    if args.eval_best:
        logging.info(f"Attempting to load best network from: {paths['best_network_path']}")
        comparators = evaluator.load_network_from_csv(paths['best_network_path'])
        source_description = "Best Network (from CSV)"
        if comparators is None:
            logging.error("Failed to load best network.")
            # Let the main function handle exit if necessary
    else:
        # Explicitly check if the model file exists before evaluating policy
        if not os.path.exists(paths['model_path']):
             logging.error(f"Model file ({MODEL_FILENAME}) not found at {paths['model_path']}. Cannot evaluate agent policy.")
             return None, "Agent Policy Output (Error: Model Missing)" # Return None if model is missing

        logging.info("Evaluating agent's learned policy...")
        try:
            # The Evaluator's __init__ already attempted to load the model
            comparators, _ = evaluator.evaluate_policy() # Ignore is_valid returned here
            source_description = "Agent Policy Output"
            if comparators is None: # Check if policy evaluation itself failed
                logging.error("Failed to generate network using agent policy (policy evaluation returned None).")
        except Exception as e:
             # Catch unexpected errors during policy evaluation run
             logging.error(f"Error during policy evaluation run: {e}", exc_info=True)
             comparators = None
             source_description = "Agent Policy Output (Error: Exception during eval)"

    return comparators, source_description


def _analyze_and_visualize(comparators: List[Tuple[int, int]], n_wires: int, source_desc: str, prune_flag: bool) -> None:
    """
    Performs analysis, visualization, and optional pruning of the network.

    Args:
        comparators (List[Tuple[int, int]]): The network comparator list.
        n_wires (int): Number of wires for the network.
        source_desc (str): Description of the network's origin.
        prune_flag (bool): Whether to attempt pruning.
    """
    logging.info(f"\n--- Network Analysis ({source_desc}) ---")
    logging.info(f"Original Length: {len(comparators)}")
    logging.info("Network Steps:")
    # Use print for cleaner list output without logger formatting
    for i, comp in enumerate(comparators):
        print(f"  Step {i+1}: {comp}")

    # Check validity after printing steps
    is_valid = is_sorting_network(n_wires, comparators)
    valid_str = "VALID" if is_valid else "INVALID"
    logging.info(f"Network Status: {valid_str}")

    logging.info("\nVisualization (Original):")
    # Use print for multi-line visualization output
    print(format_network_visualization(comparators, n_wires))

    if prune_flag:
        if not is_valid:
            logging.warning("Skipping pruning because the network is not valid.")
        else:
            logging.info("\nAttempting to prune network...")
            pruned_comparators = prune_redundant_comparators(n_wires, comparators)
            if len(pruned_comparators) < len(comparators):
                logging.info(f"Pruning successful! Reduced length to: {len(pruned_comparators)}")
                logging.info("\nVisualization (Pruned):")
                print(format_network_visualization(pruned_comparators, n_wires))
            else:
                logging.info("Pruning did not find any redundant comparators.")


def main():
    args = _get_hardcoded_arguments()

    # 2. Set up basic console logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    try:
        # 3. Validate paths and get absolute paths
        paths = _validate_paths(args.run_dir, project_root)

        # 4. Load the run-specific configuration
        config = load_config(paths["config_path"])
        n_wires = config.get('environment', {}).get('n_wires', None)
        if n_wires is None:
            raise ValueError("n_wires not found in environment configuration.")

        # 5. Initialize the Evaluator
        evaluator = Evaluator(config, paths["model_path"])

        # 6. Load or generate the network to evaluate
        comparators_to_evaluate, source_description = _load_or_generate_network(evaluator, args, paths)

        # 7. Analyze and visualize the network (if obtained successfully)
        if comparators_to_evaluate is not None:
            _analyze_and_visualize(comparators_to_evaluate, n_wires, source_description, args.prune)
        else:
            logging.error("Evaluation cannot proceed as no network comparators were loaded or generated.")
            sys.exit(1) # Exit explicitly if network couldn't be obtained

    except FileNotFoundError as e:
        # Catch errors specifically related to missing files (like model.pt if policy eval was attempted)
        logging.error(f"File not found during setup or evaluation: {e}", exc_info=False)
        sys.exit(1)
    except ValueError as e:
        # Catch errors related to configuration or invalid values
        logging.error(f"Configuration or value error: {e}", exc_info=False)
        sys.exit(1)
    except IOError as e:
        # Catch potential IO errors during file operations
        logging.error(f"IO error during evaluation: {e}", exc_info=False)
        sys.exit(1)
    except Exception as e:
        # Catch any other unexpected errors
        logging.error(f"An unexpected error occurred during evaluation: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()