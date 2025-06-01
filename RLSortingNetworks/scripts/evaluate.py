import argparse
import logging
import os
import sys
from types import SimpleNamespace
from typing import Optional, List, Tuple, Dict, Any

# --- Project Setup ---
# Determine the directory containing this script and the project root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
# Add the project root to the Python path to allow imports like 'from sorting_network_rl...'
sys.path.insert(0, project_root)

# --- Imports from Project Modules ---
# Ensure package name ('sorting_network_rl') matches your directory structure if renamed
from RLSortingNetworks.sorting_network_rl.utils.config_loader import load_config
from RLSortingNetworks.sorting_network_rl.utils.evaluation import (
    prune_redundant_comparators,
    format_network_visualization,
    is_sorting_network,
    calculate_network_depth,
    calculate_network_depth_by_levels,
)
from RLSortingNetworks.sorting_network_rl.core.evaluator import Evaluator

# --- Constants ---
CONFIG_FILENAME = "config.yaml"
MODEL_FILENAME = "model.pt"
BEST_NETWORK_FILENAME = "best_network.csv"
DEFAULT_CONFIGS_DIR = os.path.join(project_root, "configs")
DEFAULT_CHECKPOINTS_DIR_NAME = "checkpoints"

# --- Helper Functions ---
# get_config_path_for_n, get_run_dir_path_for_n, get_n_from_user
# _validate_paths_and_load_config remain the same as the previous version.
# ... (Include the implementations of these functions here) ...

def get_config_path_for_n(n_value: int, configs_dir: str) -> str:
    """Constructs the expected path for the config file based on n."""
    config_filename = f"config_n{n_value}.yaml"
    return os.path.join(configs_dir, config_filename)

def get_run_dir_path_for_n(n_value: int, max_steps: int, agent_suffix:str, checkpoints_base_dir: str) -> str:
    """Constructs the expected path for the run directory based on n and max_steps."""
    run_id = f"{n_value}w_{max_steps}s{agent_suffix}"
    return os.path.join(checkpoints_base_dir, run_id)

def get_n_from_user(min_n: int = 1, max_n: int = 17) -> int:
    """Prompts the user to enter the number of wires (n) interactively."""
    while True:
        try:
            n_input = input(f"Enter the number of wires (n) between {min_n} and {max_n}: ")
            n_value = int(n_input)
            if min_n <= n_value <= max_n:
                return n_value
            else:
                print(f"Error: Please enter a number between {min_n} and {max_n}.")
        except ValueError:
            print("Error: Invalid input. Please enter an integer.")
        except EOFError:
             print("\nInput cancelled by user.")
             sys.exit(0)

def get_agent_type_from_user(default_agent: str = "double_dqn") -> str:
    """Prompts the user to select the agent type if not specified."""
    while True:
        try:
            prompt_message = (
                f"Select agent type to evaluate ('double' or 'classic') "
                f"[Press Enter for default: {default_agent.replace('_dqn','')}]: "
            )
            agent_input = input(prompt_message).strip().lower()
            if not agent_input: # User pressed Enter, use default
                return default_agent
            if agent_input == "double":
                return "double_dqn"
            if agent_input == "classic":
                return "classic_dqn"
            else:
                print("Error: Invalid input. Please enter 'double' or 'classic'.")
        except EOFError:
            print("\nInput cancelled by user.")
            sys.exit(0)

def _validate_paths_and_load_config(n_to_run: int, agent_type_str: str, configs_dir: str, project_root: str) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Validates paths, loads config, resolves checkpoint dir."""
    # (Implementation remains the same as before)
    # 1. Get Config Path
    config_path = get_config_path_for_n(n_to_run, configs_dir)
    logging.info(f"Attempting to load configuration from: {config_path}")

    # 2. Load Config
    try:
        config = load_config(config_path)
        env_config = config.get('environment', {})
        loaded_n = env_config.get('n_wires')
        max_steps = env_config.get('max_steps')

        if loaded_n is None or max_steps is None:
            raise ValueError("Config file must contain 'environment.n_wires' and 'environment.max_steps'.")
        if loaded_n != n_to_run:
            logging.warning(f"Mismatch: Requested n={n_to_run} but config file specifies n_wires={loaded_n}. Using n={loaded_n}.")
            n_to_run = loaded_n

        logging.info(f"Successfully loaded configuration for n={n_to_run}, max_steps={max_steps}.")
    except FileNotFoundError:
        logging.error(f"Config file not found: {config_path}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading/validating config {config_path}: {e}")
        sys.exit(1)

    # 3. Resolve Checkpoint Base Directory
    try:
        exp_config = config.get('experiment', {})
        base_dir_relative = exp_config.get('base_dir', DEFAULT_CHECKPOINTS_DIR_NAME)
        if os.path.isabs(base_dir_relative):
            absolute_base_dir = base_dir_relative
        else:
            absolute_base_dir = os.path.abspath(os.path.join(project_root, base_dir_relative))
        config['experiment']['base_dir'] = absolute_base_dir
        logging.info(f"Using absolute checkpoint base directory: {absolute_base_dir}")
    except Exception as e:
         logging.error(f"Error processing checkpoint base directory: {e}")
         sys.exit(1)

    # 4. Determine Run Directory Path
    agent_suffix = "_classic" if agent_type_str.lower() == "classic_dqn" else ""
    run_dir = get_run_dir_path_for_n(n_to_run, max_steps, agent_suffix,absolute_base_dir)
    logging.info(f"Expecting run artifacts in: {run_dir}")

    # 5. Construct full paths for artifacts
    paths = {
        "run_dir": run_dir,
        "config_path": config_path,
        "model_path": os.path.join(run_dir, MODEL_FILENAME), # Re-included model path
        "best_network_path": os.path.join(run_dir, BEST_NETWORK_FILENAME),
    }

    # Check if run directory exists
    if not os.path.isdir(paths["run_dir"]):
        logging.warning(f"Run directory not found: {paths['run_dir']}. Evaluation might fail if files are missing.")

    return paths, config


# Function with fallback logic reintroduced
def _get_network_to_evaluate(evaluator: Evaluator, paths: Dict[str, str]) -> Tuple[Optional[List[Tuple[int, int]]], str]:
    """
    Attempts to load the best network from CSV first. If unavailable,
    falls back to generating a network using the agent's policy from model.pt.
    """
    comparators = None
    source_description = "Unknown"

    # --- Attempt 1: Load Best Network from CSV ---
    logging.info(f"Attempting to load best network from: {paths['best_network_path']}")
    if os.path.exists(paths['best_network_path']):
        comparators = evaluator.load_network_from_csv(paths['best_network_path'])
        if comparators:
            source_description = "Best Network (from CSV)"
            logging.info("Successfully loaded best network from CSV.")
        else:
            logging.warning(f"Found {BEST_NETWORK_FILENAME}, but failed to load content or it was empty.")
    else:
        logging.warning(f"{BEST_NETWORK_FILENAME} not found.")

    # --- Attempt 2 (Fallback): Generate from Agent Policy ---
    if comparators is None: # If loading from CSV failed or file not found
        logging.info(f"Fallback: Attempting to evaluate agent's policy from {paths['model_path']}")
        if not os.path.exists(paths['model_path']):
             logging.error(f"Model file ({MODEL_FILENAME}) not found at {paths['model_path']}. Cannot evaluate agent policy.")
             return None, "Agent Policy Output (Error: Model Missing)"

        logging.info("Evaluating agent's learned policy...")
        try:
            # The Evaluator's __init__ should have loaded the model if it exists
            comparators, _ = evaluator.evaluate_policy()
            source_description = "Agent Policy Output (Fallback)"
            if comparators is None:
                logging.error("Failed to generate network using agent policy (policy evaluation returned None).")
        except FileNotFoundError:
            logging.error(f"Model file ({MODEL_FILENAME}) not found at {paths['model_path']} during Evaluator init. Cannot evaluate agent policy.")
            return None, "Agent Policy Output (Error: Model Missing)"
        except Exception as e:
             logging.error(f"Error during policy evaluation run: {e}", exc_info=True)
             comparators = None
             source_description = "Agent Policy Output (Error: Exception during eval)"

    # --- Final Check ---
    if comparators is None:
        logging.error("Could not load best network from CSV nor generate one from the agent's policy.")
        source_description = "Error: No network obtained"

    return comparators, source_description

# Function _analyze_and_visualize with the print statement for pruned visualization corrected
# def _analyze_and_visualize(comparators: List[Tuple[int, int]], n_wires: int, source_desc: str, prune_flag: bool) -> None:
#     """Performs analysis, visualization, depth calculation, and optional pruning."""
#     logging.info(f"\n--- Network Analysis ({source_desc}) ---")
#     original_length = len(comparators)
#     logging.info(f"Original Length: {original_length}")
#     logging.info("Network Steps:")
#     for i, comp in enumerate(comparators):
#         print(f"  Step {i+1}: {comp}")
#
#     is_valid = is_sorting_network(n_wires, comparators)
#     valid_str = "VALID" if is_valid else "INVALID"
#     logging.info(f"Network Status: {valid_str}")
#
#     original_depth = calculate_network_depth(n_wires, comparators)
#     logging.info(f"Original Depth (Sequential Dependency): {original_depth}")
#
#     logging.info("\nVisualization (Original):")
#     print(format_network_visualization(comparators, n_wires)) # Print original visualization
#
#     if prune_flag:
#         if not is_valid:
#             logging.warning("Skipping pruning because the network is not valid.")
#         else:
#             logging.info("\nAttempting to prune network...")
#             pruned_comparators = prune_redundant_comparators(n_wires, comparators)
#             pruned_length = len(pruned_comparators)
#             if pruned_length < original_length:
#                 logging.info(f"Pruning successful! Reduced length to: {pruned_length}")
#                 pruned_depth = calculate_network_depth(n_wires, pruned_comparators)
#                 logging.info(f"Pruned Depth (Sequential Dependency): {pruned_depth}")
#                 logging.info("\nVisualization (Pruned):")
#                 print(format_network_visualization(pruned_comparators, n_wires))
#             else:
#                 logging.info("Pruning did not find any redundant comparators.")
#                 logging.info(f"Pruned Depth (Sequential Dependency): {original_depth} (Unchanged)")

# In scripts/evaluate.py

# Assume logger is configured via logging.basicConfig at the start of main()
# import logging # Make sure it's imported

def _analyze_and_visualize(comparators: List[Tuple[int, int]], n_wires: int, source_desc: str, prune_flag: bool) -> None:
    """Performs analysis, visualization, depth calculation, and optional pruning using logging."""
    logging.info(f"\n--- Network Analysis ({source_desc}) ---") # Add newline for spacing
    original_length = len(comparators)
    logging.info(f"Original Length: {original_length}")

    # Log Network Steps using logging
    steps_str = "Network Steps:\n" # Start multi-line string
    for i, comp in enumerate(comparators):
        steps_str += f"  Step {i+1}: {comp}\n" # Append each step
    logging.info(steps_str.strip()) # Log the complete steps string (strip trailing newline)

    # Check validity
    is_valid = is_sorting_network(n_wires, comparators)
    valid_str = "VALID" if is_valid else "INVALID"
    logging.info(f"Network Status: {valid_str}")

    # Calculate and log original depth
    original_depth = calculate_network_depth(n_wires, comparators)
    logging.info(f"Original Depth (Sequential Dependency): {original_depth}")

    # Log original visualization
    logging.info("\nVisualization (Original):") # Add newline before viz
    vis_original_str = format_network_visualization(comparators, n_wires)
    # Log each line of the visualization separately or as a single multi-line message
    # Logging as a single message preserves formatting better in some terminals/files
    logging.info(f"\n{vis_original_str}") # Add newline before the block

    # --- Pruning Section ---
    if prune_flag:
        if not is_valid:
            logging.warning("Skipping pruning because the network is not valid.")
        else:
            logging.info("\nAttempting to prune network...") # Add newline
            pruned_comparators = prune_redundant_comparators(n_wires, comparators)
            pruned_length = len(pruned_comparators)

            if pruned_length < original_length:
                logging.info(f"Pruning successful! Reduced length to: {pruned_length}")
                # Calculate and log pruned depth
                pruned_depth = calculate_network_depth(n_wires, pruned_comparators)
                logging.info(f"Pruned Depth (Sequential Dependency): {pruned_depth}")

                # Log pruned steps (optional, but consistent)
                pruned_steps_str = "Pruned Network Steps:\n"
                for i, comp in enumerate(pruned_comparators):
                    pruned_steps_str += f"  Step {i+1}: {comp}\n"
                logging.info(pruned_steps_str.strip())

                # Log pruned visualization
                logging.info("\nVisualization (Pruned):") # Add newline
                vis_pruned_str = format_network_visualization(pruned_comparators, n_wires)
                logging.info(f"\n{vis_pruned_str}") # Add newline before the block
            else:
                logging.info("Pruning did not find any redundant comparators.")
                # Log unchanged depth
                logging.info(f"Pruned Depth (Sequential Dependency): {original_depth} (Unchanged)")

# --- Main Execution ---

def main():
    # 1. Set up argument parser (only -n, -cdir, --prune)
    parser = argparse.ArgumentParser(
        description=f"Evaluates sorting network: Tries '{BEST_NETWORK_FILENAME}' first, "
                    f"falls back to agent policy from '{MODEL_FILENAME}'."
    )
    parser.add_argument(
        "-n", "--num_wires",
        type=int,
        default=None,
        help="Number of wires (n) to evaluate. If omitted, you will be prompted."
    )
    parser.add_argument(
        "-cdir", "--configs_dir",
        type=str,
        default=DEFAULT_CONFIGS_DIR,
        help=f"Path to the directory containing config files (default: {DEFAULT_CONFIGS_DIR})"
    )
    parser.add_argument(
        "--no-prune",
        action="store_false", # Sets args.prune to False if flag is present
        dest="prune",         # Store the value in 'args.prune'
        default=True,         # Default value for args.prune is True
        help="Disable pruning of the evaluated network (pruning is enabled by default)."
    )
    parser.add_argument(
        "-agent", "--agent_type",
        type=str,
        default=None,
        choices=["double_dqn", "classic_dqn"],
        help="Type of DQN agent results to evaluate ('double_dqn' or 'classic_dqn'). If omitted, you will be prompted."
    )

    args = parser.parse_args()

    # 2. Set up basic console logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    # 3. Determine 'n'
    if args.num_wires is not None:
        n_to_run = args.num_wires
        logging.info(f"Number of wires specified via argument: n={n_to_run}")
        if not (1 <= n_to_run <= 17):
             logging.error(f"Invalid value for -n/--num_wires: {n_to_run}.")
             sys.exit(1)
    else:
        n_to_run = get_n_from_user()

    if args.agent_type is not None:
        agent_to_run = args.agent_type
        logging.info(f"Agent type specified via argument: {agent_to_run}")
    else:
        agent_to_run = get_agent_type_from_user(default_agent="double_dqn")

    logging.info(f"Using agent type: {agent_to_run}")

    try:
        # 4. Validate paths, load config, resolve dirs
        paths, config = _validate_paths_and_load_config(n_to_run, agent_to_run, args.configs_dir, project_root)
        n_wires = config['environment']['n_wires']

        # 5. Initialize Evaluator
        # Must provide model_path for potential fallback and config loading by Evaluator
        evaluator = Evaluator(config, paths["model_path"])

        # 6. Try loading best CSV, fall back to generating from policy
        comparators_to_evaluate, source_description = _get_network_to_evaluate(evaluator, paths)

        # 7. Analyze and visualize if a network was obtained
        if comparators_to_evaluate is not None:
            _analyze_and_visualize(comparators_to_evaluate, n_wires, source_description, args.prune)
        else:
            # Error message already logged by _get_network_to_evaluate
            logging.error("Evaluation cannot proceed as no network comparators were loaded or generated.")
            sys.exit(1) # Exit explicitly if no network could be obtained

    # --- Error Handling ---
    except FileNotFoundError as e:
        logging.error(f"File not found during setup or evaluation: {e}", exc_info=False)
        sys.exit(1)
    except ValueError as e:
        logging.error(f"Configuration, value, or argument error: {e}", exc_info=False)
        sys.exit(1)
    except IOError as e:
        logging.error(f"IO error during evaluation: {e}", exc_info=False)
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred during evaluation: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()