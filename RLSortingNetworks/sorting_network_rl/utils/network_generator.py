import os
import logging
from typing import List, Tuple, Optional, Dict, Any

# Adjust imports to be relative within the package or absolute from project root
# Assuming the project root (containing RLSortingNetworks) is in PYTHONPATH
# or handled by the caller (webapp)
try:
    # Try absolute imports first (if RLSortingNetworks is in PYTHONPATH)
    from RLSortingNetworks.sorting_network_rl.core.evaluator import Evaluator
    from RLSortingNetworks.sorting_network_rl.utils.config_loader import load_config
except ImportError:
    # Fallback to relative imports if run as part of the package
    # This might be less robust depending on how it's called
    try:
        from ..core.evaluator import Evaluator
        from .config_loader import load_config
    except ImportError as e:
        print(f"Error importing RL modules: {e}. Ensure RLSortingNetworks is in PYTHONPATH or structured correctly.")
        # Re-raise or handle appropriately, maybe Evaluator = None
        raise

logger = logging.getLogger(__name__)

# --- Constants ---
CONFIG_FILENAME_TEMPLATE = "config_n{n_value}.yaml"
MODEL_FILENAME = "model.pt"
DEFAULT_CONFIGS_DIR_NAME = "configs"
DEFAULT_CHECKPOINTS_DIR_NAME = "checkpoints"

def _get_config_path(n_value: int, rl_project_root: str) -> str:
    """Constructs the expected path for the config file based on n."""
    configs_dir = os.path.join(rl_project_root, DEFAULT_CONFIGS_DIR_NAME)
    config_filename = CONFIG_FILENAME_TEMPLATE.format(n_value=n_value)
    return os.path.join(configs_dir, config_filename)

def _get_model_path(n_value: int, max_steps: int, rl_project_root: str, checkpoints_base_dir: Optional[str] = None) -> str:
    """Constructs the expected path for the model file based on n and max_steps."""
    if checkpoints_base_dir is None:
        checkpoints_base_dir = os.path.join(rl_project_root, DEFAULT_CHECKPOINTS_DIR_NAME)
    run_id = f"{n_value}w_{max_steps}s"
    run_dir = os.path.join(checkpoints_base_dir, run_id)
    return os.path.join(run_dir, MODEL_FILENAME)

def get_rl_network(n_wires: int, rl_project_root: str) -> Optional[List[Tuple[int, int]]]:
    """
    Generates a sorting network for n_wires using the pre-trained RL model.

    Args:
        n_wires (int): The number of wires for the sorting network.
        rl_project_root (str): The absolute path to the RLSortingNetworks project directory.

    Returns:
        Optional[List[Tuple[int, int]]]: A list of comparators defining the network,
                                         or None if generation fails.
    """
    logger.info(f"Attempting to generate RL network for n={n_wires} using root: {rl_project_root}")

    try:
        # 1. Load Configuration
        config_path = _get_config_path(n_wires, rl_project_root)
        logger.info(f"Loading config from: {config_path}")
        if not os.path.exists(config_path):
            logger.error(f"Config file not found: {config_path}")
            return None
        config = load_config(config_path)

        # Validate essential config parts
        env_config = config.get('environment', {})
        loaded_n = env_config.get('n_wires')
        max_steps = env_config.get('max_steps')
        if loaded_n is None or max_steps is None:
            logger.error("Config missing 'environment.n_wires' or 'environment.max_steps'.")
            return None
        if loaded_n != n_wires:
            logger.warning(f"Config mismatch: Requested n={n_wires}, but config has n={loaded_n}. Using config value.")
            # n_wires = loaded_n # Use the value from config

        # Resolve absolute checkpoints base dir from config if specified, otherwise default
        exp_config = config.get('experiment', {})
        base_dir_relative = exp_config.get('base_dir') # Can be None
        if base_dir_relative and os.path.isabs(base_dir_relative):
             absolute_checkpoints_dir = base_dir_relative
        elif base_dir_relative: # Relative path provided
             absolute_checkpoints_dir = os.path.abspath(os.path.join(rl_project_root, base_dir_relative))
        else: # Not specified, use default relative to project root
            absolute_checkpoints_dir = os.path.join(rl_project_root, DEFAULT_CHECKPOINTS_DIR_NAME)
        logger.info(f"Using absolute checkpoints directory: {absolute_checkpoints_dir}")


        # 2. Determine Model Path
        model_path = _get_model_path(n_wires, max_steps, rl_project_root, absolute_checkpoints_dir)
        logger.info(f"Expecting model at: {model_path}")
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return None

        # 3. Initialize Evaluator
        # Ensure Evaluator handles potential errors during init (like missing model)
        evaluator = Evaluator(config, model_path)

        # 4. Generate Network using Policy
        comparators, is_valid = evaluator.evaluate_policy()

        if comparators is None:
            logger.error(f"Failed to generate network for n={n_wires} using policy.")
            return None

        logger.info(f"Successfully generated network for n={n_wires}. Length: {len(comparators)}, Valid: {is_valid}")
        # Optional: Add pruning here if desired, though maybe better done by caller
        # comparators = prune_redundant_comparators(n_wires, comparators)
        # logger.info(f"Pruned network length: {len(comparators)}")

        return comparators

    except FileNotFoundError as e:
        logger.error(f"File not found during RL network generation: {e}", exc_info=False)
        return None
    except ValueError as e:
        logger.error(f"Configuration or value error during RL network generation: {e}", exc_info=False)
        return None
    except ImportError as e:
        logger.error(f"Import error during RL network generation - check environment/PYTHONPATH: {e}", exc_info=False)
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during RL network generation: {e}", exc_info=True)
        return None

# Example Usage (for testing this module directly)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    # --- !!! IMPORTANT !!! ---
    # Determine the RL project root relative to this script's location
    # This assumes network_generator.py is in RLSortingNetworks/sorting_network_rl/utils/
    current_dir = os.path.dirname(os.path.abspath(__file__))
    rl_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..')) # Adjust based on actual depth
    print(f"Determined RL Project Root for testing: {rl_root}")

    test_n = 8 # Example: Generate network for n=8
    generated_network = get_rl_network(test_n, rl_root)

    if generated_network:
        print(f"\nGenerated Network (n={test_n}):")
        for i, comp in enumerate(generated_network):
            print(f"  Step {i+1}: {comp}")
        print(f"Total Comparators: {len(generated_network)}")

        # You could also try visualizing it here if format_network_visualization is available
        try:
            from RLSortingNetworks.sorting_network_rl.utils.evaluation import format_network_visualization
            print("\nText Visualization:")
            print(format_network_visualization(generated_network, test_n))
        except ImportError:
            print("Could not import format_network_visualization for testing.")

    else:
        print(f"\nFailed to generate network for n={test_n}.") 