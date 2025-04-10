import argparse
import logging
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from RLSortingNetworks.sorting_network_rl.utils.config_loader import load_config
from RLSortingNetworks.sorting_network_rl.utils.logging_setup import setup_logging
from RLSortingNetworks.sorting_network_rl.core.trainer import Trainer

# --- Constants ---
# Default location for configuration files relative to the project root
DEFAULT_CONFIGS_DIR = os.path.join(project_root, "configs")
DEFAULT_CHECKPOINTS_DIR_NAME = "checkpoints"

# --- Helper Functions ---
def get_config_path_for_n(n_value: int, configs_dir: str) -> str:
    """Constructs the expected path for the config file based on n.

    Args:
        n_value (int): The number of wires.
        configs_dir (str): The directory containing the configuration files.

    Returns:
        str: The constructed path to the configuration file (e.g., 'configs/config_n4.yaml').
    """
    config_filename = f"config_n{n_value}.yaml"
    return os.path.join(configs_dir, config_filename)

def get_n_from_user(min_n: int = 1, max_n: int = 17) -> int:
    """Prompts the user to enter the number of wires (n) interactively.

    Args:
        min_n (int): Minimum allowed value for n.
        max_n (int): Maximum allowed value for n.

    Returns:
        int: The validated number of wires entered by the user.

    Raises:
        SystemExit: If the user cancels the input (e.g., Ctrl+D).
    """
    while True:
        try:
            # Prompt the user for input
            n_input = input(f"Enter the number of wires (n) between {min_n} and {max_n}: ")
            # Attempt to convert input to an integer
            n_value = int(n_input)
            # Validate the input range
            if min_n <= n_value <= max_n:
                return n_value  # Return the valid input
            else:
                print(f"Error: Please enter a number between {min_n} and {max_n}.")
        except ValueError:
            # Handle cases where input is not a valid integer
            print("Error: Invalid input. Please enter an integer.")
        except EOFError:
            # Handle cases where input stream is closed (e.g., Ctrl+D)
             print("\nInput cancelled by user.")
             sys.exit(0) # Exit gracefully

# --- Main Execution ---

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train a DQN agent to find sorting networks.")
    parser.add_argument(
        "-n", "--num_wires",
        type=int,
        # Argument is now optional; prompt if omitted
        help="Number of wires (n) to train for (e.g., 3, 4, ...). If omitted, you will be prompted."
    )
    parser.add_argument(
        "-cdir", "--configs_dir",
        type=str,
        default=DEFAULT_CONFIGS_DIR,
        help=f"Path to the directory containing configuration files (default: {DEFAULT_CONFIGS_DIR})"
    )
    args = parser.parse_args()

    # Set up basic console logging initially
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    # --- Determine the value of 'n' to run ---
    if args.num_wires is not None:
        # Use the value provided via command-line argument
        n_to_run = args.num_wires
        logging.info(f"Number of wires specified via argument: n={n_to_run}")
        # Optional: Validate the command-line argument value
        if not (1 <= n_to_run <= 17): # Adjust range as needed
             logging.error(f"Invalid value for -n/--num_wires: {n_to_run}. Must be between 1 and 17.")
             sys.exit(1)
    else:
        # Prompt the user interactively if -n was not provided
        n_to_run = get_n_from_user()
        logging.info(f"Number of wires obtained from user input: n={n_to_run}")

    # --- Determine the path to the n-specific configuration file ---
    config_path = get_config_path_for_n(n_to_run, args.configs_dir)
    logging.info(f"Attempting to load configuration from: {config_path}")

    # --- Load the specific configuration file ---
    try:
        config = load_config(config_path)
        # Optional: Verify consistency between requested 'n' and 'n_wires' in the loaded config
        loaded_n = config.get('environment', {}).get('n_wires')
        if loaded_n is not None and loaded_n != n_to_run:
            logging.warning(f"Mismatch: Requested n={n_to_run} but config file {config_path} specifies n_wires={loaded_n}. "
                            f"Using value from config file (n={loaded_n}).")
            n_to_run = loaded_n # Update n_to_run to match the config file content
            # Alternatively, exit if strict matching is required:
            # logging.error("Configuration file n_wires does not match requested n.")
            # sys.exit(1)
        elif loaded_n is None:
             logging.warning(f"Config file {config_path} is missing 'environment.n_wires'. Proceeding with requested n={n_to_run}.")
             # Ensure the config dictionary *has* the required n_wires value if it was missing
             if 'environment' not in config: config['environment'] = {}
             config['environment']['n_wires'] = n_to_run

        logging.info(f"Successfully loaded configuration for n={config.get('environment', {}).get('n_wires')}.")

    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        logging.error(f"Please ensure a configuration file named 'config_n{n_to_run}.yaml' exists in '{args.configs_dir}'.")
        sys.exit(1)
    except Exception as e:
        # Catch other potential errors during config loading (e.g., YAML parsing errors)
        logging.error(f"Error loading configuration from {config_path}: {e}")
        sys.exit(1)

    # --- Resolve the absolute path for the checkpoint/output directory ---
    try:
        exp_config = config.get('experiment', {})
        # Get the base directory path from config, default to a standard name if missing
        base_dir_relative = exp_config.get('base_dir', DEFAULT_CHECKPOINTS_DIR_NAME)

        # Check if the path is already absolute
        if os.path.isabs(base_dir_relative):
            absolute_base_dir = base_dir_relative
            logging.info(f"Using absolute checkpoint base directory from config: {absolute_base_dir}")
        else:
            # Construct absolute path relative to the project root
            absolute_base_dir = os.path.abspath(os.path.join(project_root, base_dir_relative))
            logging.info(f"Resolved relative checkpoint base directory '{base_dir_relative}' to absolute path: {absolute_base_dir}")

        # Update the config dictionary (in memory) with the absolute path
        # This ensures the Trainer uses the correct location regardless of CWD
        config['experiment']['base_dir'] = absolute_base_dir

    except Exception as e:
         # Catch errors during path processing
         logging.error(f"Error processing checkpoint base directory: {e}")
         sys.exit(1)

    # --- Initialize and Run the Trainer ---
    try:
        # Pass the loaded and processed configuration to the Trainer
        trainer = Trainer(config)

        # Set up file logging (the path is now determined correctly by the Trainer)
        setup_logging(trainer.log_path, level=logging.INFO)
        logging.info("File logging initialized.")

        # Start the training process
        trainer.train()

    except Exception as e:
        # Catch any unexpected errors during Trainer initialization or the training loop
        logging.error("An unexpected error occurred during trainer initialization or training.", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    # Entry point of the script
    main()