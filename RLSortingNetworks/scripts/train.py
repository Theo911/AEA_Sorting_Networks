import argparse
import logging
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) # Path to the project root directory
sys.path.insert(0, project_root)

from RLSortingNetworks.sorting_network_rl.utils.config_loader import load_config
from RLSortingNetworks.sorting_network_rl.utils.logging_setup import setup_logging
from RLSortingNetworks.sorting_network_rl.core.trainer import Trainer

DEFAULT_CONFIG_PATH = os.path.join(project_root, "configs", "default_config.yaml")

def main():
    parser = argparse.ArgumentParser(description="Train a DQN agent to find sorting networks.")
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to the YAML configuration file (default: {DEFAULT_CONFIG_PATH})"
    )
    args = parser.parse_args()

    # Setup basic console logging early
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    # Load configuration
    try:
        config = load_config(args.config)
        logging.info(f"Loaded configuration template from: {args.config}")
    except (FileNotFoundError, Exception) as e:
        logging.error(f"Error loading configuration: {e}")
        sys.exit(1)

    # --- START: Resolve Checkpoint Base Directory ---
    try:
        exp_config = config.get('experiment', {})
        base_dir_relative = exp_config.get('base_dir', 'checkpoints') # Get path from config

        # Check if the path in config is already absolute
        if os.path.isabs(base_dir_relative):
            absolute_base_dir = base_dir_relative
            logging.info(f"Using absolute checkpoint base directory from config: {absolute_base_dir}")
        else:
            # Construct absolute path relative to the project root
            absolute_base_dir = os.path.abspath(os.path.join(project_root, base_dir_relative))
            logging.info(f"Resolved relative checkpoint base directory '{base_dir_relative}' to absolute path: {absolute_base_dir}")

        # Update the config dictionary IN MEMORY with the absolute path
        config['experiment']['base_dir'] = absolute_base_dir

    except KeyError:
        logging.error("Configuration file is missing the 'experiment' section or 'base_dir'.")
        sys.exit(1)
    except Exception as e:
         logging.error(f"Error processing checkpoint base directory: {e}")
         sys.exit(1)


    try:
        trainer = Trainer(config)

        setup_logging(trainer.log_path, level=logging.INFO)
        logging.info("File logging initialized.") # Log after setup

        trainer.train()
    except Exception as e:
        logging.error("An unexpected error occurred during trainer initialization or training.", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()