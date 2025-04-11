import os
import time
import logging
import csv
import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional

from RLSortingNetworks.sorting_network_rl.env.sorting_env import SortingNetworkEnv
from RLSortingNetworks.sorting_network_rl.agent.dqn_agent import DQNAgent
from RLSortingNetworks.sorting_network_rl.agent.dqn_classic_agent import DQNAgent_Classic
from RLSortingNetworks.sorting_network_rl.utils.state_encoder import encode_state
from RLSortingNetworks.sorting_network_rl.utils.evaluation import is_sorting_network, prune_redundant_comparators, format_network_visualization
from RLSortingNetworks.sorting_network_rl.utils.config_loader import save_config

logger = logging.getLogger(__name__)

class Trainer:
    """Handles the training loop and associated tasks for the DQN agent."""

    def __init__(self, config: Dict[str, Any]):
        """Initializes the Trainer.

        Args:
            config (Dict[str, Any]): The configuration dictionary.
        """
        self.config = config
        self.env_cfg = config['environment']
        self.train_cfg = config['training']
        self.agent_cfg = config['agent']
        self.reward_cfg = config['reward']
        self.exp_cfg = config['experiment']

        # --- Setup Environment and Agent ---
        self.env = SortingNetworkEnv(n_wires=self.env_cfg['n_wires'], max_steps=self.env_cfg['max_steps'])
        self.state_dim = encode_state(self.env.n_wires, self.env.max_steps, []).shape[0]
        self.action_dim = self.env.get_action_space_size()

        # DQN Agent with 2 neural networks (policy and target)
        self.agent = DQNAgent(self.state_dim, self.action_dim, config)

        # Uncomment the following line to use a classic DQN agent (for comparison)
        # Classic DQN agent: uses a single neural network for Q-value approximation (for comparison purposes)
        # self.agent = DQNAgent_Classic(self.state_dim, self.action_dim, config)

        # --- Setup Experiment Directory and Paths ---
        self._setup_paths()

        # --- State Variables ---
        self.reward_history: List[float] = []
        self.step_history: List[int] = []
        self.best_network: Optional[List[Tuple[int, int]]] = None
        self.best_network_len: int = float('inf')
        self.start_episode: int = 1

    def _setup_paths(self) -> None:
        """Creates directories and defines paths for saving artifacts.
           Run directory name is based only on n_wires and max_steps.
        """
        n_wires = self.env_cfg['n_wires']
        max_steps = self.env_cfg['max_steps']

        self.run_id = f"{n_wires}w_{max_steps}s"

        self.run_dir = os.path.join(self.exp_cfg['base_dir'], self.run_id)
        # Create the run directory if it doesn't exist
        os.makedirs(self.run_dir, exist_ok=True)
        logger.info(f"Experiment artifacts directory: {self.run_dir}")

        self.model_path = os.path.join(self.run_dir, "model.pt")
        self.epsilon_path = os.path.join(self.run_dir, "epsilon.npy")
        self.config_path = os.path.join(self.run_dir, "config.yaml")
        self.log_path = os.path.join(self.run_dir, "training.log")
        self.results_csv_path = os.path.join(self.run_dir, "results.csv")
        self.best_network_csv_path = os.path.join(self.run_dir, "best_network.csv")

    def _load_checkpoint(self) -> None:
        """Loads agent state from checkpoint files if they exist."""
        if os.path.exists(self.model_path):
            try:
                self.agent.load_model(self.model_path)
                self.agent.load_epsilon(self.epsilon_path)
                # Attempt to resume episode count from results CSV (simple approach)
                if os.path.exists(self.results_csv_path):
                     try:
                         # Read last line, handle potential errors/empty file
                         with open(self.results_csv_path, 'r') as f:
                             lines = f.readlines()
                             if lines: # Check if file is not empty
                                 last_line = lines[-1]
                                 self.start_episode = int(last_line.split(',')[0]) + 1
                             else:
                                 self.start_episode = 1
                     except (IndexError, ValueError, FileNotFoundError):
                          logger.warning(f"Could not determine resume episode from {self.results_csv_path}. Starting from 1.")
                          self.start_episode = 1 # Fallback if reading fails
                     except Exception as e:
                          logger.error(f"Unexpected error reading {self.results_csv_path}: {e}. Starting from 1.")
                          self.start_episode = 1
                else:
                    self.start_episode = 1 # No results file, start from 1

                logger.info(f"Checkpoint loaded from {self.run_dir}. Resuming training from episode {self.start_episode}.")

            except (FileNotFoundError, IOError) as e:
                logger.warning(f"Checkpoint found but failed to load ({e}). Starting fresh.")
                self.start_episode = 1
        else:
            logger.info(f"No checkpoint found at {self.model_path}. Starting fresh training.")
            self.start_episode = 1
            # Ensure log/results files are cleared if starting fresh implicitly because no checkpoint exists
            if os.path.exists(self.results_csv_path):
                 try: os.remove(self.results_csv_path)
                 except OSError as e: logger.error(f"Could not remove old results file {self.results_csv_path}: {e}")
            # The FileHandler for logging is typically set to append mode 'a',
            # so old logs might still be appended to if the file exists.
            # If strict fresh start is needed, the log file could also be removed here.
            # if os.path.exists(self.log_path):
            #     try: os.remove(self.log_path)
            #     except OSError as e: logger.error(f"Could not remove old log file {self.log_path}: {e}")


    def _save_checkpoint(self, episode: int) -> None:
        """Saves the current agent state and epsilon value."""
        if not self.train_cfg.get('save_checkpoints', True):
            return
        try:
            self.agent.save_model(self.model_path)
            self.agent.save_epsilon(self.epsilon_path)
            logger.debug(f"Checkpoint saved at episode {episode}.")
        except Exception as e:
            logger.error(f"Failed to save checkpoint at episode {episode}: {e}")

    def _save_best_network(self, episode: int) -> None:
        """Saves the best network found so far to a CSV file."""
        if not self.best_network:
            return
        try:
            with open(self.best_network_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([f"Best network found at episode {episode}, Length: {self.best_network_len}"])
                for comp in self.best_network:
                    writer.writerow([comp[0], comp[1]])
            logger.info(f"Best network (length {self.best_network_len}) saved to {self.best_network_csv_path}")
        except Exception as e:
            logger.error(f"Failed to save best network: {e}")

    def _log_episode_results(self, episode: int, total_reward: float, steps: int, loss: Optional[float]) -> None:
        """Logs episode summary statistics to console and CSV file."""
        self.reward_history.append(total_reward)
        self.step_history.append(steps)

        # Log progress periodically
        if episode % self.train_cfg['print_every'] == 0:
            avg_reward = np.mean(self.reward_history[-self.train_cfg['print_every']:])
            avg_steps = np.mean(self.step_history[-self.train_cfg['print_every']:])
            best_len_str = str(self.best_network_len) if self.best_network_len != float('inf') else 'N/A'
            loss_str = f"{loss:.4f}" if loss is not None else "N/A"

            logger.info(
                f"Ep {episode:<6} | Steps: {steps:<3} | Rwd: {total_reward:<8.2f} | "
                f"Avg Rwd (last {self.train_cfg['print_every']}): {avg_reward:<8.2f} | "
                f"Avg Steps: {avg_steps:<5.1f} | Eps: {self.agent.epsilon:.3f} | "
                f"Best Len: {best_len_str:<4} | Loss: {loss_str}"
            )

            # Write to CSV (append mode)
            try:
                # Check if file exists to write header
                file_exists = os.path.exists(self.results_csv_path)
                with open(self.results_csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    if not file_exists or os.path.getsize(self.results_csv_path) == 0:
                         writer.writerow(["Episode", "TotalReward", "Steps", "AvgReward", "AvgSteps", "Epsilon", "BestNetworkLen", "Loss"]) # Header
                    writer.writerow([episode, total_reward, steps, avg_reward, avg_steps, self.agent.epsilon, best_len_str, loss_str])
            except Exception as e:
                logger.error(f"Failed to write results to CSV: {e}")


    def train(self) -> None:
        """Runs the main training loop."""
        logger.info("Starting training...")
        save_config(self.config, self.config_path) # Save config for this run
        logger.info(f"Configuration for this run saved to {self.config_path}")

        if not self.train_cfg['start_fresh']:
            self._load_checkpoint()
        else:
             # Ensure log/results files are cleared if starting fresh
             if os.path.exists(self.results_csv_path): os.remove(self.results_csv_path)

        num_episodes = self.train_cfg['num_episodes']
        target_update_freq = self.train_cfg['target_update_freq']

        last_loss = None
        for episode in range(self.start_episode, num_episodes + 1):
            state_info = self.env.reset()
            state_vector = encode_state(self.env.n_wires, self.env.max_steps, state_info['comparators'])
            done = False
            episode_reward = 0.0
            episode_steps = 0

            while not done:
                # Select action
                action = self.agent.select_action(state_vector)

                # Take step in environment
                next_state_info, _, env_done = self.env.step(action) # Basic reward is 0
                current_comparators = next_state_info['comparators']
                next_state_vector = encode_state(self.env.n_wires, self.env.max_steps, current_comparators)
                num_steps = self.env.get_current_step()

                # --- Calculate Reward (Scaled Terminal) ---
                reward = self.reward_cfg['step_penalty']
                is_successful_sort = False

                # Check termination conditions and assign terminal rewards
                if is_sorting_network(self.env.n_wires, current_comparators):
                    reward = self.reward_cfg['success_base'] / num_steps
                    done = True
                    is_successful_sort = True
                    # Check if this is the best network found so far
                    if num_steps < self.best_network_len:
                        self.best_network_len = num_steps
                        self.best_network = current_comparators.copy()
                        logger.info(f"** New best network found! Length: {self.best_network_len}, Episode: {episode} **")
                        self._save_best_network(episode)

                elif env_done: # Reached max_steps without success
                    reward = self.reward_cfg['failure']
                    done = True

                # Store transition in replay buffer
                self.agent.store_transition(state_vector, action, reward, next_state_vector, done)

                # Perform training step on agent
                loss = self.agent.train_step()
                if loss is not None: last_loss = loss # Keep track of last valid loss

                # Update state and episode stats
                state_vector = next_state_vector
                episode_reward += reward
                episode_steps = num_steps

                if done:
                    break

            # Post-episode updates
            self.agent.decay_epsilon()
            self._log_episode_results(episode, episode_reward, episode_steps, last_loss)

            # Optional: Comment if using classic DQN agent
            # Update target network periodically
            if episode % target_update_freq == 0:
                self.agent.update_target_network()

            # Save checkpoint periodically (or based on other conditions)
            if episode % self.train_cfg['print_every'] == 0: # Save checkpoint when printing progress
                self._save_checkpoint(episode)


        logger.info("Training complete.")
        logger.info(f"Best network found has length: {self.best_network_len if self.best_network_len != float('inf') else 'None found'}")

        # Final save
        self._save_checkpoint(num_episodes)
        if self.best_network:
             logger.info("Final best network visualization:")
             logger.info("\n" + format_network_visualization(self.best_network, self.env.n_wires))
             # Optional: Prune the final best network
             pruned_best = prune_redundant_comparators(self.env.n_wires, self.best_network)
             if len(pruned_best) < self.best_network_len:
                  logger.info(f"Pruned best network length: {len(pruned_best)}")
                  logger.info("\n" + format_network_visualization(pruned_best, self.env.n_wires))
             else:
                   logger.info("No further pruning possible on the best network.")