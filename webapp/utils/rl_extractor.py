"""
Enhanced RL Data Extraction Utility

Scans RLSortingNetworks/checkpoints/ directory to extract performance data
from trained RL models with support for Classic vs Double DQN separation.
"""

import os
import sys
import glob
import csv
import logging
from typing import Dict, List, Tuple, Optional

# Add path for RLSortingNetworks modules
webapp_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
project_root = os.path.dirname(webapp_dir)
rl_dir = os.path.join(project_root, 'RLSortingNetworks')
sys.path.insert(0, rl_dir)

try:
    from sorting_network_rl.utils.evaluation import calculate_network_depth, prune_redundant_comparators, is_sorting_network
    RL_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Could not import RL evaluation modules: {e}")
    RL_AVAILABLE = False

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set to INFO level to reduce debug noise

class RLCheckpointExtractor:
    """Enhanced checkpoint extraction with Classic/Double DQN separation"""
    
    def __init__(self, checkpoints_dir=None):
        if checkpoints_dir is None:
            webapp_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            project_root = os.path.dirname(webapp_dir)
            self.checkpoints_dir = os.path.join(project_root, 'RLSortingNetworks', 'checkpoints')
        else:
            self.checkpoints_dir = checkpoints_dir
    
    def extract_classic_raw_data(self):
        """Extract raw Classic DQN checkpoint data"""
        return self._extract_by_training_method('classic')
    
    def extract_double_raw_data(self):
        """Extract raw Double DQN checkpoint data"""
        return self._extract_by_training_method('double')
    
    def _extract_by_training_method(self, method_type='classic'):
        """Extract data for specific training method"""
        method_data = {'size': {}, 'depth': {}}
        
        if not RL_AVAILABLE:
            logging.warning("RL modules not available for checkpoint extraction")
            return method_data
        
        if not os.path.exists(self.checkpoints_dir):
            logging.warning(f"Checkpoints directory not found: {self.checkpoints_dir}")
            return method_data
        
        # Find directories that match the training method
        pattern = f"*{method_type}*" if method_type != 'classic' else "*"
        checkpoint_dirs = glob.glob(os.path.join(self.checkpoints_dir, pattern))
        
        for checkpoint_dir in checkpoint_dirs:
            if not os.path.isdir(checkpoint_dir):
                continue
            
            # Skip Double DQN directories if looking for Classic
            if method_type == 'classic' and 'double' in checkpoint_dir.lower():
                continue
            
            # Skip Classic directories if looking for Double
            if method_type == 'double' and 'double' not in checkpoint_dir.lower():
                continue
            
            n = self._extract_n_from_dirname(checkpoint_dir)
            if n is None:
                continue
            
            # Load network from best_network.csv
            csv_path = os.path.join(checkpoint_dir, 'best_network.csv')
            comparators = self._load_best_network_from_csv(csv_path)
            
            if comparators:
                # Calculate raw metrics
                size = len(comparators)
                depth = calculate_network_depth(comparators, n)
                
                method_data['size'][n] = size
                method_data['depth'][n] = depth
        
        return method_data
    
    def validate_against_readme(self, readme_data):
        """Validate checkpoint extractions against README authoritative data"""
        validation_results = {
            'classic_dqn': self._validate_method_data('classic', readme_data.get('rl_classic_raw', {})),
            'double_dqn': self._validate_method_data('double', readme_data.get('rl_double_raw', {}))
        }
        
        return validation_results
    
    def _validate_method_data(self, method_type, readme_method_data):
        """Validate specific method data against README"""
        checkpoint_data = self._extract_by_training_method(method_type)
        
        discrepancies = []
        matches = 0
        total_comparisons = 0
        
        readme_sizes = readme_method_data.get('size', {})
        checkpoint_sizes = checkpoint_data.get('size', {})
        
        # Compare sizes where both have data
        for n in set(readme_sizes.keys()) & set(checkpoint_sizes.keys()):
            readme_size = readme_sizes.get(n)
            checkpoint_size = checkpoint_sizes.get(n)
            
            if readme_size is not None and checkpoint_size is not None:
                total_comparisons += 1
                if readme_size == checkpoint_size:
                    matches += 1
                else:
                    discrepancies.append({
                        'n': n,
                        'metric': 'size',
                        'readme_value': readme_size,
                        'checkpoint_value': checkpoint_size,
                        'difference': abs(readme_size - checkpoint_size)
                    })
        
        consistency_ratio = matches / total_comparisons if total_comparisons > 0 else 0
        
        return {
            'consistency_ratio': consistency_ratio,
            'matches': matches,
            'total_comparisons': total_comparisons,
            'discrepancies': discrepancies,
            'confidence_level': 'high' if consistency_ratio > 0.8 else 'medium' if consistency_ratio > 0.5 else 'low'
        }
    
    def calculate_pruning_impact(self, raw_data, pruned_data):
        """Calculate optimization improvements from pruning"""
        pruning_impact = {}
        
        raw_sizes = raw_data.get('size', {})
        pruned_sizes = pruned_data.get('size', {})
        
        for n in set(raw_sizes.keys()) & set(pruned_sizes.keys()):
            raw_size = raw_sizes.get(n)
            pruned_size = pruned_sizes.get(n)
            
            if raw_size is not None and pruned_size is not None and raw_size > 0:
                reduction = raw_size - pruned_size
                reduction_pct = (reduction / raw_size) * 100
                
                pruning_impact[n] = {
                    'size_reduction': reduction,
                    'size_reduction_percent': reduction_pct,
                    'raw_size': raw_size,
                    'pruned_size': pruned_size,
                    'efficiency_gain': reduction_pct
                }
        
        return pruning_impact
    
    def get_comprehensive_rl_data(self):
        """Get comprehensive RL data for all methods and optimizations"""
        return {
            'classic_raw': self.extract_classic_raw_data(),
            'double_raw': self.extract_double_raw_data(),
            'extraction_timestamp': os.path.getmtime(self.checkpoints_dir) if os.path.exists(self.checkpoints_dir) else None,
            'checkpoints_available': os.path.exists(self.checkpoints_dir),
            'rl_modules_available': RL_AVAILABLE
        }
    
    def _extract_n_from_dirname(self, dirname: str) -> Optional[int]:
        """Extract n_wires value from checkpoint directory name like '4w_10s'"""
        try:
            basename = os.path.basename(dirname)
            if 'w_' in basename:
                n_str = basename.split('w_')[0]
                return int(n_str)
        except (ValueError, IndexError):
            pass
        return None
    
    def _load_best_network_from_csv(self, csv_path: str) -> Optional[List[Tuple[int, int]]]:
        """Load best network comparators from best_network.csv file"""
        if not os.path.exists(csv_path):
            # This is expected - most checkpoints don't have best_network.csv files
            # Only log at debug level, not warning level
            logger.debug(f"best_network.csv not found at {csv_path} (this is normal)")
            return None
        
        try:
            comparators = []
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header line
                
                for row in reader:
                    if len(row) >= 2:
                        i, j = int(row[0]), int(row[1])
                        comparators.append((i, j))
            
            logger.debug(f"Successfully loaded network from {csv_path}")
            return comparators
        except Exception as e:
            logger.warning(f"Error reading CSV file {csv_path}: {e}")
            return None

def extract_n_from_dirname(dirname: str) -> Optional[int]:
    """Extract n_wires value from checkpoint directory name like '4w_10s'"""
    try:
        basename = os.path.basename(dirname)
        if 'w_' in basename:
            n_str = basename.split('w_')[0]
            return int(n_str)
    except (ValueError, IndexError):
        pass
    return None

def load_best_network_from_csv(csv_path: str) -> Optional[List[Tuple[int, int]]]:
    """Load best network comparators from best_network.csv file"""
    if not os.path.exists(csv_path):
        # This is expected - most checkpoints don't have best_network.csv files
        # Only log at debug level, not warning level
        logger.debug(f"best_network.csv not found at {csv_path} (this is normal)")
        return None
    
    try:
        comparators = []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header line
            
            for row in reader:
                if len(row) >= 2:
                    i, j = int(row[0]), int(row[1])
                    comparators.append((i, j))
        
        logger.debug(f"Successfully loaded network from {csv_path}")
        return comparators
    except Exception as e:
        logger.warning(f"Error reading CSV file {csv_path}: {e}")
        return None

def extract_rl_checkpoint_data() -> Dict[int, Dict[str, any]]:
    """
    Scan RLSortingNetworks/checkpoints/ for trained results.
    
    Returns:
        Dict[int, Dict]: {n_wires: {size: X, depth: Y, episode: Z, pruned_size: W, pruned_depth: V}}
    """
    if not RL_AVAILABLE:
        logger.warning("RL modules not available, returning empty data")
        return {}
    
    # Find checkpoints directory
    checkpoints_dir = os.path.join(rl_dir, 'checkpoints')
    if not os.path.exists(checkpoints_dir):
        logger.error(f"Checkpoints directory not found: {checkpoints_dir}")
        return {}
    
    rl_data = {}
    
    # Scan all checkpoint directories
    checkpoint_dirs = glob.glob(os.path.join(checkpoints_dir, '*w_*s'))
    
    for checkpoint_dir in checkpoint_dirs:
        n_wires = extract_n_from_dirname(checkpoint_dir)
        if n_wires is None:
            continue
            
        best_network_path = os.path.join(checkpoint_dir, 'best_network.csv')
        comparators = load_best_network_from_csv(best_network_path)
        
        if comparators is None:
            # This is expected for most checkpoints - they don't have best_network.csv
            # Only log at debug level
            logger.debug(f"Could not load network from {best_network_path} (this is normal)")
            continue
        
        try:
            # Calculate metrics
            size = len(comparators)
            depth = calculate_network_depth(n_wires, comparators)
            
            # Try pruning for additional metrics
            pruned_comparators = prune_redundant_comparators(n_wires, comparators)
            pruned_size = len(pruned_comparators)
            pruned_depth = calculate_network_depth(n_wires, pruned_comparators)
            
            # Extract episode info from header if available
            episode = None
            try:
                with open(best_network_path, 'r') as f:
                    header = f.readline().strip()
                    if 'episode' in header.lower():
                        # Extract episode number from header like "Best network found at episode 369, Length: 5"
                        import re
                        match = re.search(r'episode\s+(\d+)', header, re.IGNORECASE)
                        if match:
                            episode = int(match.group(1))
            except:
                pass
            
            rl_data[n_wires] = {
                'size': size,
                'depth': depth,
                'episode': episode,
                'pruned_size': pruned_size,
                'pruned_depth': pruned_depth,
                'comparators': comparators,
                'pruned_comparators': pruned_comparators,
                'checkpoint_dir': checkpoint_dir
            }
            
            logger.debug(f"Extracted RL data for n={n_wires}: size={size}, depth={depth}")
            
        except Exception as e:
            logger.error(f"Error processing checkpoint {checkpoint_dir}: {e}")
            continue
    
    return rl_data

def get_rl_data_summary() -> Dict[str, any]:
    """Get a summary of available RL data for quick reference"""
    rl_data = extract_rl_checkpoint_data()
    
    if not rl_data:
        return {
            'available': False,
            'n_range': [],
            'total_checkpoints': 0
        }
    
    n_values = sorted(rl_data.keys())
    
    return {
        'available': True,
        'n_range': n_values,
        'total_checkpoints': len(rl_data),
        'size_data': {n: rl_data[n]['size'] for n in n_values},
        'depth_data': {n: rl_data[n]['depth'] for n in n_values},
        'pruned_size_data': {n: rl_data[n]['pruned_size'] for n in n_values},
        'pruned_depth_data': {n: rl_data[n]['pruned_depth'] for n in n_values}
    }

# Enhanced extraction function with method separation
def extract_rl_checkpoint_data_by_method():
    """Extract RL data separated by training method for validation"""
    extractor = RLCheckpointExtractor()
    return extractor.get_comprehensive_rl_data()

if __name__ == "__main__":
    # Test the enhanced extraction
    logging.basicConfig(level=logging.INFO)
    
    print("Testing enhanced RL checkpoint extractor...")
    extractor = RLCheckpointExtractor()
    
    # Test Classic DQN extraction
    classic_data = extractor.extract_classic_raw_data()
    print(f"Classic DQN data for n_wires: {sorted(classic_data['size'].keys())}")
    
    # Test Double DQN extraction
    double_data = extractor.extract_double_raw_data()
    print(f"Double DQN data for n_wires: {sorted(double_data['size'].keys())}")
    
    # Test comprehensive data
    comprehensive = extractor.get_comprehensive_rl_data()
    print(f"Comprehensive extraction completed: {comprehensive['checkpoints_available']}") 