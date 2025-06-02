import os
import sys
import subprocess
import json
import re
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import hashlib

# Add paths for RL modules
script_dir = os.path.dirname(os.path.abspath(__file__))
webapp_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(webapp_dir)
rl_dir = os.path.join(project_root, "RLSortingNetworks")

# Global cache for evaluate.py results
RL_EVALUATION_CACHE = {}
CACHE_EXPIRY_SECONDS = 3600  # 1 hour cache

def get_available_rl_sizes() -> List[int]:
    """Get available RL network sizes from checkpoints directory"""
    checkpoints_dir = os.path.join(rl_dir, "checkpoints")
    available_sizes = {}
    
    try:
        if not os.path.exists(checkpoints_dir):
            return []
            
        for folder in os.listdir(checkpoints_dir):
            if folder.endswith('s') and not folder.endswith('_classic'):  # Folders like "4w_10s", "6w_25s" (Double DQN)
                try:
                    # Extract n from folder name like "4w_10s" -> 4
                    n = int(folder.split('w_')[0])
                    
                    # Check if best_network.csv exists for double DQN
                    best_network_path = os.path.join(checkpoints_dir, folder, "best_network.csv")
                    if os.path.exists(best_network_path):
                        if n not in available_sizes:
                            available_sizes[n] = set()
                        available_sizes[n].add('double_dqn')
                        
                        # Check if classic version also exists
                        classic_folder = folder + '_classic'
                        classic_path = os.path.join(checkpoints_dir, classic_folder, "best_network.csv")
                        if os.path.exists(classic_path):
                            available_sizes[n].add('classic_dqn')
                            
                except ValueError:
                    continue
        
        # Return sizes where at least one agent type is available
        return sorted(list(available_sizes.keys()))
        
    except Exception as e:
        print(f"Error getting available RL sizes: {e}")
        return []

def get_available_agent_types(n: int) -> List[str]:
    """Get available agent types for a specific network size"""
    checkpoints_dir = os.path.join(rl_dir, "checkpoints")
    available_agents = set()
    
    try:
        if not os.path.exists(checkpoints_dir):
            return []
            
        for folder in os.listdir(checkpoints_dir):
            if folder.startswith(f'{n}w_'):
                # Check for classic DQN
                if folder.endswith('_classic'):
                    best_network_path = os.path.join(checkpoints_dir, folder, "best_network.csv")
                    if os.path.exists(best_network_path):
                        available_agents.add('classic_dqn')
                        
                # Check for double DQN (folders ending with 's' but not '_classic')
                elif folder.endswith('s') and not folder.endswith('_classic'):
                    best_network_path = os.path.join(checkpoints_dir, folder, "best_network.csv")
                    if os.path.exists(best_network_path):
                        available_agents.add('double_dqn')
                        
        return sorted(list(available_agents))
        
    except Exception as e:
        print(f"Error getting available agent types for n={n}: {e}")
        return []

def generate_cache_key(n: int, agent_type: str, input_values: Optional[List[int]] = None) -> str:
    """Generate a cache key for evaluation results"""
    key_data = f"{n}_{agent_type}"
    if input_values:
        key_data += f"_{hash(tuple(input_values))}"
    return hashlib.md5(key_data.encode()).hexdigest()

def is_cache_valid(cache_entry: Dict) -> bool:
    """Check if cache entry is still valid"""
    if 'timestamp' not in cache_entry:
        return False
    return time.time() - cache_entry['timestamp'] < CACHE_EXPIRY_SECONDS

def call_evaluate_py(n: int, agent_type: str, input_values: Optional[List[int]] = None) -> Dict[str, Any]:
    """Call evaluate.py and parse the output with caching"""
    
    # Generate cache key
    cache_key = generate_cache_key(n, agent_type, input_values)
    
    # Check cache first
    if cache_key in RL_EVALUATION_CACHE and is_cache_valid(RL_EVALUATION_CACHE[cache_key]):
        print(f"Using cached RL evaluation for n={n}, agent={agent_type}")
        return RL_EVALUATION_CACHE[cache_key]['data']
    
    try:
        # Prepare command - use absolute paths
        evaluate_script = os.path.abspath(os.path.join(rl_dir, "scripts", "evaluate.py"))
        cmd = [
            sys.executable, 
            evaluate_script,
            "-n", str(n),
            "-agent", agent_type
        ]
        
        # Set up environment
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{os.path.abspath(project_root)}:{env.get('PYTHONPATH', '')}"
        
        # Run evaluate.py
        print(f"Running RL evaluation: n={n}, agent={agent_type}")
        print(f"Script path: {evaluate_script}")
        print(f"Working directory: {os.path.abspath(project_root)}")
        result = subprocess.run(
            cmd,
            cwd=os.path.abspath(project_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout
        )
        
        if result.returncode != 0:
            error_msg = f"evaluate.py failed with return code {result.returncode}"
            if result.stderr:
                error_msg += f": {result.stderr}"
            raise RuntimeError(error_msg)
        
        # Parse the output - check both stdout and stderr since logging goes to stderr
        output_to_parse = result.stdout if result.stdout.strip() else result.stderr
        parsed_data = parse_evaluate_output(output_to_parse)
        
        # Cache the result
        RL_EVALUATION_CACHE[cache_key] = {
            'data': parsed_data,
            'timestamp': time.time()
        }
        
        return parsed_data
        
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"RL evaluation timed out for n={n}")
    except Exception as e:
        raise RuntimeError(f"RL evaluation failed: {str(e)}")

def parse_evaluate_output(output: str) -> Dict[str, Any]:
    """Parse the output from evaluate.py"""
    lines = output.split('\n')
    
    result = {
        'original_comparators': [],
        'pruned_comparators': [],
        'original_length': 0,
        'pruned_length': 0,
        'original_depth': 0,
        'pruned_depth': 0,
        'network_valid': False,
        'pruning_success': False,
        'original_visualization': '',
        'pruned_visualization': '',
        'source_description': 'Unknown'
    }
    
    current_section = None
    collecting_viz = False
    viz_lines = []
    
    for line in lines:
        line = line.strip()
        
        # Skip log formatting
        if '[INFO]' in line:
            line = line.split('[INFO]')[-1].strip()
        
        # Also handle lines that already have the log prefix stripped
        original_line = line
        
        # Parse network analysis section
        if '--- Network Analysis' in line:
            if 'Best Network (from CSV)' in line:
                result['source_description'] = 'Best Network (from CSV)'
            elif 'Agent Policy Output' in line:
                result['source_description'] = 'Agent Policy Output'
                
        elif 'Original Length:' in line:
            try:
                result['original_length'] = int(line.split(':')[-1].strip())
            except ValueError:
                pass
                
        elif 'Pruned Depth (Sequential Dependency):' in line and 'Unchanged' in line:
            result['pruned_depth'] = result['original_depth']
            result['pruning_success'] = False
            
        elif 'Pruned Depth (Sequential Dependency):' in line and 'Unchanged' not in line:
            try:
                result['pruned_depth'] = int(line.split(':')[-1].strip())
                result['pruning_success'] = True
            except ValueError:
                pass
                
        elif 'Original Depth (Sequential Dependency):' in line:
            try:
                result['original_depth'] = int(line.split(':')[-1].strip())
            except ValueError:
                pass
                
        elif 'Reduced length to:' in line:
            try:
                result['pruned_length'] = int(line.split(':')[-1].strip())
                result['pruning_success'] = True
            except ValueError:
                pass
                
        elif 'Network Status: VALID' in line:
            result['network_valid'] = True
            
        elif 'Network Status: INVALID' in line:
            result['network_valid'] = False
            
        # Parse network steps
        elif line.startswith('Network Steps:'):
            current_section = 'original_steps'
            
        elif line.startswith('Pruned Network Steps:'):
            current_section = 'pruned_steps'
            
        elif current_section and ('Step ' in line or line.strip().startswith('Step ')):
            try:
                # Parse "Step X: (a, b)" format or "  Step X: (a, b)" (with indentation)
                step_match = re.search(r'Step \d+: \((\d+), (\d+)\)', line)
                if step_match:
                    comparator = (int(step_match.group(1)), int(step_match.group(2)))
                    if current_section == 'original_steps':
                        result['original_comparators'].append(comparator)
                    elif current_section == 'pruned_steps':
                        result['pruned_comparators'].append(comparator)
            except (ValueError, AttributeError):
                pass
                
        # Parse visualizations
        elif 'Visualization (Original):' in line:
            collecting_viz = True
            current_section = 'original_viz'
            viz_lines = []
            
        elif 'Visualization (Pruned):' in line:
            collecting_viz = True
            current_section = 'pruned_viz'
            viz_lines = []
            
        elif collecting_viz:
            if line.startswith('w') and ':' in line:
                viz_lines.append(line)
            elif line == '' or 'Attempting to prune' in line or 'Pruning' in line:
                # End of visualization
                if current_section == 'original_viz':
                    result['original_visualization'] = '\n'.join(viz_lines)
                elif current_section == 'pruned_viz':
                    result['pruned_visualization'] = '\n'.join(viz_lines)
                collecting_viz = False
                current_section = None
                viz_lines = []
    
    # Fill in missing data
    if not result['pruned_comparators']:
        result['pruned_comparators'] = result['original_comparators']
        result['pruned_length'] = result['original_length']
        result['pruned_depth'] = result['original_depth']
        
    if not result['pruned_visualization']:
        result['pruned_visualization'] = result['original_visualization']
        
    return result

def create_clean_rl_visualization(comparators: List[Tuple[int, int]], input_values: List[int], n: int, agent_type: str) -> str:
    """Create clean RL visualization using Batcher's visualization approach"""
    
    # Import Batcher's visualization function using the same approach as app.py
    try:
        # Add path for batcher_odd_even_mergesort (same as app.py)
        # From utils/rl_evaluation.py -> webapp -> AEA_Sorting_Networks (where batcher module is)
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
            
        # Import using the same pattern as app.py
        from batcher_odd_even_mergesort.visualization import visualize_network_execution
        
        # Use Batcher's proven visualization approach
        fig = visualize_network_execution(comparators, input_values)
        
        # Convert to base64 using the same approach as app.py
        import io
        import base64
        buffer = io.BytesIO()
        
        # Check if fig is the plt module or a figure object (same logic as app.py's fig_to_base64)
        if fig == plt:
            # If it's the plt module
            plt.gcf().set_size_inches(10, 4)  # Same size as app.py
            plt.tight_layout(pad=0.5)
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.clf()  # Clear the current figure
        else:
            # Handle regular figure objects
            fig.set_size_inches(10, 4)  # Same size as app.py
            fig.tight_layout(pad=0.5)
            fig.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)  # Close the figure to free memory
        
        return image_base64
        
    except ImportError as e:
        print(f"Could not import Batcher visualization: {e}")
        # Fallback to the modern visualization if Batcher's isn't available
        return create_modern_rl_visualization(comparators, input_values, n, {'agent_type': agent_type}, show_pruned=False)

def create_modern_rl_visualization(
    comparators: List[Tuple[int, int]], 
    input_values: List[int], 
    n: int,
    analysis_data: Dict[str, Any],
    show_pruned: bool = True
) -> str:
    """Create a modern visualization similar to Batcher's execution demo"""
    
    # Use the pruned network if available and requested
    if show_pruned and analysis_data.get('pruned_comparators'):
        viz_comparators = analysis_data['pruned_comparators']
        title_suffix = " (Pruned)"
        depth = analysis_data.get('pruned_depth', analysis_data.get('original_depth', len(viz_comparators)))
    else:
        viz_comparators = comparators
        title_suffix = " (Original)"
        depth = analysis_data.get('original_depth', len(viz_comparators))
    
    # Calculate figure size based on network complexity
    fig_width = max(12, min(20, len(viz_comparators) * 0.8))
    fig_height = max(8, n * 1.2)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height))
    fig.suptitle(f'RL Sorting Network Execution{title_suffix} (n={n})', fontsize=16, fontweight='bold')
    
    # Colors for different wire states
    colors = plt.cm.Set3(np.linspace(0, 1, n))
    
    # Left plot: Network structure with execution trace
    ax1.set_title('Network Structure & Execution Trace', fontsize=14, fontweight='bold')
    
    # Draw wires
    for i in range(n):
        ax1.axhline(y=i, color='lightgray', linewidth=2, alpha=0.7)
        ax1.text(-0.5, i, f'w{i}', ha='right', va='center', fontweight='bold', fontsize=10)
    
    # Track values at each step
    current_values = input_values.copy()
    step_values = [current_values.copy()]
    
    # Draw comparators and trace execution
    for step, (i, j) in enumerate(viz_comparators):
        x_pos = step + 0.5
        
        # Draw comparator
        ax1.plot([x_pos, x_pos], [i, j], 'ko-', linewidth=3, markersize=8)
        
        # Add step number
        mid_y = (i + j) / 2
        ax1.text(x_pos, mid_y + 0.3, f'{step+1}', ha='center', va='bottom', 
                fontsize=8, fontweight='bold', 
                bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.8))
        
        # Perform comparison and swap if needed
        if current_values[i] > current_values[j]:
            current_values[i], current_values[j] = current_values[j], current_values[i]
        
        step_values.append(current_values.copy())
    
    # Show values at key positions
    positions_to_show = [0, len(viz_comparators)//3, 2*len(viz_comparators)//3, len(viz_comparators)]
    for pos in positions_to_show:
        if pos < len(step_values):
            x_pos = pos if pos == 0 else pos + 0.5
            for i, val in enumerate(step_values[pos]):
                color = colors[input_values.index(val)]
                ax1.text(x_pos, i - 0.3, str(val), ha='center', va='top', 
                        fontsize=10, fontweight='bold', color=color,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    ax1.set_xlim(-1, len(viz_comparators) + 1)
    ax1.set_ylim(-0.8, n - 0.2)
    ax1.set_xlabel('Execution Step', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Wire Index', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Before/After comparison with analysis
    ax2.set_title('Input/Output Analysis', fontsize=14, fontweight='bold')
    
    # Input section
    input_y = 0.8
    ax2.text(0.05, input_y, 'INPUT:', fontsize=12, fontweight='bold', 
             transform=ax2.transAxes)
    for i, val in enumerate(input_values):
        color = colors[input_values.index(val)]
        rect = FancyBboxPatch((0.15 + i*0.08, input_y-0.03), 0.06, 0.06, 
                             boxstyle="round,pad=0.01", 
                             facecolor=color, alpha=0.7,
                             transform=ax2.transAxes)
        ax2.add_patch(rect)
        ax2.text(0.18 + i*0.08, input_y, str(val), ha='center', va='center', 
                fontsize=10, fontweight='bold', transform=ax2.transAxes)
    
    # Output section
    output_y = 0.65
    output_values = step_values[-1]
    ax2.text(0.05, output_y, 'OUTPUT:', fontsize=12, fontweight='bold',
             transform=ax2.transAxes)
    for i, val in enumerate(output_values):
        original_color = colors[input_values.index(val)]
        rect = FancyBboxPatch((0.15 + i*0.08, output_y-0.03), 0.06, 0.06, 
                             boxstyle="round,pad=0.01", 
                             facecolor=original_color, alpha=0.7,
                             transform=ax2.transAxes)
        ax2.add_patch(rect)
        ax2.text(0.18 + i*0.08, output_y, str(val), ha='center', va='center', 
                fontsize=10, fontweight='bold', transform=ax2.transAxes)
    
    # Analysis section
    analysis_y = 0.4
    ax2.text(0.05, analysis_y, 'ANALYSIS:', fontsize=12, fontweight='bold',
             transform=ax2.transAxes)
    
    # Network metrics
    metrics = [
        f"Comparators: {len(viz_comparators)}",
        f"Depth: {depth}",
        f"Valid: {'✓' if analysis_data.get('network_valid', True) else '✗'}",
        f"Sorted: {'✓' if output_values == sorted(input_values) else '✗'}"
    ]
    
    for i, metric in enumerate(metrics):
        ax2.text(0.05, analysis_y - 0.08 - i*0.05, metric, fontsize=10,
                transform=ax2.transAxes,
                color='green' if '✓' in metric else 'red' if '✗' in metric else 'black')
    
    # Pruning info if available
    if show_pruned and analysis_data.get('pruning_success'):
        pruning_y = 0.1
        ax2.text(0.05, pruning_y, 'PRUNING:', fontsize=12, fontweight='bold',
                transform=ax2.transAxes)
        
        original_size = analysis_data.get('original_length', len(comparators))
        pruned_size = len(viz_comparators)
        reduction = ((original_size - pruned_size) / original_size * 100) if original_size > 0 else 0
        
        pruning_info = [
            f"Original: {original_size} comparators",
            f"Pruned: {pruned_size} comparators", 
            f"Reduction: {reduction:.1f}%"
        ]
        
        for i, info in enumerate(pruning_info):
            ax2.text(0.05, pruning_y - 0.05 - i*0.04, info, fontsize=9,
                    transform=ax2.transAxes, color='blue')
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    plt.tight_layout()
    
    # Convert to base64
    import io
    import base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return image_base64

def execute_rl_network(n: int, agent_type: str, input_values: List[int]) -> Dict[str, Any]:
    """Execute RL network and return comprehensive results with README data prioritized for display metrics"""
    
    try:
        # Get evaluation data from evaluate.py for actual network execution and visualization
        analysis_data = call_evaluate_py(n, agent_type, input_values)
        
        # Use the original comparators for execution
        comparators = analysis_data['original_comparators']
        
        if not comparators:
            raise RuntimeError("No comparators found in RL network")
        
        # Execute the network on input values
        output_values = input_values.copy()
        for i, j in comparators:
            if output_values[i] > output_values[j]:
                output_values[i], output_values[j] = output_values[j], output_values[i]
        
        # Use Batcher's clean visualization for consistency
        execution_img = create_clean_rl_visualization(comparators, input_values, n, agent_type)
        
        # --- NEW: Try to get README table data first for display metrics ---
        readme_data = None
        display_metrics_source = "evaluate.py"
        
        try:
            # Import and initialize README parser
            import sys
            import os
            
            # Add webapp directory to path for imports
            webapp_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if webapp_dir not in sys.path:
                sys.path.insert(0, webapp_dir)
            
            from utils.readme_parser import ReadmePerformanceParser
            
            # Get README data
            readme_parser = ReadmePerformanceParser()
            if readme_parser.is_data_available():
                # Map agent_type to README data structure
                if agent_type == "double_dqn":
                    readme_raw = readme_parser.extract_double_dqn_data()['raw']
                    readme_pruned = readme_parser.extract_double_dqn_data()['pruned']
                elif agent_type == "classic_dqn":
                    readme_raw = readme_parser.extract_classic_dqn_data()['raw']
                    readme_pruned = readme_parser.extract_classic_dqn_data()['pruned']
                else:
                    readme_raw = readme_parser.extract_double_dqn_data()['raw']  # Default to double
                    readme_pruned = readme_parser.extract_double_dqn_data()['pruned']
                
                # Check if README has data for this n value
                if n in readme_raw.get('size', {}) and readme_raw['size'][n] is not None:
                    readme_data = {
                        'original_size': readme_raw['size'][n],
                        'original_depth': readme_raw.get('depth', {}).get(n),
                        'pruned_size': readme_pruned.get('size', {}).get(n),
                        'pruned_depth': readme_pruned.get('depth', {}).get(n)
                    }
                    display_metrics_source = "README table"
                    print(f"Using README table data for n={n}, agent={agent_type}: {readme_data}")
                else:
                    print(f"No README data found for n={n}, agent={agent_type}, falling back to evaluate.py")
                    
        except Exception as e:
            print(f"Failed to load README data, using evaluate.py: {e}")
        
        # Use README data if available, otherwise fall back to evaluate.py data
        if readme_data:
            # Use README metrics for display
            original_length = readme_data['original_size']
            original_depth = readme_data['original_depth'] or analysis_data.get('original_depth', 0)
            pruned_length = readme_data['pruned_size'] or readme_data['original_size']
            pruned_depth = readme_data['pruned_depth'] or readme_data['original_depth'] or original_depth
            
            # Calculate pruning efficiency based on README data
            pruning_efficiency = ((original_length - pruned_length) / original_length * 100) if original_length > 0 else 0
            pruning_applied = pruned_length < original_length
            
            # Determine source description with README priority
            source_description = f"README Table ({display_metrics_source})"
        else:
            # Fall back to evaluate.py data
            original_length = analysis_data.get('original_length', len(comparators))
            original_depth = analysis_data.get('original_depth', 0)
            pruned_length = analysis_data.get('pruned_length', original_length)
            pruned_depth = analysis_data.get('pruned_depth', original_depth)
            
            # Calculate pruning efficiency from evaluate.py data
            pruning_efficiency = ((original_length - pruned_length) / original_length * 100) if original_length > 0 else 0
            pruning_applied = analysis_data.get('pruning_success', False)
            
            # Use evaluate.py source description
            source_description = analysis_data.get('source_description', 'Unknown')
        
        # Calculate vs optimal comparison using display metrics
        optimal_estimates = {2: 1, 3: 3, 4: 5, 5: 9, 6: 12, 7: 16, 8: 19, 9: 25, 10: 29}
        optimal_size = optimal_estimates.get(n, n * (n-1) // 2)  # fallback to bubble sort bound
        vs_optimal_diff = original_length - optimal_size
        vs_optimal_percent = ((original_length / optimal_size - 1) * 100) if optimal_size > 0 else 0
        
        return {
            'execution_img': execution_img,
            'input_values': input_values,
            'output_values': output_values,
            'success': output_values == sorted(input_values),
            'comparators_count': original_length,  # Use README table data (Double DQN Best Size)
            'network_depth': original_depth,  # Use display metrics
            'rl_analysis': {
                'agent_type': agent_type.replace('_', ' ').title(),
                'pruned_size': pruned_length,
                'pruned_depth': pruned_depth,
                'pruning_efficiency': f"{pruning_efficiency:.1f}%",
                'pruning_applied': pruning_applied,
                'network_status': 'Valid' if analysis_data.get('network_valid', False) else 'Invalid',
                'vs_optimal': f"+{vs_optimal_diff}" if vs_optimal_diff > 0 else str(vs_optimal_diff)
            }
        }
        
    except Exception as e:
        raise RuntimeError(f"RL network execution failed: {str(e)}") 