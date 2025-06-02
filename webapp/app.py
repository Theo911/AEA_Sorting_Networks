"""
Interactive Web Demo for Batcher's Odd-Even Mergesort Algorithm
"""

import io
import base64
import random
import traceback
import functools
import datetime
import time
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
import sys
import os

# Add path for batcher_odd_even_mergesort
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
# Add path for RLSortingNetworks
rl_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'RLSortingNetworks'))
sys.path.insert(0, rl_dir)

# Import data manager for Phase 2 API endpoints
from data_manager import get_data_manager

# Import functionality from existing modules
from batcher_odd_even_mergesort.core import generate_sorting_network, apply_comparators
from batcher_odd_even_mergesort.visualization import (
    draw_network, 
    visualize_network_execution, 
    draw_depth_layers
)
from batcher_odd_even_mergesort.performance_analysis import (
    analyze_comparator_count,
    analyze_network_depth,
    compare_with_optimal,
    timing_analysis,
    count_depth
)
from batcher_odd_even_mergesort.network_properties import (
    verify_zero_one_principle,
    get_network_properties_summary
)

# Import future algorithm placeholders
# Import improved Batcher (should be available)
try:
    from batcher_odd_even_mergesort.improved_batcher import generate_improved_batcher_network
    IMPROVED_BATCHER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import improved Batcher: {e}. Improved Batcher features will be disabled.")
    generate_improved_batcher_network = lambda n: None
    IMPROVED_BATCHER_AVAILABLE = False

# Import improved RL (placeholder - not implemented yet)
# The algorithms directory was removed, so this is just a placeholder
generate_improved_rl_network = lambda n: None
IMPROVED_RL_AVAILABLE = False

# Overall improved algorithms availability
IMPROVED_ALGORITHMS_AVAILABLE = IMPROVED_BATCHER_AVAILABLE or IMPROVED_RL_AVAILABLE

# Import functionality from RL
try:
    from sorting_network_rl.utils.network_generator import get_rl_network
    # Also import any necessary RL evaluation utils if needed directly (e.g., pruning)
    # from sorting_network_rl.utils.evaluation import prune_redundant_comparators
    RL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import RLSortingNetworks modules: {e}. RL features will be disabled.")
    get_rl_network = None # Define as None to avoid NameError later
    RL_AVAILABLE = False

# Import RL evaluation utilities
try:
    from utils.rl_evaluation import get_available_rl_sizes, get_available_agent_types, execute_rl_network
    RL_EVALUATION_AVAILABLE = True
    # Get available RL sizes on startup
    AVAILABLE_RL_SIZES = get_available_rl_sizes()
    print(f"Available RL network sizes: {AVAILABLE_RL_SIZES}")
    # Print available agent types for each size
    for n in AVAILABLE_RL_SIZES:
        agents = get_available_agent_types(n)
        print(f"  n={n}: {agents}")
except ImportError as e:
    print(f"Warning: Could not import RL evaluation utilities: {e}")
    get_available_rl_sizes = lambda: []
    get_available_agent_types = lambda n: []
    execute_rl_network = None
    RL_EVALUATION_AVAILABLE = False
    AVAILABLE_RL_SIZES = []

# Initialize Flask app
app = Flask(__name__, static_folder='static', static_url_path='/static')

# Initialize data manager instance
data_manager = get_data_manager()

def handle_exceptions(f):
    """Decorator to handle exceptions in route handlers"""
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': str(e)})
    return wrapper

def validate_input_size(n):
    """Validate input size for sorting network"""
    if n < 2:
        return {'error': 'Input size must be at least 2'}
    if n > 32:
        return {'error': 'Input size must be at most 32 (larger sizes may cause performance issues)'}
    return None

def validate_comparators(comparators, n):
    """Validate generated comparators"""
    if len(comparators) == 0:
        return {'error': f'No comparators generated for size {n}. Try a larger input size.'}
    return None

def configure_matplotlib_for_size(n):
    """Configure matplotlib settings based on input size"""
    # Set up figure height based on input size
    if n <= 4:
        fig_height = 3
    elif n <= 8:
        fig_height = 4
    elif n <= 16:
        fig_height = 6
    else:
        fig_height = 8
        
    # Override matplotlib defaults for visualization
    plt.rcParams.update({
        'figure.figsize': (10, fig_height),
        'figure.dpi': 100,
        'figure.autolayout': True,
        'figure.constrained_layout.use': True,
        'figure.constrained_layout.h_pad': 0.1,
        'figure.constrained_layout.w_pad': 0.1
    })

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string for embedding in HTML"""
    buf = io.BytesIO()
    
    # Check if fig is the plt module (which happens when some visualization functions return plt)
    if fig == plt:
        # Adjust figure size to fit better in the card
        fig.gcf().set_size_inches(10, 4)  # Width, height in inches
        fig.tight_layout(pad=0.5)
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        fig.clf()  # Clear the current figure
        return img_str
    else:
        # Handle regular figure objects
        # Adjust figure size to fit better in the card
        fig.set_size_inches(10, 4)  # Width, height in inches
        fig.tight_layout(pad=0.5)
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)  # Close the figure to free memory
        return img_str

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

def generate_network_visualizations(comparators, n, algorithm_name="Sorting Network"):
    """Generate network visualizations and convert to base64 strings"""
    # Configure matplotlib settings based on input size
    configure_matplotlib_for_size(n)
    
    # Generate network visualization with dynamic title
    fig = draw_network(comparators, n, title=f"{algorithm_name} (n={n})")
    network_img = fig_to_base64(fig)
    
    # Generate depth visualization with dynamic title
    result = draw_depth_layers(comparators, n, title=f"{algorithm_name} by Depth")
    # Check if result is a tuple (fig, layers) or just plt module
    if isinstance(result, tuple):
        fig_depth, _ = result
    else:
        fig_depth = result
    depth_img = fig_to_base64(fig_depth)
    
    # Reset matplotlib defaults
    plt.rcdefaults()
    
    return network_img, depth_img

@app.route('/generate_network', methods=['POST'])
@handle_exceptions
def generate_network():
    """Generate a sorting network based on input size"""
    n = int(request.form.get('input_size', 8))
    
    # Input validation
    validation_error = validate_input_size(n)
    if validation_error:
        return jsonify(validation_error)
    
    # Generate the network
    comparators = generate_sorting_network(n)
    
    # For tiny networks, ensure we have valid data
    validation_error = validate_comparators(comparators, n)
    if validation_error:
        return jsonify(validation_error)
    
    # Get all network properties
    properties = get_network_properties_summary(n)
    
    try:
        # Generate visualizations with specific title
        network_img, depth_img = generate_network_visualizations(comparators, n, algorithm_name="Batcher's Odd-Even Mergesort Network")
        
        # Handle zero-one principle verification message
        if n <= 6:
            # For small networks, directly verify (computationally feasible)
            # We've fixed the special cases in batcher_odd_even_mergesort.py
            zero_one_status = verify_zero_one_principle(n, comparators)
        else:
            # For larger networks, use the mathematical proof 
            # (Batcher's odd-even mergesort is proven to satisfy the principle)
            zero_one_status = "proven"
        
        # Prepare response with all properties
        response = {
            'network_img': network_img,
            'depth_img': depth_img,
            'num_comparators': properties['num_comparators'],
            'depth': properties['depth'],
            'redundancy': properties['redundancy'],
            'efficiency': properties['efficiency'],
            'num_layers': properties['depth'],
            'zero_one_principle': zero_one_status,
            # Wire usage data
            'min_wire_usage': properties['wire_usage']['min_usage'],
            'max_wire_usage': properties['wire_usage']['max_usage'],
            'avg_wire_usage': properties['wire_usage']['avg_usage'],
            # Layer properties
            'min_comparators_per_layer': properties['layer_properties']['min_comparators_per_layer'],
            'max_comparators_per_layer': properties['layer_properties']['max_comparators_per_layer'],
            'avg_comparators_per_layer': properties['layer_properties']['avg_comparators_per_layer']
        }
        
        return jsonify(response)
    except Exception as viz_error:
        traceback.print_exc()
        return jsonify({'error': f'Visualization error: {str(viz_error)}'})

def generate_execution_visualization(comparators, input_values, n):
    """Generate execution visualization and convert to base64 string"""
    # Configure matplotlib settings based on input size
    configure_matplotlib_for_size(n)
    
    # Generate visualization of execution
    fig = visualize_network_execution(comparators, input_values)
    execution_img = fig_to_base64(fig)
    
    # Reset matplotlib defaults
    plt.rcdefaults()
    
    return execution_img

@app.route('/execute_network', methods=['POST'])
@handle_exceptions
def execute_network():
    """Execute the sorting network on a specific input"""
    n = int(request.form.get('input_size', 8))
    
    # Get the algorithm type
    algorithm = request.form.get('algorithm', 'batcher')
    
    # Enhanced input validation for RL algorithms
    if algorithm == 'rl':
        if not RL_EVALUATION_AVAILABLE:
            return jsonify({'error': 'RL evaluation not available. Please check installation.'})
        
        if n not in AVAILABLE_RL_SIZES:
            return jsonify({'error': f'RL algorithms only available for sizes: {AVAILABLE_RL_SIZES}'})
    else:
        # Limit Batcher algorithms to n<=16 for better visualization
        if n > 16:
            return jsonify({'error': 'Batcher algorithms limited to n≤16 for optimal visualization. Use RL algorithms for larger sizes.'})
    
    # Standard input validation
    validation_error = validate_input_size(n)
    if validation_error:
        return jsonify(validation_error)
    
    # Get input sequence or generate random
    input_type = request.form.get('input_type', 'random')
    if input_type == 'custom':
        try:
            input_values = [int(x.strip()) for x in request.form.get('input_values', '').split(',')]
            if len(input_values) != n:
                return jsonify({'error': f'Input must contain exactly {n} values'})
        except ValueError:
            return jsonify({'error': 'Invalid input values. Must be comma-separated integers.'})
    else:
        # Generate random input
        input_values = list(range(1, n+1))
        random.shuffle(input_values)
    
    # Start timing for execution tracking
    start_time = time.time()
    
    # Handle RL algorithms with evaluate.py integration
    if algorithm in ['rl', 'rl_double_dqn', 'rl_classic_dqn']:
        # Determine agent type from algorithm value
        if algorithm == 'rl_classic_dqn':
            agent_type = 'classic_dqn'
            algorithm_display_name = "RL Algorithm (Classic DQN)"
        else:  # rl or rl_double_dqn
            agent_type = 'double_dqn'
            algorithm_display_name = "RL Algorithm (Double DQN)"
        
        try:
            # Use the enhanced RL execution with evaluate.py
            result = execute_rl_network(n, agent_type, input_values)
            
            # Calculate execution time
            total_execution_time = (time.time() - start_time) * 1000
            result['execution_time_ms'] = total_execution_time
            
            # Enhanced algorithm name
            result['algorithm'] = algorithm_display_name
            
            # Store execution result for analysis
            try:
                execution_data = {
                    'algorithm': 'rl',  # Normalize for storage
                    'agent_type': agent_type,
                    'n_wires': n,
                    'input_values': input_values,
                    'output_values': result['output_values'],
                    'execution_time_ms': total_execution_time,
                    'comparators_count': result['comparators_count'],
                    'network_depth': result['network_depth'],
                    'success': result['success'],
                    'input_type': input_type,
                    'rl_analysis': result.get('rl_analysis', {})
                }
                data_manager.store_execution_result(execution_data)
            except Exception as e:
                print(f"Warning: Could not store RL execution result: {e}")
            
            return jsonify(result)
            
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': f'RL execution failed: {str(e)}'})
    
    # Handle Batcher algorithms (existing logic)
    if algorithm == 'batcher_improved':
        if not IMPROVED_BATCHER_AVAILABLE:
            return jsonify({'error': f'Enhanced Batcher algorithm not available. Import flag: {IMPROVED_BATCHER_AVAILABLE}'})
        
        try:
            # Enhanced Batcher algorithm
            comparators = generate_improved_batcher_network(n)
            if comparators is None:
                return jsonify({'error': f'Enhanced Batcher algorithm returned None for n={n}. This might indicate an implementation issue.'})
            if len(comparators) == 0:
                return jsonify({'error': f'Enhanced Batcher algorithm returned empty network for n={n}.'})
            print(f"Enhanced Batcher generated {len(comparators)} comparators for n={n}")
        except Exception as e:
            return jsonify({'error': f'Enhanced Batcher algorithm failed: {str(e)}'})
    else:
        # Default to Traditional Batcher's algorithm
        comparators = generate_sorting_network(n)
    
    # Calculate execution timing
    network_generation_time = time.time() - start_time
    
    # For tiny networks, ensure we have valid data
    validation_error = validate_comparators(comparators, n)
    if validation_error:
        return jsonify(validation_error)
    
    # Execute the network
    execution_start_time = time.time()
    output_values = apply_comparators(input_values, comparators)
    execution_time = time.time() - execution_start_time
    
    # Calculate total execution time
    total_execution_time = (network_generation_time + execution_time) * 1000  # Convert to milliseconds
    
    # Calculate network properties for tracking
    network_depth = count_depth(comparators, n)
    is_sorted = output_values == sorted(input_values)
    
    # Store execution result for analysis
    try:
        execution_data = {
            'algorithm': algorithm,
            'n_wires': n,
            'input_values': input_values,
            'output_values': output_values,
            'execution_time_ms': total_execution_time,
            'network_generation_time_ms': network_generation_time * 1000,
            'sort_execution_time_ms': execution_time * 1000,
            'comparators_count': len(comparators),
            'network_depth': network_depth,
            'success': is_sorted,
            'input_type': input_type
        }
        data_manager.store_execution_result(execution_data)
    except Exception as e:
        print(f"Warning: Could not store execution result: {e}")
    
    try:
        # Generate execution visualization
        execution_img = generate_execution_visualization(comparators, input_values, n)
        
        # Set the appropriate algorithm name for response
        algorithm_names = {
            'batcher': "Batcher's Odd-Even Mergesort",
            'batcher_improved': "Enhanced Batcher Algorithm"
        }
        algorithm_name = algorithm_names.get(algorithm, "Unknown Algorithm")
        
        return jsonify({
            'execution_img': execution_img,
            'input_values': input_values,
            'output_values': output_values,
            'algorithm': algorithm_name,
            'execution_time_ms': total_execution_time,
            'comparators_count': len(comparators),
            'network_depth': network_depth,
            'success': is_sorted
        })
    except Exception as viz_error:
        traceback.print_exc()
        return jsonify({
            'error': f'Visualization error: {str(viz_error)}',
            'input_values': input_values,
            'output_values': output_values,
            'algorithm': algorithm_name,
            'execution_time_ms': total_execution_time,
            'success': is_sorted
        })

@app.route('/performance_data', methods=['GET'])
@handle_exceptions
def performance_data():
    """Get performance data for plotting"""
    # Get requested algorithm from query parameters
    algorithm = request.args.get('algorithm', 'all')  # default to all algorithms
    
    # Get comparator counts and depths for different sizes for Batcher
    batcher_comparator_counts = analyze_comparator_count(2, 16)
    batcher_depths = analyze_network_depth(2, 16)
    
    # Get timing analysis for Batcher
    batcher_generation_times = timing_analysis(2, 16, 3)
    
    # Get comparison with optimal
    comparison = compare_with_optimal()
    
    # Initialize RL data containers
    rl_comparator_counts = {}
    rl_depths = {}
    rl_generation_times = {}
    
    # If RL is available and requested, get RL data
    if RL_AVAILABLE and get_rl_network is not None and algorithm in ['rl', 'all']:
        try:
            # This is just a placeholder - actual implementation would depend on
            # how RL performance data is collected/stored
            # For now, we'll return empty data which can be filled in later
            
            # Sample RL data for demonstration
            # In a real implementation, this would come from actual RL algorithm measurements
            rl_comparator_counts = {n: batcher_comparator_counts[n] for n in batcher_comparator_counts}
            rl_depths = {n: batcher_depths[n] for n in batcher_depths}
            rl_generation_times = {n: batcher_generation_times[n]*1.2 for n in batcher_generation_times}  # slightly slower for demo
        except Exception as e:
            print(f"Warning: Could not generate RL performance data: {e}")
    
    response = {
        'batcher_comparator_counts': batcher_comparator_counts,
        'batcher_depths': batcher_depths,
        'batcher_generation_times': batcher_generation_times,
        'rl_comparator_counts': rl_comparator_counts,
        'rl_depths': rl_depths,
        'rl_generation_times': rl_generation_times,
        'optimal_comparison': comparison
    }
    
    # If specific algorithm requested, filter to just that algorithm's data
    if algorithm == 'batcher':
        response = {
            'comparator_counts': batcher_comparator_counts,
            'depths': batcher_depths,
            'generation_times': batcher_generation_times,
            'optimal_comparison': comparison
        }
    elif algorithm == 'rl' and RL_AVAILABLE and get_rl_network is not None:
        response = {
            'comparator_counts': rl_comparator_counts,
            'depths': rl_depths,
            'generation_times': rl_generation_times,
            'optimal_comparison': comparison
        }
    else:
        # For backward compatibility, include the original keys too
        response.update({
            'comparator_counts': batcher_comparator_counts,
            'depths': batcher_depths,
            'generation_times': batcher_generation_times,
        })
    
    return jsonify(response)

# --- Route for RL Network Generation ---
@app.route('/generate_rl_network', methods=['POST'])
@handle_exceptions
def generate_rl_network():
    """Generate a sorting network using the RL model."""
    if not RL_AVAILABLE or get_rl_network is None:
        return jsonify({'error': 'RL Sorting Network module is not available. Check server logs.'})

    n = int(request.form.get('input_size', 8))
    
    # Input validation (reuse existing validator)
    # Consider if RL has different size limits (e.g., 2-8 based on available models)
    # For now, using the same limits as Batcher
    validation_error = validate_input_size(n)
    if validation_error:
        return jsonify(validation_error)

    # Determine RL project root path (adjust if needed)
    rl_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'RLSortingNetworks'))
    if not os.path.isdir(rl_project_root):
         return jsonify({'error': f'RL Project directory not found at expected location: {rl_project_root}'})

    # Generate the network using the RL utility function
    comparators = get_rl_network(n, rl_project_root)
    
    if comparators is None:
        # Error message already logged by get_rl_network
        return jsonify({'error': f'Failed to generate RL network for n={n}. Check server logs for details (config/model missing?).'})

    # Validate the generated comparators (reuse existing validator)
    validation_error = validate_comparators(comparators, n)
    if validation_error:
        # This might indicate an issue with the RL generation if it returns an empty list
        return jsonify(validation_error)
    
    # Reuse existing visualization function!
    try:
        # Generate visualizations with specific title
        network_img, depth_img = generate_network_visualizations(comparators, n, algorithm_name="RL Sorting Network")
        
        # Prepare response (we don't have all properties like Batcher's, calculate what we can)
        num_comparators_rl = len(comparators)
        depth_rl = count_depth(comparators, n)
        # Other properties like efficiency, redundancy, zero-one need specific calculation/verification for RL

        response = {
            'network_img': network_img,
            'depth_img': depth_img,
            'num_comparators': num_comparators_rl,
            'depth': depth_rl,
            # Add placeholders or calculate other properties as needed
            'redundancy': 'N/A',
            'efficiency': 'N/A',
            'num_layers': depth_rl, # Assuming depth corresponds to layers here
            'zero_one_principle': 'Unknown' # Needs verification (is_sorting_network)
        }
        
        return jsonify(response)
    except Exception as viz_error:
        traceback.print_exc()
        return jsonify({'error': f'Visualization or property calculation error for RL network: {str(viz_error)}'})

# ========== Phase 2: New API Endpoints ==========

@app.route('/api/performance_data', methods=['GET'])
@handle_exceptions
def api_enhanced_performance_data():
    """Return 7-algorithm comprehensive performance comparison data with validation metrics"""
    try:
        data_manager = get_data_manager()
        performance_data = data_manager.get_comprehensive_performance_data()
        
        return jsonify({
            'success': True,
            'data': performance_data,
            'timestamp': datetime.datetime.now().isoformat(),
            'api_version': '2.0_enhanced'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.datetime.now().isoformat()
        })

@app.route('/api/algorithm_comparison', methods=['GET'])
@handle_exceptions
def api_algorithm_evolution_analysis():
    """Return algorithm evolution analysis (Classic→Double, Raw→Pruned)"""
    try:
        data_manager = get_data_manager()
        evolution_data = data_manager.get_algorithm_evolution_analysis()
        
        return jsonify({
            'success': True,
            'evolution_analysis': evolution_data,
            'timestamp': datetime.datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.datetime.now().isoformat()
        })

@app.route('/api/data_validation', methods=['GET'])
@handle_exceptions
def api_data_quality_metrics():
    """Return data validation and consistency metrics"""
    try:
        data_manager = get_data_manager()
        performance_data = data_manager.get_comprehensive_performance_data()
        validation_data = performance_data.get('data_quality', {})
        
        return jsonify({
            'success': True,
            'data_quality': validation_data,
            'data_sources': performance_data.get('data_sources', {}),
            'algorithm_status': performance_data.get('algorithm_status', {}),
            'timestamp': datetime.datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.datetime.now().isoformat()
        })

@app.route('/api/execution_analytics', methods=['GET'])
@handle_exceptions
def api_execution_analytics():
    """Return execution analytics for dashboard"""
    try:
        data_manager = get_data_manager()
        analytics = data_manager.get_execution_analytics()
        
        return jsonify({
            'success': True,
            'analytics': analytics,
            'timestamp': datetime.datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.datetime.now().isoformat()
        })

@app.route('/api/store_execution', methods=['POST'])
@handle_exceptions
def api_store_enhanced_execution():
    """Store execution results with 7-algorithm classification and enhanced tracking"""
    try:
        execution_data = request.get_json()
        
        if not execution_data:
            return jsonify({
                'success': False,
                'error': 'No execution data provided'
            })
        
        # Enhanced execution data structure
        enhanced_execution = {
            'timestamp': datetime.datetime.now().isoformat(),
            'algorithm': execution_data.get('algorithm', 'unknown'),
            'algorithm_variant': {
                'base_type': execution_data.get('base_type', 'unknown'),  # batcher, rl, optimal
                'training_method': execution_data.get('training_method'),  # classic_dqn, double_dqn
                'optimization': execution_data.get('optimization', 'unknown')  # raw, pruned
            },
            'n_wires': execution_data.get('n_wires'),
            'input_values': execution_data.get('input_values'),
            'execution_time_ms': execution_data.get('execution_time_ms'),
            'comparators_count': execution_data.get('comparators_count'),
            'network_depth': execution_data.get('network_depth'),
            'success': execution_data.get('success', False),
            'performance_metrics': execution_data.get('performance_metrics', {}),
            'user_session': execution_data.get('user_session', 'anonymous')
        }
        
        # Store the enhanced execution result
        data_manager = get_data_manager()
        data_manager.store_execution_result(enhanced_execution)
        
        return jsonify({
            'success': True,
            'message': 'Enhanced execution result stored successfully',
            'execution_id': f"exec_{enhanced_execution['timestamp'].replace(':', '').replace('-', '').replace('.', '')}",
            'timestamp': enhanced_execution['timestamp']
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.datetime.now().isoformat()
        })

@app.route('/api/algorithm_status', methods=['GET'])
@handle_exceptions
def api_algorithm_status():
    """Return availability status of all algorithms with data sources"""
    try:
        # Test the improved Batcher function
        try:
            test_result = generate_improved_batcher_network(4)
            batcher_improved_test = test_result is not None and len(test_result) > 0
        except Exception as e:
            batcher_improved_test = False
            print(f"Improved Batcher test failed: {e}")
        
        status = {
            'batcher': {
                'available': True,
                'description': "Batcher's Traditional Algorithm",
                'status': 'Classic reliable algorithm',
                'size_range': '2-16'
            },
            'batcher_improved': {
                'available': IMPROVED_BATCHER_AVAILABLE and batcher_improved_test,
                'description': "Enhanced Batcher Algorithm", 
                'status': 'Optimized with modern improvements',
                'test_result': batcher_improved_test,
                'import_flag': IMPROVED_BATCHER_AVAILABLE,
                'size_range': '2-16'
            },
            'rl': {
                'available': RL_EVALUATION_AVAILABLE,
                'description': "RL Algorithm (Double DQN)",
                'status': 'ML-trained networks using advanced Double DQN' if RL_EVALUATION_AVAILABLE else 'Evaluation not available',
                'size_range': ', '.join(map(str, AVAILABLE_RL_SIZES)) if AVAILABLE_RL_SIZES else 'None'
            }
        }
        
        return jsonify({
            'success': True,
            'algorithms': status,
            'available_rl_sizes': AVAILABLE_RL_SIZES,
            'rl_evaluation_available': RL_EVALUATION_AVAILABLE,
            'timestamp': datetime.datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.datetime.now().isoformat()
        })

@app.route('/api/available_sizes', methods=['GET'])
@handle_exceptions  
def api_available_sizes():
    """Return available input sizes for each algorithm type"""
    try:
        return jsonify({
            'success': True,
            'sizes': {
                'batcher': list(range(2, 17)),  # 2-16
                'batcher_improved': list(range(2, 17)),  # 2-16
                'rl': AVAILABLE_RL_SIZES
            },
            'rl_evaluation_available': RL_EVALUATION_AVAILABLE,
            'available_rl_sizes': AVAILABLE_RL_SIZES
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to get available sizes: {str(e)}'
        })



if __name__ == '__main__':
    app.run(debug=True) 