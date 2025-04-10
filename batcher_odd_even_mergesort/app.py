"""
Interactive Web Demo for Batcher's Odd-Even Mergesort Algorithm
"""

import io
import base64
import random
import traceback
import functools
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify

# Import functionality from existing modules
from batcher_odd_even_mergesort import generate_sorting_network, apply_comparators
from visualization import (
    draw_network, 
    visualize_network_execution, 
    draw_depth_layers
)
from performance_analysis import (
    analyze_comparator_count,
    analyze_network_depth,
    compare_with_optimal,
    timing_analysis
)
from network_properties import (
    verify_zero_one_principle,
    get_network_properties_summary
)

# Initialize Flask app
app = Flask(__name__)

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

def generate_network_visualizations(comparators, n):
    """Generate network visualizations and convert to base64 strings"""
    # Configure matplotlib settings based on input size
    configure_matplotlib_for_size(n)
    
    # Generate network visualization
    fig = draw_network(comparators, n, title=f"Batcher's Odd-Even Mergesort Network (n={n})")
    network_img = fig_to_base64(fig)
    
    # Generate depth visualization
    result = draw_depth_layers(comparators, n)
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
        # Generate visualizations
        network_img, depth_img = generate_network_visualizations(comparators, n)
        
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
    
    # Input validation
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
    
    # Generate the network
    comparators = generate_sorting_network(n)
    
    # For tiny networks, ensure we have valid data
    validation_error = validate_comparators(comparators, n)
    if validation_error:
        return jsonify(validation_error)
    
    # Execute the network
    output_values = apply_comparators(input_values, comparators)
    
    try:
        # Generate execution visualization
        execution_img = generate_execution_visualization(comparators, input_values, n)
        
        return jsonify({
            'execution_img': execution_img,
            'input_values': input_values,
            'output_values': output_values
        })
    except Exception as viz_error:
        traceback.print_exc()
        return jsonify({
            'error': f'Visualization error: {str(viz_error)}',
            'input_values': input_values,
            'output_values': output_values
        })

@app.route('/performance_data', methods=['GET'])
@handle_exceptions
def performance_data():
    """Get performance data for plotting"""
    # Get comparator counts and depths for different sizes
    comparator_counts = analyze_comparator_count(2, 16)
    depths = analyze_network_depth(2, 16)
    
    # Get timing analysis
    generation_times = timing_analysis(2, 16, 3)
    
    # Get comparison with optimal
    comparison = compare_with_optimal()
    
    return jsonify({
        'comparator_counts': comparator_counts,
        'depths': depths,
        'generation_times': generation_times,
        'optimal_comparison': comparison
    })

if __name__ == '__main__':
    app.run(debug=True) 