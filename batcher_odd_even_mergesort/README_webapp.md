# Batcher's Odd-Even Mergesort Interactive Web Demo

This web application provides an interactive demonstration of Batcher's Odd-Even Mergesort algorithm for sorting networks.

## Features

- **Network Visualization**: Generate and visualize sorting networks for different input sizes
- **Execution Demo**: See how the sorting network processes specific inputs
- **Performance Analysis**: Analyze the algorithm's performance metrics (comparator count, depth)
- **Algorithm Theory**: Learn about the theoretical aspects of the algorithm

## Installation

1. Make sure you have Python 3.6+ installed
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the Application

Run the application with:

```
python app.py
```

Then open your web browser and navigate to `http://127.0.0.1:5000/`.

## Usage Guide

### Network Visualization
- Select the number of inputs (2-32)
- Click "Generate Network" to visualize the sorting network
- View network properties and depth visualization
- **Note**: While the app supports inputs from 2-32, for optimal visualization it's recommended to use values between 4-16

### Execution Demo
- Set the number of inputs
- Choose between random or custom input values
- Click "Execute Network" to see how the network sorts the given input
- For smaller input sizes (n < 4), some metrics like efficiency might show unusual values due to the algorithm's properties

### Performance Analysis
- View charts showing comparator count and depth for different input sizes
- Compare Batcher's algorithm performance with optimal sorting networks

## Technology Stack

- Backend: Flask (Python)
- Frontend: HTML, CSS, JavaScript
- Visualization: Matplotlib, Chart.js
- Styling: Bootstrap 5

## Project Structure

- `app.py` - Main Flask application
- `templates/index.html` - Frontend interface
- `batcher_odd_even_mergesort.py` - Core algorithm implementation
- `visualization.py` - Functions for visualizing networks
- `performance_analysis.py` - Performance measurement tools
- `network_properties.py` - Network analysis utilities 

## Troubleshooting

If you encounter any issues:

1. **Input size too small**: For very small input sizes (n=1 or n=2), some metrics may show unusual values. This is expected behavior as some calculations don't make sense for trivial cases.

2. **Input size too large**: For very large input sizes (n>20), visualizations may be less readable and calculations might take longer to process. 