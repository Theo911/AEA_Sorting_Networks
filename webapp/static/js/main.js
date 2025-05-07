// Network visualization form submission
document.getElementById('networkForm').addEventListener('submit', function(e) {
    e.preventDefault();
    
    // Show loading spinner
    document.getElementById('loadingSpinner').style.display = 'inline-block';
    
    // Get form data
    const formData = new FormData(this);
    const selectedAlgorithm = formData.get('algorithm');
    
    // Determine endpoint based on algorithm
    const endpoint = selectedAlgorithm === 'rl' ? '/generate_rl_network' : '/generate_network';
    const algoName = selectedAlgorithm === 'rl' ? 'RL Sorting Network' : "Batcher's Odd-Even Mergesort";

    // Update titles
    document.getElementById('main-title').textContent = algoName;
    document.getElementById('network-viz-title').textContent = `${algoName} Visualization`;
    document.getElementById('depth-viz-title').textContent = `Depth Visualization (${algoName})`;
    document.getElementById('network-properties-title').textContent = `${algoName} Properties`;

    // Send request to server
    fetch(endpoint, {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // Hide loading spinner
        document.getElementById('loadingSpinner').style.display = 'none';
        
        if (data.error) {
            alert(data.error);
            return;
        }
        
        // Update network visualization
        document.getElementById('networkVisualization').innerHTML = 
            `<img src="data:image/png;base64,${data.network_img}" alt="Sorting Network">`;
        
        // Update depth visualization
        document.getElementById('depthVisualization').innerHTML = 
            `<img src="data:image/png;base64,${data.depth_img}" alt="Depth Visualization">`;
        
        // Update network properties
        document.getElementById('numComparators').textContent = data.num_comparators;
        document.getElementById('networkDepth').textContent = data.depth;
        document.getElementById('redundancy').textContent = 
            typeof data.redundancy === 'number' ? data.redundancy.toFixed(2) + '%' : data.redundancy;
        document.getElementById('efficiency').textContent = 
            typeof data.efficiency === 'number' ? data.efficiency.toFixed(2) + '%' : data.efficiency;
        document.getElementById('numLayers').textContent = data.num_layers;

        // Update Zero-One Principle verification status (handle string cases)
        if (data.zero_one_principle === "proven") {
            document.getElementById('zeroOnePrinciple').textContent = "Yes (mathematically proven)";
        } else if (data.zero_one_principle === true) {
            document.getElementById('zeroOnePrinciple').textContent = "Yes (verified)";
        } else if (data.zero_one_principle === false) {
            document.getElementById('zeroOnePrinciple').textContent = "No";
        } else { // Handle other string cases like 'Unknown'
            document.getElementById('zeroOnePrinciple').textContent = data.zero_one_principle;
        }

        document.getElementById('minWireUsage').textContent = data.min_wire_usage !== undefined ? data.min_wire_usage : 'N/A';
        document.getElementById('maxWireUsage').textContent = data.max_wire_usage !== undefined ? data.max_wire_usage : 'N/A';
        document.getElementById('avgWireUsage').textContent = 
            typeof data.avg_wire_usage === 'number' ? data.avg_wire_usage.toFixed(2) : (data.avg_wire_usage !== undefined ? data.avg_wire_usage : 'N/A');
        document.getElementById('minCompPerLayer').textContent = data.min_comparators_per_layer !== undefined ? data.min_comparators_per_layer : 'N/A';
        document.getElementById('maxCompPerLayer').textContent = data.max_comparators_per_layer !== undefined ? data.max_comparators_per_layer : 'N/A';
        document.getElementById('avgCompPerLayer').textContent = 
            typeof data.avg_comparators_per_layer === 'number' ? data.avg_comparators_per_layer.toFixed(2) : (data.avg_comparators_per_layer !== undefined ? data.avg_comparators_per_layer : 'N/A');
    })
    .catch(error => {
        document.getElementById('loadingSpinner').style.display = 'none';
        console.error('Error:', error);
        alert('An error occurred. Please try again.');
    });
});

// Execution form events
document.getElementById('execInputSize').addEventListener('change', function() {
    document.getElementById('requiredInputs').textContent = this.value;
});

document.querySelectorAll('input[name="input_type"]').forEach(radio => {
    radio.addEventListener('change', function() {
        const customInputDiv = document.getElementById('customInputDiv');
        if (this.value === 'custom') {
            customInputDiv.style.display = 'block';
        } else {
            customInputDiv.style.display = 'none';
        }
    });
});

// Algorithm Theory toggle
document.getElementById('batcherTheoryBtn').addEventListener('click', function() {
    // Show Batcher theory, hide RL theory
    document.getElementById('batcherTheory').style.display = 'block';
    document.getElementById('rlTheory').style.display = 'none';
    
    // Update button states
    this.classList.add('active');
    this.classList.remove('btn-outline-primary');
    this.classList.add('btn-primary');
    
    const rlBtn = document.getElementById('rlTheoryBtn');
    rlBtn.classList.remove('active');
    rlBtn.classList.remove('btn-primary');
    rlBtn.classList.add('btn-outline-primary');
});

document.getElementById('rlTheoryBtn').addEventListener('click', function() {
    // Show RL theory, hide Batcher theory
    document.getElementById('batcherTheory').style.display = 'none';
    document.getElementById('rlTheory').style.display = 'block';
    
    // Update button states
    this.classList.add('active');
    this.classList.remove('btn-outline-primary');
    this.classList.add('btn-primary');
    
    const batcherBtn = document.getElementById('batcherTheoryBtn');
    batcherBtn.classList.remove('active');
    batcherBtn.classList.remove('btn-primary');
    batcherBtn.classList.add('btn-outline-primary');
});

// Execution form submission
document.getElementById('executionForm').addEventListener('submit', function(e) {
    e.preventDefault();
    
    // Show loading spinner
    document.getElementById('executionLoadingSpinner').style.display = 'inline-block';
    
    // Get form data
    const formData = new FormData(this);
    
    // Send request to server
    fetch('/execute_network', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // Hide loading spinner
        document.getElementById('executionLoadingSpinner').style.display = 'none';
        
        if (data.error) {
            alert(data.error);
            return;
        }
        
        // Update execution visualization
        document.getElementById('executionVisualization').innerHTML = 
            `<img src="data:image/png;base64,${data.execution_img}" alt="Execution Visualization">`;
        
        // Update input/output sequences
        document.getElementById('inputSequence').textContent = data.input_values.join(', ');
        document.getElementById('outputSequence').textContent = data.output_values.join(', ');
    })
    .catch(error => {
        document.getElementById('executionLoadingSpinner').style.display = 'none';
        console.error('Error:', error);
        alert('An error occurred. Please try again.');
    });
});

// --- Add event listener for algorithm change ---
document.querySelectorAll('input[name="algorithm"]').forEach(radio => {
    radio.addEventListener('change', function() {
        const algoName = this.value === 'rl' ? 'RL Sorting Network' : "Batcher's Odd-Even Mergesort";
        document.getElementById('main-title').textContent = algoName;
        // Optional: You could also reset the visualization/properties here if desired
        // document.getElementById('networkVisualization').innerHTML = '<p class="text-muted">Generate a network to see visualization</p>';
        // document.getElementById('depthVisualization').innerHTML = '<p class="text-muted">Generate a network to see depth visualization</p>';
        // Clear properties...
    });
});
// --- End added listener ---

// Performance charts
let comparatorChart, depthChart, optimalSizeChart, optimalDepthChart, generationTimeChart;

// Tab change event for initializing charts
document.getElementById('performance-tab').addEventListener('click', function() {
    if (!comparatorChart) {
        fetch('/performance_data')
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(data.error);
                return;
            }
            
            // Prepare chart data
            const sizes = Object.keys(data.comparator_counts).map(Number);
            const comparatorCounts = Object.values(data.comparator_counts);
            const depths = Object.values(data.depths);
            const generationTimes = Object.values(data.generation_times);
            
            // Comparator count chart
            const ctxComparator = document.getElementById('comparatorChart').getContext('2d');
            comparatorChart = new Chart(ctxComparator, {
                type: 'line',
                data: {
                    labels: sizes,
                    datasets: [{
                        label: 'Number of Comparators',
                        data: comparatorCounts,
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1,
                        fill: false
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Input Size (n)'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Number of Comparators'
                            },
                            beginAtZero: true
                        }
                    }
                }
            });
            
            // Depth chart
            const ctxDepth = document.getElementById('depthChart').getContext('2d');
            depthChart = new Chart(ctxDepth, {
                type: 'line',
                data: {
                    labels: sizes,
                    datasets: [{
                        label: 'Network Depth',
                        data: depths,
                        borderColor: 'rgb(255, 99, 132)',
                        tension: 0.1,
                        fill: false
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Input Size (n)'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Depth (Parallel Steps)'
                            },
                            beginAtZero: true
                        }
                    }
                }
            });
            
            // Optimal comparison charts
            const comparison = data.optimal_comparison;
            const commonSizes = Object.keys(comparison.optimal_sizes).map(Number);
            
            // Size comparison
            const ctxOptimalSize = document.getElementById('optimalSizeChart').getContext('2d');
            optimalSizeChart = new Chart(ctxOptimalSize, {
                type: 'line',
                data: {
                    labels: commonSizes,
                    datasets: [{
                        label: 'Optimal Size',
                        data: Object.values(comparison.optimal_sizes),
                        borderColor: 'rgb(54, 162, 235)',
                        tension: 0.1,
                        fill: false
                    }, {
                        label: "Batcher's Size",
                        data: commonSizes.map(n => comparison.batcher_sizes[n]),
                        borderColor: 'rgb(255, 159, 64)',
                        tension: 0.1,
                        fill: false
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Input Size (n)'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Number of Comparators'
                            },
                            beginAtZero: true
                        }
                    }
                }
            });
            
            // Depth comparison
            const ctxOptimalDepth = document.getElementById('optimalDepthChart').getContext('2d');
            optimalDepthChart = new Chart(ctxOptimalDepth, {
                type: 'line',
                data: {
                    labels: commonSizes,
                    datasets: [{
                        label: 'Optimal Depth',
                        data: Object.values(comparison.optimal_depths),
                        borderColor: 'rgb(54, 162, 235)',
                        tension: 0.1,
                        fill: false
                    }, {
                        label: "Batcher's Depth",
                        data: commonSizes.map(n => comparison.batcher_depths[n]),
                        borderColor: 'rgb(255, 159, 64)',
                        tension: 0.1,
                        fill: false
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Input Size (n)'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Depth (Parallel Steps)'
                            },
                            beginAtZero: true
                        }
                    }
                }
            });

            // Generation time chart
            const ctxGenerationTime = document.getElementById('generationTimeChart').getContext('2d');
            generationTimeChart = new Chart(ctxGenerationTime, {
                type: 'line',
                data: {
                    labels: sizes,
                    datasets: [{
                        label: 'Generation Time (ms)',
                        data: generationTimes,
                        borderColor: 'rgb(75, 192, 192)',
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        borderWidth: 2,
                        tension: 0.1,
                        fill: true,
                        pointRadius: 3
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Input Size (n)'
                            }
                        },
                        y: {
                            type: 'logarithmic',
                            title: {
                                display: true,
                                text: 'Time (ms, log scale)'
                            }
                        }
                    },
                    plugins: {
                        title: {
                            display: true,
                            text: 'Time to Generate Sorting Network'
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    return `${context.raw.toFixed(2)} ms`;
                                }
                            }
                        }
                    }
                }
            });
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred loading performance data.');
        });
    }
}); 