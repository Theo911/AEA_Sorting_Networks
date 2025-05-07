// Global algorithm state management
let currentAlgorithm = localStorage.getItem('selectedAlgorithm') || 'batcher';

// Track the last algorithm used for each tab
let tabAlgorithmState = {
    'network': currentAlgorithm,
    'execution': currentAlgorithm,
    'performance': currentAlgorithm,
    'theory': currentAlgorithm
};

// Function to update all UI elements based on algorithm selection
function updateUIForAlgorithm(algorithm) {
    // Check if algorithm changed
    const algorithmChanged = currentAlgorithm !== algorithm;
    
    currentAlgorithm = algorithm;
    localStorage.setItem('selectedAlgorithm', algorithm);
    
    // Update main title
    const algoName = algorithm === 'rl' ? 'RL Sorting Network' : "Batcher's Odd-Even Mergesort";
    document.getElementById('main-title').textContent = algoName;
    
    // Update other UI elements based on the current tab
    // Network visualization tab
    if (document.getElementById('network').classList.contains('active')) {
        document.getElementById('network-viz-title').textContent = `${algoName} Visualization`;
        document.getElementById('depth-viz-title').textContent = `Depth Visualization (${algoName})`;
        document.getElementById('network-properties-title').textContent = `${algoName} Properties`;
        
        // Reset visualizations if algorithm changed
        if (algorithmChanged) {
            document.getElementById('networkVisualization').innerHTML = '<p class="text-muted">Generate a network to see visualization</p>';
            document.getElementById('depthVisualization').innerHTML = '<p class="text-muted">Generate a network to see depth visualization</p>';
            
            // Reset network properties
            document.getElementById('numComparators').textContent = '-';
            document.getElementById('networkDepth').textContent = '-';
            document.getElementById('redundancy').textContent = '-';
            document.getElementById('efficiency').textContent = '-';
            document.getElementById('numLayers').textContent = '-';
            document.getElementById('zeroOnePrinciple').textContent = '-';
            document.getElementById('minWireUsage').textContent = '-';
            document.getElementById('maxWireUsage').textContent = '-';
            document.getElementById('avgWireUsage').textContent = '-';
            document.getElementById('minCompPerLayer').textContent = '-';
            document.getElementById('maxCompPerLayer').textContent = '-';
            document.getElementById('avgCompPerLayer').textContent = '-';
        }
        
        // Update algorithm state for this tab
        tabAlgorithmState['network'] = algorithm;
    }
    
    // Update algorithm selection in all tabs
    document.querySelectorAll(`input[name="algorithm"][value="${algorithm}"]`)
        .forEach(radio => { radio.checked = true; });
    
    // Update theory tab selection
    if (algorithm === 'rl') {
        if (document.getElementById('rlTheoryBtn')) {
            document.getElementById('rlTheoryBtn').click();
        }
    } else {
        if (document.getElementById('batcherTheoryBtn')) {
            document.getElementById('batcherTheoryBtn').click();
        }
    }
    
    // Always update tab algorithm state for all tabs
    Object.keys(tabAlgorithmState).forEach(tab => {
        if (!document.getElementById(tab).classList.contains('active')) {
            // This tab isn't active, so mark it as needing a reset next time it's viewed
            tabAlgorithmState[tab] = null;
        } else {
            tabAlgorithmState[tab] = algorithm;
        }
    });
}

// Initialize UI based on stored preference when page loads
document.addEventListener('DOMContentLoaded', function() {
    console.log("Document ready - initializing app");
    
    // Initialize all tab algorithm states to current algorithm
    Object.keys(tabAlgorithmState).forEach(tab => {
        tabAlgorithmState[tab] = currentAlgorithm;
    });
    
    // Destroy any charts that might exist (should be none, but just in case)
    destroyExistingCharts('document-ready');
    
    // Add a global event listener for performance tab to handle potential chart issues
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.target.id === 'performance' && 
                mutation.attributeName === 'class' &&
                mutation.target.classList.contains('active')) {
                console.log("Performance tab became active via DOM mutation");
                setTimeout(() => {
                    destroyExistingCharts('mutation-observer');
                    if (!comparatorChart) {
                        console.log("Charts don't exist after tab activation, creating them");
                        loadPerformanceData('all');
                    }
                }, 50);
            }
        });
    });
    
    // Start observing the performance tab for class changes
    const performanceTab = document.getElementById('performance');
    if (performanceTab) {
        observer.observe(performanceTab, { attributes: true });
    }
    
    // Update UI based on current algorithm
    updateUIForAlgorithm(currentAlgorithm);
    
    // Add event listener for input size changes to reset visualizations
    document.getElementById('inputSize').addEventListener('change', function() {
        // Reset network visualizations when input size changes
        document.getElementById('networkVisualization').innerHTML = '<p class="text-muted">Generate a network to see visualization</p>';
        document.getElementById('depthVisualization').innerHTML = '<p class="text-muted">Generate a network to see depth visualization</p>';
        
        // Reset network properties
        document.getElementById('numComparators').textContent = '-';
        document.getElementById('networkDepth').textContent = '-';
        document.getElementById('redundancy').textContent = '-';
        document.getElementById('efficiency').textContent = '-';
        document.getElementById('numLayers').textContent = '-';
        document.getElementById('zeroOnePrinciple').textContent = '-';
        document.getElementById('minWireUsage').textContent = '-';
        document.getElementById('maxWireUsage').textContent = '-';
        document.getElementById('avgWireUsage').textContent = '-';
        document.getElementById('minCompPerLayer').textContent = '-';
        document.getElementById('maxCompPerLayer').textContent = '-';
        document.getElementById('avgCompPerLayer').textContent = '-';
    });
});

// Network visualization form submission
document.getElementById('networkForm').addEventListener('submit', function(e) {
    e.preventDefault();
    
    // Show loading spinner
    document.getElementById('loadingSpinner').style.display = 'inline-block';
    
    // Get form data
    const formData = new FormData(this);
    const selectedAlgorithm = formData.get('algorithm');
    
    // Update global algorithm state
    updateUIForAlgorithm(selectedAlgorithm);
    
    // Determine endpoint based on algorithm
    const endpoint = selectedAlgorithm === 'rl' ? '/generate_rl_network' : '/generate_network';
    
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
    
    // Reset execution visualizations when input size changes
    document.getElementById('executionVisualization').innerHTML = '<p class="text-muted">Execute the network to see visualization</p>';
    document.getElementById('inputSequence').textContent = '';
    document.getElementById('outputSequence').textContent = '';
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

// Algorithm selection event listener for all tabs
document.querySelectorAll('input[name="algorithm"]').forEach(radio => {
    radio.addEventListener('change', function() {
        const selectedAlgorithm = this.value;
        updateUIForAlgorithm(selectedAlgorithm);
        
        // Reset execution visualization if we're on the execution tab
        if (document.getElementById('execution').classList.contains('active')) {
            document.getElementById('executionVisualization').innerHTML = '<p class="text-muted">Execute the network to see visualization</p>';
            document.getElementById('inputSequence').textContent = '';
            document.getElementById('outputSequence').textContent = '';
            
            // Update this tab's algorithm state
            tabAlgorithmState['execution'] = selectedAlgorithm;
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
    
    // Update global algorithm state if needed
    if (currentAlgorithm !== 'batcher') {
        updateUIForAlgorithm('batcher');
    }
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
    
    // Update global algorithm state if needed
    if (currentAlgorithm !== 'rl') {
        updateUIForAlgorithm('rl');
    }
});

// Execution form submission
document.getElementById('executionForm').addEventListener('submit', function(e) {
    e.preventDefault();
    
    // Show loading spinner
    document.getElementById('executionLoadingSpinner').style.display = 'inline-block';
    
    // Get form data
    const formData = new FormData(this);
    
    // Add the current algorithm to the form data
    formData.append('algorithm', currentAlgorithm);
    
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

// Tab change event handlers
document.querySelectorAll('button[data-bs-toggle="tab"]').forEach(tab => {
    tab.addEventListener('shown.bs.tab', function(event) {
        const targetId = event.target.getAttribute('data-bs-target').substring(1);
        
        // Check if algorithm changed while away from this tab
        if (tabAlgorithmState[targetId] !== currentAlgorithm) {
            // Algorithm changed, reset tab-specific visualizations
            if (targetId === 'network') {
                // Reset network visualizations
                document.getElementById('networkVisualization').innerHTML = '<p class="text-muted">Generate a network to see visualization</p>';
                document.getElementById('depthVisualization').innerHTML = '<p class="text-muted">Generate a network to see depth visualization</p>';
                
                // Reset network properties
                document.getElementById('numComparators').textContent = '-';
                document.getElementById('networkDepth').textContent = '-';
                document.getElementById('redundancy').textContent = '-';
                document.getElementById('efficiency').textContent = '-';
                document.getElementById('numLayers').textContent = '-';
                document.getElementById('zeroOnePrinciple').textContent = '-';
                document.getElementById('minWireUsage').textContent = '-';
                document.getElementById('maxWireUsage').textContent = '-';
                document.getElementById('avgWireUsage').textContent = '-';
                document.getElementById('minCompPerLayer').textContent = '-';
                document.getElementById('maxCompPerLayer').textContent = '-';
                document.getElementById('avgCompPerLayer').textContent = '-';
                
                // Update tab titles
                const algoName = currentAlgorithm === 'rl' ? 'RL Sorting Network' : "Batcher's Odd-Even Mergesort";
                document.getElementById('network-viz-title').textContent = `${algoName} Visualization`;
                document.getElementById('depth-viz-title').textContent = `Depth Visualization (${algoName})`;
                document.getElementById('network-properties-title').textContent = `${algoName} Properties`;
            } else if (targetId === 'execution') {
                // Reset execution visualizations
                document.getElementById('executionVisualization').innerHTML = '<p class="text-muted">Execute the network to see visualization</p>';
                document.getElementById('inputSequence').textContent = '';
                document.getElementById('outputSequence').textContent = '';
            } else if (targetId === 'performance') {
                // Handle performance tab with care - use setTimeout to avoid race conditions
                console.log("Performance tab needs reset due to algorithm change");
                destroyExistingCharts('tab-change-algorithm-change');
                
                // Use setTimeout to ensure DOM is updated before creating new charts
                setTimeout(() => {
                    // Reload performance data based on current algorithm
                    if (currentAlgorithm === 'batcher') {
                        document.getElementById('showBatcherBtn').click();
                    } else if (currentAlgorithm === 'rl') {
                        document.getElementById('showRLBtn').click();
                    } else {
                        loadPerformanceData('all');
                    }
                }, 100);
            }
            
            // Update stored algorithm state for this tab
            tabAlgorithmState[targetId] = currentAlgorithm;
        }
        
        // Additional tab-specific behavior
        if (targetId === 'theory') {
            // Pre-select the theory content based on current algorithm
            if (currentAlgorithm === 'rl') {
                document.getElementById('rlTheoryBtn').click();
            } else {
                document.getElementById('batcherTheoryBtn').click();
            }
            
            // Update stored algorithm state for this tab
            tabAlgorithmState[targetId] = currentAlgorithm;
        } else if (targetId === 'performance') {
            console.log("Performance tab activated - checking chart state");
            logChartState('tab-change-performance');
            
            // Always make sure charts are properly initialized or reinitialized
            destroyExistingCharts('tab-change-performance');
            
            // Use setTimeout to ensure DOM is updated before creating new charts
            setTimeout(() => {
                if (currentAlgorithm === 'batcher') {
                    document.getElementById('showBatcherBtn').click();
                } else if (currentAlgorithm === 'rl') {
                    document.getElementById('showRLBtn').click();
                } else {
                    loadPerformanceData('all');
                }
                
                // Update stored algorithm state for this tab
                tabAlgorithmState[targetId] = currentAlgorithm;
            }, 100);
        }
    });
});

// Performance charts - initialize as null to ensure they don't exist initially
let comparatorChart = null;
let depthChart = null;
let optimalSizeChart = null;
let optimalDepthChart = null;
let generationTimeChart = null;

// Add debug function to check chart state
function logChartState(location) {
    console.log(`[${location}] Chart states:`, {
        comparatorChart: comparatorChart ? 'exists' : 'null',
        depthChart: depthChart ? 'exists' : 'null',
        optimalSizeChart: optimalSizeChart ? 'exists' : 'null',
        optimalDepthChart: optimalDepthChart ? 'exists' : 'null',
        generationTimeChart: generationTimeChart ? 'exists' : 'null'
    });
}

// Tab change event for initializing charts
document.getElementById('performance-tab').addEventListener('click', function() {
    console.log("Performance tab clicked");
    
    // Always make sure charts are destroyed before creating new ones
    destroyExistingCharts('tab-click');
    
    // Only load data if performance tab is now active - with a small delay to allow DOM to update
    setTimeout(() => {
        if (document.getElementById('performance').classList.contains('active')) {
            console.log("Loading performance data after tab click");
            loadPerformanceData('all');
        }
    }, 100);
});

// Function to destroy any existing charts to prevent "Canvas already in use" errors
function destroyExistingCharts(source) {
    console.log(`Destroying charts (called from: ${source})`);
    logChartState('before-destroy');
    
    try {
        if (comparatorChart) {
            console.log("Destroying comparatorChart");
            comparatorChart.destroy();
        }
    } catch (e) {
        console.error("Error destroying comparatorChart:", e);
    }
    comparatorChart = null;
    
    try {
        if (depthChart) {
            console.log("Destroying depthChart");
            depthChart.destroy();
        }
    } catch (e) {
        console.error("Error destroying depthChart:", e);
    }
    depthChart = null;
    
    try {
        if (optimalSizeChart) {
            console.log("Destroying optimalSizeChart");
            optimalSizeChart.destroy();
        }
    } catch (e) {
        console.error("Error destroying optimalSizeChart:", e);
    }
    optimalSizeChart = null;
    
    try {
        if (optimalDepthChart) {
            console.log("Destroying optimalDepthChart");
            optimalDepthChart.destroy();
        }
    } catch (e) {
        console.error("Error destroying optimalDepthChart:", e);
    }
    optimalDepthChart = null;
    
    try {
        if (generationTimeChart) {
            console.log("Destroying generationTimeChart");
            generationTimeChart.destroy();
        }
    } catch (e) {
        console.error("Error destroying generationTimeChart:", e);
    }
    generationTimeChart = null;
    
    logChartState('after-destroy');
}

// Function to load performance data with specified algorithm filter
function loadPerformanceData(algorithmFilter) {
    console.log(`Loading performance data for algorithm: ${algorithmFilter}`);
    
    // Clear any existing charts
    destroyExistingCharts('loadPerformanceData');
    
    // Clear canvas elements to ensure fresh state
    document.querySelectorAll('canvas').forEach(canvas => {
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    });
    
    fetch(`/performance_data?algorithm=${algorithmFilter}`)
    .then(response => response.json())
    .then(data => {
        console.log("Performance data received");
        
        if (data.error) {
            console.error("Server returned error:", data.error);
            alert(data.error);
            return;
        }
        
        // For 'all' mode, we need to combine datasets
        if (algorithmFilter === 'all') {
            // Create charts with both algorithms
            createComparisonCharts(data);
        } else {
            // Create charts for a single algorithm
            createSingleAlgorithmCharts(data);
        }
        
        console.log("Charts created successfully");
        logChartState('after-chart-creation');
    })
    .catch(error => {
        console.error('Error fetching performance data:', error);
        alert('An error occurred loading performance data.');
    });
}

// Function to create charts showing a single algorithm's data
function createSingleAlgorithmCharts(data) {
    console.log("Creating single algorithm charts");
    
    setTimeout(() => {
        try {
            const sizes = Object.keys(data.comparator_counts).map(Number);
            const comparatorCounts = Object.values(data.comparator_counts);
            const depths = Object.values(data.depths);
            const generationTimes = Object.values(data.generation_times);
            
            // Determine the color based on current algorithm
            const borderColor = currentAlgorithm === 'rl' ? 'rgb(153, 102, 255)' : 'rgb(75, 192, 192)';
            
            // Comparator count chart
            const ctxComparator = document.getElementById('comparatorChart');
            if (!ctxComparator) {
                console.error("Cannot find comparatorChart canvas element");
                return;
            }
            
            // Reset canvas to avoid reuse issues
            ctxComparator.getContext('2d').clearRect(0, 0, ctxComparator.width, ctxComparator.height);
            
            console.log("Creating comparatorChart");
            comparatorChart = new Chart(ctxComparator.getContext('2d'), {
                type: 'line',
                data: {
                    labels: sizes,
                    datasets: [{
                        label: 'Number of Comparators',
                        data: comparatorCounts,
                        borderColor: borderColor,
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
            const ctxDepth = document.getElementById('depthChart');
            if (!ctxDepth) {
                console.error("Cannot find depthChart canvas element");
                return;
            }
            
            // Reset canvas to avoid reuse issues
            ctxDepth.getContext('2d').clearRect(0, 0, ctxDepth.width, ctxDepth.height);
            
            console.log("Creating depthChart");
            depthChart = new Chart(ctxDepth.getContext('2d'), {
                type: 'line',
                data: {
                    labels: sizes,
                    datasets: [{
                        label: 'Network Depth',
                        data: depths,
                        borderColor: borderColor,
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
            const ctxOptimalSize = document.getElementById('optimalSizeChart');
            if (!ctxOptimalSize) {
                console.error("Cannot find optimalSizeChart canvas element");
                return;
            }
            
            // Reset canvas to avoid reuse issues
            ctxOptimalSize.getContext('2d').clearRect(0, 0, ctxOptimalSize.width, ctxOptimalSize.height);
            
            console.log("Creating optimalSizeChart");
            optimalSizeChart = new Chart(ctxOptimalSize.getContext('2d'), {
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
                        label: currentAlgorithm === 'rl' ? "RL Size" : "Batcher's Size",
                        data: commonSizes.map(n => comparison.batcher_sizes[n]),
                        borderColor: borderColor,
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
            const ctxOptimalDepth = document.getElementById('optimalDepthChart');
            if (!ctxOptimalDepth) {
                console.error("Cannot find optimalDepthChart canvas element");
                return;
            }
            
            // Reset canvas to avoid reuse issues
            ctxOptimalDepth.getContext('2d').clearRect(0, 0, ctxOptimalDepth.width, ctxOptimalDepth.height);
            
            console.log("Creating optimalDepthChart");
            optimalDepthChart = new Chart(ctxOptimalDepth.getContext('2d'), {
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
                        label: currentAlgorithm === 'rl' ? "RL Depth" : "Batcher's Depth",
                        data: commonSizes.map(n => comparison.batcher_depths[n]),
                        borderColor: borderColor,
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
            const ctxGenerationTime = document.getElementById('generationTimeChart');
            if (!ctxGenerationTime) {
                console.error("Cannot find generationTimeChart canvas element");
                return;
            }
            
            // Reset canvas to avoid reuse issues
            ctxGenerationTime.getContext('2d').clearRect(0, 0, ctxGenerationTime.width, ctxGenerationTime.height);
            
            console.log("Creating generationTimeChart");
            generationTimeChart = new Chart(ctxGenerationTime.getContext('2d'), {
                type: 'line',
                data: {
                    labels: sizes,
                    datasets: [{
                        label: 'Generation Time (ms)',
                        data: generationTimes,
                        borderColor: borderColor,
                        backgroundColor: `${borderColor.slice(0, -1)}, 0.2)`,
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
            
            logChartState('after-single-charts');
        } catch (error) {
            console.error("Error creating single algorithm charts:", error);
        }
    }, 100); // Small delay to ensure DOM is ready
}

// Function to create charts comparing both algorithms
function createComparisonCharts(data) {
    console.log("Creating comparison charts");
    
    setTimeout(() => {
        try {
            // Common sizes for x-axis
            const sizes = Object.keys(data.batcher_comparator_counts).map(Number);
            
            // Comparator count chart
            const ctxComparator = document.getElementById('comparatorChart');
            if (!ctxComparator) {
                console.error("Cannot find comparatorChart canvas element");
                return;
            }
            
            // Reset canvas to avoid reuse issues
            ctxComparator.getContext('2d').clearRect(0, 0, ctxComparator.width, ctxComparator.height);
            
            console.log("Creating comparison comparatorChart");
            comparatorChart = new Chart(ctxComparator.getContext('2d'), {
                type: 'line',
                data: {
                    labels: sizes,
                    datasets: [{
                        label: "Batcher's Comparators",
                        data: Object.values(data.batcher_comparator_counts),
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1,
                        fill: false
                    }, {
                        label: "RL Comparators",
                        data: Object.values(data.rl_comparator_counts),
                        borderColor: 'rgb(153, 102, 255)',
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
            const ctxDepth = document.getElementById('depthChart');
            if (!ctxDepth) {
                console.error("Cannot find depthChart canvas element");
                return;
            }
            
            // Reset canvas to avoid reuse issues
            ctxDepth.getContext('2d').clearRect(0, 0, ctxDepth.width, ctxDepth.height);
            
            console.log("Creating comparison depthChart");
            depthChart = new Chart(ctxDepth.getContext('2d'), {
                type: 'line',
                data: {
                    labels: sizes,
                    datasets: [{
                        label: "Batcher's Depth",
                        data: Object.values(data.batcher_depths),
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1,
                        fill: false
                    }, {
                        label: "RL Depth",
                        data: Object.values(data.rl_depths),
                        borderColor: 'rgb(153, 102, 255)',
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
            const ctxOptimalSize = document.getElementById('optimalSizeChart');
            if (!ctxOptimalSize) {
                console.error("Cannot find optimalSizeChart canvas element");
                return;
            }
            
            // Reset canvas to avoid reuse issues
            ctxOptimalSize.getContext('2d').clearRect(0, 0, ctxOptimalSize.width, ctxOptimalSize.height);
            
            console.log("Creating comparison optimalSizeChart");
            optimalSizeChart = new Chart(ctxOptimalSize.getContext('2d'), {
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
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1,
                        fill: false
                    }, {
                        label: "RL Size",
                        data: commonSizes.map(n => data.rl_comparator_counts[n] || null),
                        borderColor: 'rgb(153, 102, 255)',
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
            const ctxOptimalDepth = document.getElementById('optimalDepthChart');
            if (!ctxOptimalDepth) {
                console.error("Cannot find optimalDepthChart canvas element");
                return;
            }
            
            // Reset canvas to avoid reuse issues
            ctxOptimalDepth.getContext('2d').clearRect(0, 0, ctxOptimalDepth.width, ctxOptimalDepth.height);
            
            console.log("Creating comparison optimalDepthChart");
            optimalDepthChart = new Chart(ctxOptimalDepth.getContext('2d'), {
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
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1,
                        fill: false
                    }, {
                        label: "RL Depth",
                        data: commonSizes.map(n => data.rl_depths[n] || null),
                        borderColor: 'rgb(153, 102, 255)',
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
            const ctxGenerationTime = document.getElementById('generationTimeChart');
            if (!ctxGenerationTime) {
                console.error("Cannot find generationTimeChart canvas element");
                return;
            }
            
            // Reset canvas to avoid reuse issues
            ctxGenerationTime.getContext('2d').clearRect(0, 0, ctxGenerationTime.width, ctxGenerationTime.height);
            
            console.log("Creating comparison generationTimeChart");
            generationTimeChart = new Chart(ctxGenerationTime.getContext('2d'), {
                type: 'line',
                data: {
                    labels: sizes,
                    datasets: [{
                        label: "Batcher's Generation Time (ms)",
                        data: Object.values(data.batcher_generation_times),
                        borderColor: 'rgb(75, 192, 192)',
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        borderWidth: 2,
                        tension: 0.1,
                        fill: true,
                        pointRadius: 3
                    }, {
                        label: "RL Generation Time (ms)",
                        data: Object.values(data.rl_generation_times),
                        borderColor: 'rgb(153, 102, 255)',
                        backgroundColor: 'rgba(153, 102, 255, 0.2)',
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
            
            logChartState('after-comparison-charts');
        } catch (error) {
            console.error("Error creating comparison charts:", error);
        }
    }, 100);
}

// Performance tab algorithm toggle
document.getElementById('showBatcherBtn')?.addEventListener('click', function() {
    console.log("Batcher button clicked");
    
    // Update button states
    this.classList.add('active');
    this.classList.add('btn-primary');
    this.classList.remove('btn-outline-primary');
    
    document.getElementById('showRLBtn').classList.remove('active');
    document.getElementById('showRLBtn').classList.remove('btn-primary');
    document.getElementById('showRLBtn').classList.add('btn-outline-primary');
    
    document.getElementById('showComparisonBtn').classList.remove('active');
    document.getElementById('showComparisonBtn').classList.remove('btn-primary');
    document.getElementById('showComparisonBtn').classList.add('btn-outline-primary');
    
    // Update performance note
    document.getElementById('performanceNote').textContent = "Showing performance data for Batcher's algorithm. Select an option above to change the view.";
    
    // Update global algorithm state if needed
    if (currentAlgorithm !== 'batcher') {
        updateUIForAlgorithm('batcher');
    }
    
    // Ensure charts are destroyed before loading new data
    destroyExistingCharts('batcher-btn-click');
    
    // Use setTimeout to ensure DOM is fully updated before creating charts
    setTimeout(() => {
        // Load Batcher-specific data
        loadPerformanceData('batcher');
    }, 50);
});

document.getElementById('showRLBtn')?.addEventListener('click', function() {
    console.log("RL button clicked");
    
    // Update button states
    this.classList.add('active');
    this.classList.add('btn-primary');
    this.classList.remove('btn-outline-primary');
    
    document.getElementById('showBatcherBtn').classList.remove('active');
    document.getElementById('showBatcherBtn').classList.remove('btn-primary');
    document.getElementById('showBatcherBtn').classList.add('btn-outline-primary');
    
    document.getElementById('showComparisonBtn').classList.remove('active');
    document.getElementById('showComparisonBtn').classList.remove('btn-primary');
    document.getElementById('showComparisonBtn').classList.add('btn-outline-primary');
    
    // Update performance note
    document.getElementById('performanceNote').textContent = "Showing performance data for RL algorithm. Select an option above to change the view.";
    
    // Update global algorithm state if needed
    if (currentAlgorithm !== 'rl') {
        updateUIForAlgorithm('rl');
    }
    
    // Ensure charts are destroyed before loading new data
    destroyExistingCharts('rl-btn-click');
    
    // Use setTimeout to ensure DOM is fully updated before creating charts
    setTimeout(() => {
        // Load RL-specific data
        loadPerformanceData('rl');
    }, 50);
});

document.getElementById('showComparisonBtn')?.addEventListener('click', function() {
    console.log("Comparison button clicked");
    
    // Update button states
    this.classList.add('active');
    this.classList.add('btn-primary');
    this.classList.remove('btn-outline-primary');
    
    document.getElementById('showBatcherBtn').classList.remove('active');
    document.getElementById('showBatcherBtn').classList.remove('btn-primary');
    document.getElementById('showBatcherBtn').classList.add('btn-outline-primary');
    
    document.getElementById('showRLBtn').classList.remove('active');
    document.getElementById('showRLBtn').classList.remove('btn-primary');
    document.getElementById('showRLBtn').classList.add('btn-outline-primary');
    
    // Update performance note
    document.getElementById('performanceNote').textContent = "Comparing performance of both algorithms. Select an option above to focus on a specific algorithm.";
    
    // Don't update global algorithm state here as we're in comparison mode
    
    // Ensure charts are destroyed before loading new data
    destroyExistingCharts('comparison-btn-click');
    
    // Use setTimeout to ensure DOM is fully updated before creating charts
    setTimeout(() => {
        // Load comparison data
        loadPerformanceData('all');
    }, 50);
}); 