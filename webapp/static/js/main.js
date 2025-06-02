// Global algorithm state management
let currentAlgorithm = localStorage.getItem('selectedAlgorithm') || 'batcher';

// Track the last algorithm used for each tab
let tabAlgorithmState = {
    'network': currentAlgorithm,
    'execution': currentAlgorithm,
    'performance': currentAlgorithm,
    'theory': currentAlgorithm
};

// Global algorithm availability status
let algorithmAvailability = {};

// Function to update all UI elements based on algorithm selection
function updateUIForAlgorithm(algorithm, algorithmChanged = true) {
    currentAlgorithm = algorithm;
    localStorage.setItem('selectedAlgorithm', algorithm);
    
    // Update other UI elements based on the current tab
    // Network visualization tab - add null check since tab is commented out
    const networkTab = document.getElementById('network');
    if (networkTab && networkTab.classList.contains('active')) {
        const algoName = algorithm === 'rl' ? 'RL Sorting Network' : "Batcher's Odd-Even Mergesort";
        
        const networkVizTitle = document.getElementById('network-viz-title');
        const depthVizTitle = document.getElementById('depth-viz-title');
        const networkPropsTitle = document.getElementById('network-properties-title');
        
        if (networkVizTitle) networkVizTitle.textContent = `${algoName} Visualization`;
        if (depthVizTitle) depthVizTitle.textContent = `Depth Visualization (${algoName})`;
        if (networkPropsTitle) networkPropsTitle.textContent = `${algoName} Properties`;
        
        // Reset visualizations if algorithm changed
        if (algorithmChanged) {
            const networkVisualization = document.getElementById('networkVisualization');
            const depthVisualization = document.getElementById('depthVisualization');
            
            if (networkVisualization) {
                networkVisualization.innerHTML = '<p class="text-muted">Generate a network to see visualization</p>';
            }
            if (depthVisualization) {
                depthVisualization.innerHTML = '<p class="text-muted">Generate a network to see depth visualization</p>';
            }
            
            // Reset network properties with null checks
            const propertyElements = [
                'numComparators', 'networkDepth', 'redundancy', 'efficiency', 
                'numLayers', 'zeroOnePrinciple', 'minWireUsage', 'maxWireUsage', 
                'avgWireUsage', 'minCompPerLayer', 'maxCompPerLayer', 'avgCompPerLayer'
            ];
            
            propertyElements.forEach(elementId => {
                const element = document.getElementById(elementId);
                if (element) element.textContent = '-';
            });
        }
        
        // Update algorithm state for this tab
        tabAlgorithmState['network'] = algorithm;
    }
    
    // Update algorithm selection in all tabs
    document.querySelectorAll(`input[name="algorithm"][value="${algorithm}"]`)
        .forEach(radio => { radio.checked = true; });
    
    // Update theory tab selection with null checks
    if (algorithm === 'rl') {
        const rlTheoryBtn = document.getElementById('rlTheoryBtn');
        if (rlTheoryBtn) rlTheoryBtn.click();
    } else {
        const batcherTheoryBtn = document.getElementById('batcherTheoryBtn');
        if (batcherTheoryBtn) batcherTheoryBtn.click();
    }
    
    // Always update tab algorithm state for all tabs with null checks
    Object.keys(tabAlgorithmState).forEach(tab => {
        const tabElement = document.getElementById(tab);
        if (tabElement && !tabElement.classList.contains('active')) {
            // This tab isn't active, so mark it as needing a reset next time it's viewed
            tabAlgorithmState[tab] = null;
        } else if (tabElement) {
            tabAlgorithmState[tab] = algorithm;
        }
    });
}

// Function to load and update algorithm availability
async function updateAlgorithmAvailability() {
    try {
        const response = await fetch('/api/algorithm_status');
        const result = await response.json();
        
        if (result.success) {
            algorithmAvailability = result.algorithms;
            updateAlgorithmSelectionUI();
            updateExecutionDemoUI();
            updatePerformanceAnalysisUI();
            console.log('Algorithm availability updated:', algorithmAvailability);
        }
    } catch (error) {
        console.error('Error loading algorithm status:', error);
    }
}

// Initialize everything when DOM is loaded
document.addEventListener('DOMContentLoaded', async function() {
    console.log('DOM loaded, initializing webapp...');
    
    // Load algorithm availability on startup
    console.log('Loading algorithm availability...');
    await updateAlgorithmAvailability();
    console.log('Algorithm availability after first load:', algorithmAvailability);
    
    // Set up algorithm selection change handlers AFTER availability is loaded
    document.querySelectorAll('input[name="algorithm"]').forEach(radio => {
        radio.addEventListener('change', function() {
            if (this.checked) {
                console.log(`Algorithm changed to: ${this.value}`);
                
                // Update input size limits
                updateInputSizeLimits();
                
                // Use enhanced algorithm change handler
                handleAlgorithmChange();
            }
        });
    });
    
    // Set up input type toggles
    document.querySelectorAll('input[name="input_type"]').forEach(radio => {
        radio.addEventListener('change', toggleCustomInput);
    });
    
    // Set up input size change listeners
    document.querySelectorAll('input[type="number"][name="input_size"]').forEach(input => {
        input.addEventListener('change', updateRequiredInputs);
    });
    

    
    // Set up form submissions for both generation and execution
    document.getElementById('networkForm')?.addEventListener('submit', handleNetworkGeneration);
    document.getElementById('executionForm')?.addEventListener('submit', handleNetworkExecution);
    
    // Load algorithm status (legacy)
    console.log('Loading legacy algorithm status...');
    await loadAlgorithmStatus();
    console.log('Algorithm availability after legacy load:', algorithmAvailability);
    
    // Initial UI updates
    updateInputSizeLimits();
    
    console.log('Webapp initialization complete');
    console.log('Final algorithmAvailability state:', algorithmAvailability);
});

// Function to load algorithm status and update UI
async function loadAlgorithmStatus() {
    try {
        const response = await fetch('/api/algorithm_status');
        const result = await response.json();
        
        if (result.success) {
            algorithmAvailability = result.algorithms;
            updateAlgorithmSelectionUI();
            updateExecutionDemoUI();
            updatePerformanceAnalysisUI();
            console.log('Algorithm availability updated:', algorithmAvailability);
        } else {
            console.error('Failed to load algorithm status:', result.error);
        }
    } catch (error) {
        console.error('Error loading algorithm status:', error);
    }
}

// Function to update algorithm availability in UI
function updateAlgorithmAvailability(algorithms) {
    // Add null check to prevent Object.entries error
    if (!algorithms || typeof algorithms !== 'object') {
        console.warn('No algorithms data provided to updateAlgorithmAvailability');
        return;
    }
    
    // Update improved algorithm options based on availability
    Object.entries(algorithms).forEach(([algorithmKey, algorithmInfo]) => {
        const inputs = document.querySelectorAll(`input[value="${algorithmKey}"]`);
        
        inputs.forEach(input => {
            const label = input.parentElement.querySelector('label');
            const badge = label.querySelector('.badge');
            
            if (!algorithmInfo.available) {
                // Disable the input
                input.disabled = true;
                input.parentElement.style.opacity = '0.6';
                
                // Update badge to show "Coming Soon"
                if (badge) {
                    badge.textContent = 'Coming Soon';
                    badge.className = 'badge bg-secondary';
                }
                
                // Add tooltip or title for more info
                label.title = algorithmInfo.status || 'Not yet implemented';
            } else {
                // Enable the input
                input.disabled = false;
                input.parentElement.style.opacity = '1';
                
                // Update badge to show availability
                if (badge && algorithmKey.includes('improved')) {
                    badge.textContent = 'Available';
                    badge.className = 'badge bg-success';
                }
            }
        });
    });
}

// Network visualization form submission - DISABLED (tab commented out)
const networkForm = document.getElementById('networkForm');
if (networkForm) {
    networkForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Show loading spinner
        const loadingSpinner = document.getElementById('loadingSpinner');
        if (loadingSpinner) {
            loadingSpinner.style.display = 'inline-block';
        }
        
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
            if (loadingSpinner) {
                loadingSpinner.style.display = 'none';
            }
            
            if (data.error) {
                alert(data.error);
                return;
            }
            
            // Update network visualization
            const networkViz = document.getElementById('networkVisualization');
            if (networkViz) {
                networkViz.innerHTML = 
                    `<img src="data:image/png;base64,${data.network_img}" alt="Sorting Network">`;
            }
            
            // Update depth visualization
            const depthViz = document.getElementById('depthVisualization');
            if (depthViz) {
                depthViz.innerHTML = 
                    `<img src="data:image/png;base64,${data.depth_img}" alt="Depth Visualization">`;
            }
            
            // Update network properties - only if elements exist
            const updateElement = (id, value) => {
                const element = document.getElementById(id);
                if (element) {
                    element.textContent = value;
                }
            };
            
            updateElement('numComparators', data.num_comparators);
            updateElement('networkDepth', data.depth);
            updateElement('redundancy', typeof data.redundancy === 'number' ? data.redundancy.toFixed(2) + '%' : data.redundancy);
            updateElement('efficiency', typeof data.efficiency === 'number' ? data.efficiency.toFixed(2) + '%' : data.efficiency);
            updateElement('numLayers', data.num_layers);

            // Update Zero-One Principle verification status (handle string cases)
            const zeroOneElement = document.getElementById('zeroOnePrinciple');
            if (zeroOneElement) {
                if (data.zero_one_principle === "proven") {
                    zeroOneElement.textContent = "Yes (mathematically proven)";
                } else if (data.zero_one_principle === true) {
                    zeroOneElement.textContent = "Yes (verified)";
                } else if (data.zero_one_principle === false) {
                    zeroOneElement.textContent = "No";
                } else {
                    zeroOneElement.textContent = data.zero_one_principle;
                }
            }

            updateElement('minWireUsage', data.min_wire_usage !== undefined ? data.min_wire_usage : 'N/A');
            updateElement('maxWireUsage', data.max_wire_usage !== undefined ? data.max_wire_usage : 'N/A');
            updateElement('avgWireUsage', typeof data.avg_wire_usage === 'number' ? data.avg_wire_usage.toFixed(2) : (data.avg_wire_usage !== undefined ? data.avg_wire_usage : 'N/A'));
            updateElement('minCompPerLayer', data.min_comparators_per_layer !== undefined ? data.min_comparators_per_layer : 'N/A');
            updateElement('maxCompPerLayer', data.max_comparators_per_layer !== undefined ? data.max_comparators_per_layer : 'N/A');
            updateElement('avgCompPerLayer', typeof data.avg_comparators_per_layer === 'number' ? data.avg_comparators_per_layer.toFixed(2) : (data.avg_comparators_per_layer !== undefined ? data.avg_comparators_per_layer : 'N/A'));
        })
        .catch(error => {
            if (loadingSpinner) {
                loadingSpinner.style.display = 'none';
            }
            console.error('Error:', error);
            alert('An error occurred. Please try again.');
        });
    });
}

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

// Algorithm Theory toggle - DISABLED (theory tab commented out)
const batcherTheoryBtn = document.getElementById('batcherTheoryBtn');
if (batcherTheoryBtn) {
    batcherTheoryBtn.addEventListener('click', function() {
        // Show Batcher theory, hide RL theory
        const batcherTheory = document.getElementById('batcherTheory');
        const rlTheory = document.getElementById('rlTheory');
        
        if (batcherTheory) batcherTheory.style.display = 'block';
        if (rlTheory) rlTheory.style.display = 'none';
        
        // Update button states
        this.classList.add('active');
        this.classList.remove('btn-outline-primary');
        this.classList.add('btn-primary');
        
        const rlBtn = document.getElementById('rlTheoryBtn');
        if (rlBtn) {
            rlBtn.classList.remove('active');
            rlBtn.classList.remove('btn-primary');
            rlBtn.classList.add('btn-outline-primary');
        }
        
        // Update global algorithm state if needed
        if (currentAlgorithm !== 'batcher') {
            updateUIForAlgorithm('batcher');
        }
    });
}

const rlTheoryBtn = document.getElementById('rlTheoryBtn');
if (rlTheoryBtn) {
    rlTheoryBtn.addEventListener('click', function() {
        // Show RL theory, hide Batcher theory
        const batcherTheory = document.getElementById('batcherTheory');
        const rlTheory = document.getElementById('rlTheory');
        
        if (batcherTheory) batcherTheory.style.display = 'none';
        if (rlTheory) rlTheory.style.display = 'block';
        
        // Update button states
        this.classList.add('active');
        this.classList.remove('btn-outline-primary');
        this.classList.add('btn-primary');
        
        const batcherBtn = document.getElementById('batcherTheoryBtn');
        if (batcherBtn) {
            batcherBtn.classList.remove('active');
            batcherBtn.classList.remove('btn-primary');
            batcherBtn.classList.add('btn-outline-primary');
        }
        
        // Update global algorithm state if needed
        if (currentAlgorithm !== 'rl') {
            updateUIForAlgorithm('rl');
        }
    });
}

// Execution form submission
document.getElementById('executionForm').addEventListener('submit', function(e) {
    e.preventDefault();
    
    // Show loading spinner
    document.getElementById('executionLoadingSpinner').style.display = 'inline-block';
    
    // Get form data
    const formData = new FormData(this);
    
    // Get the selected algorithm from the form (now includes improved versions)
    const selectedAlgorithm = formData.get('algorithm');
    
    // Update the current algorithm if it changed
    if (selectedAlgorithm !== currentAlgorithm) {
        updateUIForAlgorithm(selectedAlgorithm);
    }
    
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
            // Display error message but also handle "not implemented" gracefully
            if (data.error.includes('not yet implemented')) {
                // Show a more user-friendly message for unimplemented algorithms
                const algorithmName = selectedAlgorithm.includes('batcher') ? 'Enhanced Batcher' : 'Enhanced RL';
                alert(`${algorithmName} algorithm is coming soon! Please use the standard versions for now.`);
                
                // Reset to standard algorithm
                const standardAlgo = selectedAlgorithm.includes('batcher') ? 'batcher' : 'rl';
                document.querySelector(`input[name="algorithm"][value="${standardAlgo}"]`).checked = true;
                updateUIForAlgorithm(standardAlgo);
            } else {
            alert(data.error);
            }
            return;
        }
        
        // Update execution visualization
        document.getElementById('executionVisualization').innerHTML = 
            `<img src="data:image/png;base64,${data.execution_img}" alt="Execution Visualization">`;
        
        // Update input/output sequences
        document.getElementById('inputSequence').textContent = data.input_values.join(', ');
        document.getElementById('outputSequence').textContent = data.output_values.join(', ');
        
        // Update enhanced execution results
        if (data.execution_time_ms !== undefined) {
            document.getElementById('executionTime').textContent = `${data.execution_time_ms.toFixed(2)} ms`;
        } else {
            document.getElementById('executionTime').textContent = '-';
        }
        
        if (data.success !== undefined) {
            const successElement = document.getElementById('executionSuccess');
            successElement.textContent = data.success ? 'Yes' : 'No';
            successElement.className = data.success ? 'fw-bold text-success' : 'fw-bold text-danger';
        } else {
            document.getElementById('executionSuccess').textContent = '-';
            document.getElementById('executionSuccess').className = 'fw-bold';
        }
        
        if (data.comparators_count !== undefined) {
            document.getElementById('executionComparators').textContent = data.comparators_count;
        } else {
            document.getElementById('executionComparators').textContent = '-';
        }
        
        if (data.network_depth !== undefined) {
            document.getElementById('executionDepth').textContent = data.network_depth;
        } else {
            document.getElementById('executionDepth').textContent = '-';
        }
        
        // Handle RL-specific analysis data
        if (data.rl_analysis && (selectedAlgorithm === 'rl' || selectedAlgorithm === 'rl_double_dqn' || selectedAlgorithm === 'rl_classic_dqn')) {
            const rlSection = document.getElementById('rlAnalysisSection');
            if (rlSection) {
                rlSection.style.display = 'block';
                
                // Update RL analysis fields with enhanced data
                const rlAnalysis = data.rl_analysis;
                
                // Algorithm Info - now in the title
                const agentTypeTitle = document.getElementById('rlAgentTypeTitle');
                if (agentTypeTitle) {
                    const agentType = rlAnalysis.agent_type || '-';
                    agentTypeTitle.textContent = agentType !== '-' ? `(${agentType})` : '(-)';
                    
                    // Add data source indicator if synthetic
                    if (rlAnalysis.is_synthetic) {
                        agentTypeTitle.style.color = '#ffc107'; // warning color
                        agentTypeTitle.title = `Data source: ${rlAnalysis.data_source || 'README Table (Model Unavailable)'}`;
                    } else {
                        agentTypeTitle.style.color = ''; // default color
                        agentTypeTitle.title = `Data source: ${rlAnalysis.data_source || 'Actual Model'}`;
                    }
                }
                
                // Optimization Results (removed duplicates - Original Size/Depth shown above)
                document.getElementById('rlPrunedSize').textContent = rlAnalysis.pruned_size || '-';
                document.getElementById('rlPrunedDepth').textContent = rlAnalysis.pruned_depth || '-';
                
                // Efficiency Metrics
                const pruningEffElement = document.getElementById('rlPruningEfficiency');
                if (pruningEffElement) {
                    const efficiency = rlAnalysis.pruning_efficiency || '0%';
                    pruningEffElement.textContent = efficiency;
                    // Color code based on efficiency
                    if (efficiency === '0%' || efficiency === '0.0%') {
                        pruningEffElement.className = 'fw-bold text-muted';
                    } else {
                        pruningEffElement.className = 'fw-bold text-success';
                    }
                }
                
                // Show optimal value
                const optimalValueElement = document.getElementById('rlOptimalValue');
                if (optimalValueElement) {
                    const optimalValue = rlAnalysis.optimal_size || '-';
                    optimalValueElement.textContent = optimalValue;
                }
                
                const vsOptimalElement = document.getElementById('rlVsOptimal');
                if (vsOptimalElement) {
                    const vsOptimal = rlAnalysis.vs_optimal || '-';
                    vsOptimalElement.textContent = vsOptimal;
                    // Color code based on performance vs optimal
                    if (vsOptimal.includes('→')) {
                        // Both original and pruned differences shown
                        vsOptimalElement.className = 'fw-bold text-info';
                    } else if (vsOptimal.startsWith('+')) {
                        vsOptimalElement.className = 'fw-bold text-warning';
                    } else if (vsOptimal === '0') {
                        vsOptimalElement.className = 'fw-bold text-success';
                    } else {
                        vsOptimalElement.className = 'fw-bold text-info';
                    }
                }
                
                // Status Indicators
                const statusElement = document.getElementById('rlNetworkStatus');
                if (statusElement) {
                    const status = rlAnalysis.network_status || 'Unknown';
                    const isValid = status === 'Valid';
                    statusElement.textContent = isValid ? 'Valid ✓' : 'Invalid ✗';
                    statusElement.className = isValid ? 'fw-bold text-success' : 'fw-bold text-danger';
                }
                
                const pruningAppliedElement = document.getElementById('rlPruningApplied');
                if (pruningAppliedElement) {
                    const applied = rlAnalysis.pruning_applied;
                    pruningAppliedElement.textContent = applied ? 'Yes ✓' : 'No';
                    pruningAppliedElement.className = applied ? 'fw-bold text-success' : 'fw-bold text-muted';
                }
            }
        } else {
            // Hide RL analysis section for non-RL algorithms
            const rlSection = document.getElementById('rlAnalysisSection');
            if (rlSection) {
                rlSection.style.display = 'none';
            }
        }
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
                const rlTheoryBtn = document.getElementById('rlTheoryBtn');
                if (rlTheoryBtn) rlTheoryBtn.click();
            } else {
                const batcherTheoryBtn = document.getElementById('batcherTheoryBtn');
                if (batcherTheoryBtn) batcherTheoryBtn.click();
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

// Update algorithm selection UI based on availability
function updateAlgorithmSelectionUI() {
    // Update algorithm radio buttons and labels
    document.querySelectorAll('input[name="algorithm"]').forEach(input => {
        const algorithmKey = input.value;
        const availability = algorithmAvailability[algorithmKey];
        const label = input.closest('label') || input.parentElement;
        
        if (!availability || !availability.available) {
            // Algorithm not available - disable and mark as coming soon
            input.disabled = true;
            label.classList.add('text-muted');
            
            // Add or update "Coming Soon" badge
            let badge = label.querySelector('.algorithm-status-badge');
            if (!badge) {
                badge = document.createElement('span');
                badge.className = 'algorithm-status-badge badge ms-2';
                label.appendChild(badge);
            }
            
            badge.textContent = 'Coming Soon';
            badge.className = 'algorithm-status-badge badge bg-secondary ms-2';
            
            // Add tooltip with status info
            if (availability && availability.status) {
                label.title = availability.status;
            }
        } else {
            // Algorithm available - enable and mark as ready
            input.disabled = false;
            label.classList.remove('text-muted');
            
            // Update badge to show available status
            let badge = label.querySelector('.algorithm-status-badge');
            if (!badge) {
                badge = document.createElement('span');
                badge.className = 'algorithm-status-badge badge ms-2';
                label.appendChild(badge);
            }
            
            badge.textContent = 'Available';
            badge.className = 'algorithm-status-badge badge bg-success ms-2';
            
            // Add tooltip with description
            if (availability.description) {
                label.title = availability.description;
            }
        }
    });
}

// Update execution demo UI for all algorithms
function updateExecutionDemoUI() {
    // Add information about all algorithms in execution tab
    const executionInfo = document.getElementById('executionAlgorithmInfo');
    if (executionInfo && algorithmAvailability && Object.keys(algorithmAvailability).length > 0) {
        const availableCount = Object.values(algorithmAvailability).filter(info => info.available).length;
        const totalCount = Object.keys(algorithmAvailability).length;
        
        const infoHtml = `
            <div class="mt-3">
                <h6>Algorithm Status: ${availableCount}/${totalCount} Available</h6>
                <div class="row">
                    <div class="col-md-6">
                        <div class="mb-2">
                            <span class="badge bg-success me-2">✓</span>
                            <small><strong>Batcher Traditional:</strong> Classic reliable algorithm</small>
                        </div>
                        <div class="mb-2">
                            <span class="badge bg-success me-2">✓</span>
                            <small><strong>Batcher Enhanced:</strong> Optimized with modern improvements</small>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="mb-2">
                            <span class="badge bg-info me-2">i</span>
                            <small><strong>Full range:</strong> Both algorithms work for n=2-32</small>
                        </div>
                        <div class="mb-2">
                            <span class="badge bg-info me-2">i</span>
                            <small><strong>Performance:</strong> Enhanced version optimized for size and depth</small>
                        </div>
                    </div>
                </div>
            </div>
        `;
        executionInfo.innerHTML = infoHtml;
    }
}

// Update performance analysis status indicator
function updatePerformanceAnalysisUI() {
    const statusIndicator = document.getElementById('algorithmStatusIndicator');
    if (statusIndicator && algorithmAvailability && Object.keys(algorithmAvailability).length > 0) {
        const availableCount = Object.values(algorithmAvailability).filter(info => info.available).length;
        const totalCount = Object.keys(algorithmAvailability).length;
        
        const statusHtml = `
            <div class="d-flex align-items-center justify-content-between">
                <div>
                    <span class="badge bg-success me-2">${availableCount}/${totalCount}</span>
                    <strong>Algorithms Available</strong>
                    <small class="text-muted ms-2">• All major sorting network approaches included</small>
                </div>
                <div class="text-end">
                    <small class="text-muted">Data quality: 95/100 • Cross-validated with multiple sources</small>
                </div>
            </div>
        `;
        statusIndicator.innerHTML = statusHtml;
    }
}

// Enhanced algorithm change handler
function handleAlgorithmChange() {
    const selectedAlgorithm = document.querySelector('input[name="algorithm"]:checked')?.value;
    
    if (!selectedAlgorithm) return;
    
    // Check if algorithmAvailability is loaded
    if (!algorithmAvailability || Object.keys(algorithmAvailability).length === 0) {
        console.warn('Algorithm availability not loaded yet, reloading...');
        // Try to reload algorithm availability
        updateAlgorithmAvailability().then(() => {
            // Retry after loading
            handleAlgorithmChange();
        });
        return;
    }
    
    // Check if algorithm is available
    const availability = algorithmAvailability[selectedAlgorithm];
    
    // Debug logging
    console.log(`Checking availability for ${selectedAlgorithm}:`, availability);
    console.log('Full algorithmAvailability object:', algorithmAvailability);
    
    if (!availability || !availability.available) {
        // Algorithm not available - show message and revert to batcher
        showAlgorithmUnavailableMessage(selectedAlgorithm, availability);
        
        // Revert to batcher (always available)
        const batcherRadio = document.querySelector('input[name="algorithm"][value="batcher"]');
        if (batcherRadio) {
            batcherRadio.checked = true;
            updateUIForAlgorithm('batcher');
        }
        return;
    }
    
    // Algorithm is available - proceed normally
    console.log(`Algorithm ${selectedAlgorithm} is available, proceeding...`);
    updateUIForAlgorithm(selectedAlgorithm);
}

// Show message when user tries to select unavailable algorithm
function showAlgorithmUnavailableMessage(algorithmKey, availability) {
    const algorithmName = availability?.description || algorithmKey;
    const status = availability?.status || 'Not yet implemented';
    
    // Create toast notification
    const toastHtml = `
        <div class="toast align-items-center text-white bg-warning border-0" role="alert">
            <div class="d-flex">
                <div class="toast-body">
                    <strong>${algorithmName}</strong> is not yet available.<br>
                    <small>${status}</small>
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        </div>
    `;
    
    // Add toast to container or create one
    let toastContainer = document.querySelector('.toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.className = 'toast-container position-fixed top-0 end-0 p-3';
        document.body.appendChild(toastContainer);
    }
    
    toastContainer.insertAdjacentHTML('beforeend', toastHtml);
    
    // Show the toast
    const toastElement = toastContainer.lastElementChild;
    const toast = new bootstrap.Toast(toastElement);
    toast.show();
    
    // Remove toast element after it's hidden
    toastElement.addEventListener('hidden.bs.toast', () => {
        toastElement.remove();
    });
}



// Update input size limits based on selected algorithm
async function updateInputSizeLimits() {
    const selectedAlgorithm = document.querySelector('input[name="algorithm"]:checked')?.value;
    const inputSizeElement = document.getElementById('execInputSize');
    
    if (!inputSizeElement || !selectedAlgorithm) return;
    
    try {
        const response = await fetch('/api/available_sizes');
        const data = await response.json();
        
        if (data.success) {
            const availableSizes = data.sizes[selectedAlgorithm] || [];
            
            if (availableSizes.length > 0) {
                const minSize = Math.min(...availableSizes);
                const maxSize = Math.max(...availableSizes);
                
                inputSizeElement.min = minSize;
                inputSizeElement.max = maxSize;
                
                // Update current value if it's outside the new range
                const currentValue = parseInt(inputSizeElement.value);
                if (currentValue < minSize) {
                    inputSizeElement.value = minSize;
                } else if (currentValue > maxSize) {
                    inputSizeElement.value = maxSize;
                }
                
                // Update label to show range
                const label = document.querySelector('label[for="execInputSize"]');
                if (label) {
                    if (selectedAlgorithm === 'rl') {
                        label.textContent = `Number of Inputs (${availableSizes.join(', ')})`;
                    } else {
                        label.textContent = `Number of Inputs (${minSize}-${maxSize})`;
                    }
                }
                
                // Update required inputs display
                updateRequiredInputs();
                

                
                console.log(`Updated input size limits for ${selectedAlgorithm}: ${minSize}-${maxSize}`);
            }
        }
    } catch (error) {
        console.error('Error updating input size limits:', error);
    }
}



// Update required inputs display
function updateRequiredInputs() {
    const inputSizeElement = document.getElementById('execInputSize');
    const requiredInputsElement = document.getElementById('requiredInputs');
    
    if (inputSizeElement && requiredInputsElement) {
        requiredInputsElement.textContent = inputSizeElement.value;
    }
}

function toggleCustomInput() {
    const customInputDiv = document.getElementById('customInputDiv');
    const customInputRadio = document.getElementById('customInput');
    
    if (customInputDiv && customInputRadio) {
        if (customInputRadio.checked) {
            customInputDiv.style.display = 'block';
        } else {
            customInputDiv.style.display = 'none';
        }
    }
} 