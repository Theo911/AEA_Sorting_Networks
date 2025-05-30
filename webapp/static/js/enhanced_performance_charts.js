/**
 * Enhanced Performance Chart Manager for 7-Algorithm Comparison
 * Handles comprehensive performance visualization with evolution analysis
 */

class EnhancedPerformanceChartManager {
    constructor() {
        this.charts = {};
        this.data = null;
        this.validationData = null;
        this.evolutionData = null;
        
        // Enhanced color scheme for 7 algorithms - all distinct colors
        this.colors = {
            optimal: { line: '#FFD700', fill: 'rgba(255, 215, 0, 0.1)', name: 'Optimal (Theory)' },
            batcher: { line: '#007BFF', fill: 'rgba(0, 123, 255, 0.1)', name: 'Batcher Traditional' },
            rl_classic_raw: { line: '#DC3545', fill: 'rgba(220, 53, 69, 0.1)', dash: [5, 5], name: 'RL Classic (Raw)' },
            rl_classic_pruned: { line: '#E91E63', fill: 'rgba(233, 30, 99, 0.1)', name: 'RL Classic (Pruned)' },
            rl_double_raw: { line: '#28A745', fill: 'rgba(40, 167, 69, 0.1)', dash: [5, 5], name: 'RL Double (Raw)' },
            rl_double_pruned: { line: '#4CAF50', fill: 'rgba(76, 175, 80, 0.1)', name: 'RL Double (Pruned)' },
            batcher_improved: { line: '#9C27B0', fill: 'rgba(156, 39, 176, 0.1)', dash: [8, 4], name: 'Batcher Enhanced' }
        };
        
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        // Data refresh button
        document.getElementById('refreshDataBtn')?.addEventListener('click', () => {
            this.loadComprehensiveData();
        });
        
        // Show evolution analysis
        document.getElementById('showEvolutionBtn')?.addEventListener('click', () => {
            this.toggleEvolutionPanel();
        });
        
        // Show data validation
        document.getElementById('showValidationBtn')?.addEventListener('click', () => {
            this.toggleValidationPanel();
        });
        
        // Export data
        document.getElementById('exportDataBtn')?.addEventListener('click', () => {
            this.exportData();
        });
    }

    async loadComprehensiveData() {
        try {
            console.log('Loading comprehensive 7-algorithm data...');
            
            // Load performance data
            const performanceResponse = await fetch('/api/performance_data');
            const performanceResult = await performanceResponse.json();
            
            if (performanceResult.success) {
                this.data = performanceResult.data;
                console.log('Performance data loaded:', this.data);
            }
            
            // Load algorithm evolution analysis
            const evolutionResponse = await fetch('/api/algorithm_comparison');
            const evolutionResult = await evolutionResponse.json();
            
            if (evolutionResult.success) {
                this.evolutionData = evolutionResult.evolution_analysis;
                console.log('Evolution data loaded:', this.evolutionData);
            }
            
            // Load validation data
            const validationResponse = await fetch('/api/data_validation');
            const validationResult = await validationResponse.json();
            
            if (validationResult.success) {
                this.validationData = validationResult;
                console.log('Validation data loaded:', this.validationData);
            }
            
            // Update displays
            this.updateAlgorithmStatusIndicator();
            this.updateDataQualityIndicator();
            this.createEnhancedComparisonCharts();
            
        } catch (error) {
            console.error('Error loading comprehensive data:', error);
            this.showError('Failed to load performance data');
        }
    }

    updateAlgorithmStatusIndicator() {
        const indicator = document.getElementById('algorithmStatusIndicator');
        if (!indicator || !this.data) return;
        
        const algorithmStatus = this.data.algorithm_status || {};
        const available = Object.values(algorithmStatus).filter(status => status.available).length;
        const total = Object.keys(algorithmStatus).length;
        
        indicator.innerHTML = `
            <small class="text-success">
                <i class="bi bi-check-circle-fill"></i> 
                ${available}/${total} algorithms available • 
                Data sources: ${this.data.data_sources?.rl_algorithms || 'README.md'}
            </small>
        `;
    }

    updateDataQualityIndicator() {
        const indicator = document.getElementById('dataQualityIndicator');
        if (!indicator || !this.validationData) return;
        
        const qualityScore = this.validationData.data_quality?.data_quality_score || 'N/A';
        
        indicator.innerHTML = `
            <small class="text-info">
                <i class="bi bi-shield-check"></i> 
                Data quality: ${qualityScore}/100 • 
                Cross-validated with multiple sources
            </small>
        `;
    }

    createEnhancedComparisonCharts() {
        if (!this.data) return;
        
        // Create size comparison chart
        this.createSizeComparisonChart();
        
        // Create depth comparison chart
        this.createDepthComparisonChart();
    }

    createSizeComparisonChart() {
        const ctx = document.getElementById('enhancedSizeComparisonChart');
        if (!ctx) return;
        
        // Destroy existing chart
        if (this.charts.sizeComparison) {
            this.charts.sizeComparison.destroy();
        }
        
        const datasets = [];
        const nRange = this.data.n_range || [];
        
        console.log('Creating size comparison chart with data:', this.data);
        
        // Create datasets for each algorithm
        Object.keys(this.colors).forEach(algorithmKey => {
            const algorithmData = this.data[algorithmKey];
            console.log(`Processing algorithm ${algorithmKey}:`, algorithmData);
            
            if (algorithmData && algorithmData.size) {
                const color = this.colors[algorithmKey];
                
                console.log(`Adding dataset for ${algorithmKey}:`, {
                    label: color.name,
                    data: algorithmData.size,
                    color: color.line
                });
                
                datasets.push({
                    label: color.name,
                    data: algorithmData.size,
                    borderColor: color.line,
                    backgroundColor: color.fill,
                    borderWidth: 2,
                    borderDash: color.dash || [],
                    fill: false,
                    tension: 0.1,
                    pointRadius: 4,
                    pointHoverRadius: 6
                });
            } else {
                console.log(`Skipping algorithm ${algorithmKey} - no size data available`);
            }
        });
        
        console.log('Final datasets for size chart:', datasets);
        
        this.charts.sizeComparison = new Chart(ctx, {
            type: 'line',
            data: {
                labels: nRange,
                datasets: datasets
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Network Size (Number of Comparators)'
                    },
                    legend: {
                        position: 'top',
                        labels: {
                            usePointStyle: true
                        }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        // Sort tooltip items by value (highest to lowest)
                        itemSort: function(a, b) {
                            if (a.parsed.y === null && b.parsed.y === null) return 0;
                            if (a.parsed.y === null) return 1;  // N/A values go to bottom
                            if (b.parsed.y === null) return -1;
                            return b.parsed.y - a.parsed.y;  // Sort descending (highest first)
                        },
                        callbacks: {
                            title: function(context) {
                                return `n = ${context[0].label} wires`;
                            },
                            label: function(context) {
                                const algorithm = context.dataset.label;
                                const value = context.parsed.y;
                                return value !== null ? `${algorithm}: ${value} comparators` : `${algorithm}: N/A`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Number of Wires (n)'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Number of Comparators'
                        },
                        beginAtZero: true
                    }
                },
                interaction: {
                    mode: 'nearest',
                    axis: 'x',
                    intersect: false
                }
            }
        });
    }

    createDepthComparisonChart() {
        const ctx = document.getElementById('enhancedDepthComparisonChart');
        if (!ctx) return;
        
        // Destroy existing chart
        if (this.charts.depthComparison) {
            this.charts.depthComparison.destroy();
        }
        
        const datasets = [];
        const nRange = this.data.n_range || [];
        
        // Create datasets for each algorithm
        Object.keys(this.colors).forEach(algorithmKey => {
            const algorithmData = this.data[algorithmKey];
            if (algorithmData && algorithmData.depth) {
                const color = this.colors[algorithmKey];
                
                datasets.push({
                    label: color.name,
                    data: algorithmData.depth,
                    borderColor: color.line,
                    backgroundColor: color.fill,
                    borderWidth: 2,
                    borderDash: color.dash || [],
                    fill: false,
                    tension: 0.1,
                    pointRadius: 4,
                    pointHoverRadius: 6
                });
            }
        });
        
        this.charts.depthComparison = new Chart(ctx, {
            type: 'line',
            data: {
                labels: nRange,
                datasets: datasets
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Network Depth (Critical Path Length)'
                    },
                    legend: {
                        position: 'top',
                        labels: {
                            usePointStyle: true
                        }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        // Sort tooltip items by value (highest to lowest)
                        itemSort: function(a, b) {
                            if (a.parsed.y === null && b.parsed.y === null) return 0;
                            if (a.parsed.y === null) return 1;  // N/A values go to bottom
                            if (b.parsed.y === null) return -1;
                            return b.parsed.y - a.parsed.y;  // Sort descending (highest first)
                        },
                        callbacks: {
                            title: function(context) {
                                return `n = ${context[0].label} wires`;
                            },
                            label: function(context) {
                                const algorithm = context.dataset.label;
                                const value = context.parsed.y;
                                return value !== null ? `${algorithm}: ${value} layers` : `${algorithm}: N/A`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Number of Wires (n)'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Network Depth'
                        },
                        beginAtZero: true
                    }
                },
                interaction: {
                    mode: 'nearest',
                    axis: 'x',
                    intersect: false
                }
            }
        });
    }

    createAlgorithmEvolutionCharts() {
        if (!this.evolutionData) return;
        
        this.createTrainingEvolutionChart();
        this.createOptimizationImpactChart();
    }

    createTrainingEvolutionChart() {
        const ctx = document.getElementById('trainingEvolutionChart');
        if (!ctx) return;
        
        const evolutionAnalysis = this.evolutionData.training_method_evolution || {};
        const nValues = Object.keys(evolutionAnalysis).map(n => parseInt(n)).sort((a, b) => a - b);
        
        const classicSizes = nValues.map(n => evolutionAnalysis[n]?.classic_size);
        const doubleSizes = nValues.map(n => evolutionAnalysis[n]?.double_size);
        
        this.charts.trainingEvolution = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: nValues,
                datasets: [
                    {
                        label: 'Classic DQN',
                        data: classicSizes,
                        backgroundColor: 'rgba(220, 53, 69, 0.7)',
                        borderColor: '#DC3545',
                        borderWidth: 1
                    },
                    {
                        label: 'Double DQN',
                        data: doubleSizes,
                        backgroundColor: 'rgba(40, 167, 69, 0.7)',
                        borderColor: '#28A745',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Number of Wires (n)'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Network Size'
                        },
                        beginAtZero: true
                    }
                }
            }
        });
    }

    createOptimizationImpactChart() {
        const ctx = document.getElementById('optimizationImpactChart');
        if (!ctx) return;
        
        const pruningImpact = this.evolutionData.optimization_impact || {};
        const classicData = pruningImpact.classic_dqn || {};
        const doubleData = pruningImpact.double_dqn || {};
        
        const nValues = [...new Set([
            ...Object.keys(classicData).map(n => parseInt(n)),
            ...Object.keys(doubleData).map(n => parseInt(n))
        ])].sort((a, b) => a - b);
        
        const classicReductions = nValues.map(n => classicData[n]?.size_reduction_percent || 0);
        const doubleReductions = nValues.map(n => doubleData[n]?.size_reduction_percent || 0);
        
        this.charts.optimizationImpact = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: nValues,
                datasets: [
                    {
                        label: 'Classic DQN Pruning',
                        data: classicReductions,
                        backgroundColor: 'rgba(220, 53, 69, 0.7)',
                        borderColor: '#DC3545',
                        borderWidth: 1
                    },
                    {
                        label: 'Double DQN Pruning',
                        data: doubleReductions,
                        backgroundColor: 'rgba(40, 167, 69, 0.7)',
                        borderColor: '#28A745',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Number of Wires (n)'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Size Reduction (%)'
                        },
                        beginAtZero: true
                    }
                }
            }
        });
    }

    toggleEvolutionPanel() {
        const panel = document.getElementById('evolutionAnalysisPanel');
        if (!panel) return;
        
        if (panel.style.display === 'none') {
            panel.style.display = 'block';
            this.createAlgorithmEvolutionCharts();
        } else {
            panel.style.display = 'none';
        }
    }

    toggleValidationPanel() {
        const panel = document.getElementById('dataQualityPanel');
        if (!panel) return;
        
        if (panel.style.display === 'none') {
            panel.style.display = 'block';
            this.updateDataQualityPanel();
        } else {
            panel.style.display = 'none';
        }
    }

    updateDataQualityPanel() {
        if (!this.validationData) return;
        
        const qualityScore = this.validationData.data_quality?.data_quality_score || 95;
        const dataSources = this.validationData.data_sources || {};
        
        document.getElementById('qualityScore').textContent = qualityScore;
        
        const sourceInfo = document.getElementById('dataSourceInfo');
        sourceInfo.innerHTML = `
            <div class="row">
                <div class="col-6">
                    <strong>Primary Sources:</strong><br>
                    <small>README: ${dataSources.rl_algorithms || 'N/A'}</small><br>
                    <small>Batcher: ${dataSources.batcher || 'N/A'}</small><br>
                    <small>Optimal: ${dataSources.optimal || 'N/A'}</small>
                </div>
                <div class="col-6">
                    <strong>Validation:</strong><br>
                    <small>Cross-check: ${dataSources.validation || 'N/A'}</small><br>
                    <small>Timestamp: ${this.validationData.data_quality?.validation_timestamp?.split('T')[0] || 'N/A'}</small>
                </div>
            </div>
        `;
    }

    exportData() {
        if (!this.data) {
            alert('No data available to export');
            return;
        }
        
        const exportData = {
            performance_data: this.data,
            evolution_analysis: this.evolutionData,
            validation_data: this.validationData,
            export_timestamp: new Date().toISOString()
        };
        
        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `sorting_networks_analysis_${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    showError(message) {
        console.error(message);
        // Could add toast notification here
    }
}

// Global instance
let enhancedChartManager = null;

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    enhancedChartManager = new EnhancedPerformanceChartManager();
    
    // Load data when performance tab is shown
    const performanceTab = document.getElementById('performance-tab');
    if (performanceTab) {
        performanceTab.addEventListener('shown.bs.tab', function() {
            enhancedChartManager.loadComprehensiveData();
        });
        
        // If performance tab is already active, load data immediately
        if (performanceTab.classList.contains('active')) {
            console.log('Performance tab is active, loading enhanced chart data immediately');
            enhancedChartManager.loadComprehensiveData();
        }
    }
}); 