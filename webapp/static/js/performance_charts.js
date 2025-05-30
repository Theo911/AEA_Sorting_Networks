/**
 * Performance Charts Manager for Sorting Networks Webapp
 * Handles Chart.js visualizations for 5-algorithm comparison dashboard
 */

class PerformanceChartManager {
    constructor() {
        this.charts = {};
        this.data = null;
        this.algorithmStatus = null;
        this.colors = {
            optimal: {
                line: '#FFD700',
                fill: 'rgba(255, 215, 0, 0.1)'
            },
            batcher: {
                line: '#007BFF', 
                fill: 'rgba(0, 123, 255, 0.1)'
            },
            rl: {
                line: '#DC3545',
                fill: 'rgba(220, 53, 69, 0.1)'
            },
            batcher_improved: {
                line: '#28A745',
                fill: 'rgba(40, 167, 69, 0.1)',
                dash: [5, 5]
            }
        };
    }

    async init() {
        try {
            await this.loadData();
            await this.loadAlgorithmStatus();
            this.createAllCharts();
            this.updateStatusIndicator();
            this.updateSummaryTable();
            this.setupEventHandlers();
        } catch (error) {
            console.error('Error initializing performance charts:', error);
            this.showErrorMessage('Failed to load performance data');
        }
    }

    async loadData() {
        try {
            const response = await fetch('/api/performance_data');
            const result = await response.json();
            
            if (result.success) {
                this.data = result.data;
                this.algorithmStatus = result.algorithm_status;
            } else {
                throw new Error(result.error || 'Failed to load performance data');
            }
        } catch (error) {
            console.error('Error loading performance data:', error);
            throw error;
        }
    }

    async loadAlgorithmStatus() {
        try {
            const response = await fetch('/api/algorithm_status');
            const result = await response.json();
            
            if (result.success) {
                this.algorithmStatus = result.algorithms;
            }
        } catch (error) {
            console.error('Error loading algorithm status:', error);
        }
    }

    createAllCharts() {
        this.createSizeComparisonChart();
        this.createDepthComparisonChart();
        this.createEfficiencyChart();
        this.createExecutionAnalysisChart();
    }

    createSizeComparisonChart() {
        const ctx = document.getElementById('sizeComparisonChart').getContext('2d');
        
        this.charts.sizeComparison = new Chart(ctx, {
            type: 'line',
            data: {
                labels: this.data.n_range,
                datasets: this.buildDatasets('size')
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Number of Comparators'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Input Size (n)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        callbacks: {
                            title: function(context) {
                                return `n = ${context[0].label}`;
                            },
                            label: function(context) {
                                const algorithm = context.dataset.label;
                                const value = context.parsed.y;
                                return value !== null ? `${algorithm}: ${value} comparators` : `${algorithm}: N/A`;
                            }
                        },
                        itemSort: function(a, b) {
                            if (a.parsed.y === null && b.parsed.y === null) return 0;
                            if (a.parsed.y === null) return 1;
                            if (b.parsed.y === null) return -1;
                            return b.parsed.y - a.parsed.y;
                        }
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
        const ctx = document.getElementById('depthComparisonChart').getContext('2d');
        
        this.charts.depthComparison = new Chart(ctx, {
            type: 'line',
            data: {
                labels: this.data.n_range,
                datasets: this.buildDatasets('depth')
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Network Depth'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Input Size (n)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        callbacks: {
                            title: function(context) {
                                return `n = ${context[0].label}`;
                            },
                            label: function(context) {
                                const algorithm = context.dataset.label;
                                const value = context.parsed.y;
                                return value !== null ? `${algorithm}: ${value} layers` : `${algorithm}: N/A`;
                            }
                        },
                        itemSort: function(a, b) {
                            if (a.parsed.y === null && b.parsed.y === null) return 0;
                            if (a.parsed.y === null) return 1;
                            if (b.parsed.y === null) return -1;
                            return b.parsed.y - a.parsed.y;
                        }
                    }
                }
            }
        });
    }

    createEfficiencyChart() {
        const ctx = document.getElementById('efficiencyChart').getContext('2d');
        
        // Calculate efficiency ratios
        const batcherEfficiency = this.calculateEfficiency('batcher', 'size');
        const rlEfficiency = this.calculateEfficiency('rl', 'size');
        
        this.charts.efficiency = new Chart(ctx, {
            type: 'line',
            data: {
                labels: this.data.n_range,
                datasets: [
                    {
                        label: 'Batcher Efficiency',
                        data: batcherEfficiency,
                        borderColor: this.colors.batcher.line,
                        backgroundColor: this.colors.batcher.fill,
                        fill: false,
                        tension: 0.1
                    },
                    {
                        label: 'RL Efficiency',
                        data: rlEfficiency,
                        borderColor: this.colors.rl.line,
                        backgroundColor: this.colors.rl.fill,
                        fill: false,
                        tension: 0.1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1,
                        title: {
                            display: true,
                            text: 'Efficiency (Optimal/Actual)'
                        },
                        ticks: {
                            callback: function(value) {
                                return (value * 100).toFixed(0) + '%';
                            }
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Input Size (n)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const algorithm = context.dataset.label;
                                const value = context.parsed.y;
                                return value !== null ? `${algorithm}: ${(value * 100).toFixed(1)}%` : `${algorithm}: N/A`;
                            }
                        }
                    }
                }
            }
        });
    }

    createExecutionAnalysisChart() {
        const ctx = document.getElementById('executionAnalysisChart').getContext('2d');
        
        // For now, create a placeholder chart - this will be updated with actual execution data
        this.charts.executionAnalysis = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Batcher', 'RL', 'Batcher Improved'],
                datasets: [{
                    label: 'Algorithm Usage',
                    data: [100, 50, 0], // Placeholder data
                    backgroundColor: [
                        this.colors.batcher.line,
                        this.colors.rl.line, 
                        this.colors.batcher_improved.line
                    ],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        position: 'bottom'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.label;
                                const value = context.parsed;
                                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                const percentage = ((value / total) * 100).toFixed(1);
                                return `${label}: ${value} executions (${percentage}%)`;
                            }
                        }
                    }
                }
            }
        });
    }

    buildDatasets(metric) {
        const datasets = [];
        
        // Optimal dataset
        if (this.data.optimal && this.data.optimal[metric]) {
            datasets.push({
                label: 'Optimal',
                data: this.data.optimal[metric],
                borderColor: this.colors.optimal.line,
                backgroundColor: this.colors.optimal.fill,
                fill: false,
                tension: 0.1,
                pointStyle: 'triangle',
                pointRadius: 6
            });
        }

        // Batcher dataset
        if (this.data.batcher && this.data.batcher[metric]) {
            datasets.push({
                label: 'Batcher',
                data: this.data.batcher[metric],
                borderColor: this.colors.batcher.line,
                backgroundColor: this.colors.batcher.fill,
                fill: false,
                tension: 0.1
            });
        }

        // RL dataset
        if (this.data.rl && this.data.rl[metric]) {
            datasets.push({
                label: 'RL',
                data: this.data.rl[metric],
                borderColor: this.colors.rl.line,
                backgroundColor: this.colors.rl.fill,
                fill: false,
                tension: 0.1
            });
        }

        // Improved algorithms (if available)
        if (this.data.batcher_improved && this.data.batcher_improved[metric]) {
            datasets.push({
                label: 'Batcher Improved',
                data: this.data.batcher_improved[metric],
                borderColor: this.colors.batcher_improved.line,
                backgroundColor: this.colors.batcher_improved.fill,
                borderDash: this.colors.batcher_improved.dash,
                fill: false,
                tension: 0.1
            });
        }

        return datasets;
    }

    calculateEfficiency(algorithm, metric) {
        if (!this.data[algorithm] || !this.data[algorithm][metric] || 
            !this.data.optimal || !this.data.optimal[metric]) {
            return this.data.n_range.map(() => null);
        }

        return this.data.n_range.map((n, i) => {
            const optimal = this.data.optimal[metric][i];
            const actual = this.data[algorithm][metric][i];
            
            if (optimal && actual && optimal > 0 && actual > 0) {
                return optimal / actual;
            }
            return null;
        });
    }

    updateStatusIndicator() {
        const indicator = document.getElementById('algorithmStatusIndicator');
        if (!indicator || !this.algorithmStatus) return;

        const statusItems = Object.entries(this.algorithmStatus).map(([key, info]) => {
            const badge = info.available ? 'bg-success' : 'bg-secondary';
            const text = info.available ? 'Available' : 'Coming Soon';
            return `<span class="badge ${badge} me-2">${info.description}: ${text}</span>`;
        });

        indicator.innerHTML = statusItems.join('');
    }

    updateSummaryTable() {
        const tbody = document.querySelector('#performanceSummaryTable tbody');
        if (!tbody || !this.data) return;

        tbody.innerHTML = '';

        this.data.n_range.forEach((n, i) => {
            const row = document.createElement('tr');
            
            // Helper function to format values
            const formatValue = (data, index) => {
                return data && data[index] !== null && data[index] !== undefined 
                    ? data[index] 
                    : '<span class="text-muted">N/A</span>';
            };

            // Calculate efficiencies
            const batcherEff = this.calculateEfficiency('batcher', 'size')[i];
            const rlEff = this.calculateEfficiency('rl', 'size')[i];
            
            row.innerHTML = `
                <td><strong>${n}</strong></td>
                <td>${formatValue(this.data.optimal.size, i)}</td>
                <td>${formatValue(this.data.batcher.size, i)}</td>
                <td>${formatValue(this.data.rl.size, i)}</td>
                <td>${formatValue(this.data.optimal.depth, i)}</td>
                <td>${formatValue(this.data.batcher.depth, i)}</td>
                <td>${formatValue(this.data.rl.depth, i)}</td>
                <td>${batcherEff ? (batcherEff * 100).toFixed(1) + '%' : '<span class="text-muted">N/A</span>'}</td>
                <td>${rlEff ? (rlEff * 100).toFixed(1) + '%' : '<span class="text-muted">N/A</span>'}</td>
            `;
            
            tbody.appendChild(row);
        });
    }

    setupEventHandlers() {
        // Refresh data button
        const refreshBtn = document.getElementById('refreshDataBtn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', async () => {
                refreshBtn.disabled = true;
                refreshBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status"></span> Refreshing...';
                
                try {
                    await this.loadData();
                    this.updateAllCharts();
                    this.updateSummaryTable();
                    this.updateStatusIndicator();
                } catch (error) {
                    console.error('Error refreshing data:', error);
                    this.showErrorMessage('Failed to refresh data');
                } finally {
                    refreshBtn.disabled = false;
                    refreshBtn.innerHTML = '<i class="bi bi-arrow-clockwise"></i> Refresh Data';
                }
            });
        }

        // Export data button
        const exportBtn = document.getElementById('exportDataBtn');
        if (exportBtn) {
            exportBtn.addEventListener('click', () => {
                this.exportData();
            });
        }
    }

    updateAllCharts() {
        Object.values(this.charts).forEach(chart => {
            if (chart) {
                chart.destroy();
            }
        });
        this.charts = {};
        this.createAllCharts();
    }

    exportData() {
        if (!this.data) {
            this.showErrorMessage('No data available to export');
            return;
        }

        const csvContent = this.generateCSV();
        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `sorting_networks_performance_${new Date().toISOString().split('T')[0]}.csv`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
    }

    generateCSV() {
        const headers = ['n', 'Optimal_Size', 'Batcher_Size', 'RL_Size', 'Optimal_Depth', 'Batcher_Depth', 'RL_Depth'];
        const rows = [headers.join(',')];

        this.data.n_range.forEach((n, i) => {
            const row = [
                n,
                this.data.optimal.size[i] || '',
                this.data.batcher.size[i] || '',
                this.data.rl.size[i] || '',
                this.data.optimal.depth[i] || '',
                this.data.batcher.depth[i] || '',
                this.data.rl.depth[i] || ''
            ];
            rows.push(row.join(','));
        });

        return rows.join('\n');
    }

    showErrorMessage(message) {
        // Create a simple error message display
        const indicator = document.getElementById('algorithmStatusIndicator');
        if (indicator) {
            indicator.innerHTML = `<span class="badge bg-danger">Error: ${message}</span>`;
        }
    }
}

// Global instance
let performanceChartManager = null;

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    // Only initialize if we're using the old performance tab layout (not the enhanced one)
    // Check for enhanced charts first - if they exist, don't initialize the old manager
    if (document.getElementById('enhancedSizeComparisonChart')) {
        console.log('Enhanced performance charts detected, skipping old chart manager');
        return;
    }
    
    // Only initialize if we're on the old performance tab layout
    if (document.getElementById('sizeComparisonChart')) {
        performanceChartManager = new PerformanceChartManager();
        
        // Initialize when performance tab is shown
        const performanceTab = document.getElementById('performance-tab');
        if (performanceTab) {
            performanceTab.addEventListener('shown.bs.tab', function() {
                if (performanceChartManager && !performanceChartManager.data) {
                    performanceChartManager.init();
                }
            });
            
            // If performance tab is already active, initialize immediately
            if (performanceTab.classList.contains('active')) {
                performanceChartManager.init();
            }
        }
    }
}); 