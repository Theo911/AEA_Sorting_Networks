# AI Integration Instructions for Sorting Networks Webapp

## Overview

This document provides guidance for integrating and enhancing multiple sorting network algorithms (Batcher's and RL-based) in the webapp. The application was originally built for Batcher's algorithm but now needs consistent integration of both algorithms throughout all features.

## Current Architecture

- **Frontend**: HTML/CSS/JS with Bootstrap for styling
- **Backend**: Flask serving visualization and analytics
- **Directory Structure**:
  - `/static/css/`: CSS styles
  - `/static/js/`: JavaScript functionality
  - `/templates/`: HTML templates

## Required Enhancements

### 1. Global Algorithm State Management

```javascript
// Example state management in main.js
// Store the currently selected algorithm globally
let currentAlgorithm = localStorage.getItem('selectedAlgorithm') || 'batcher';

// Function to update all UI elements based on algorithm selection
function updateUIForAlgorithm(algorithm) {
    currentAlgorithm = algorithm;
    localStorage.setItem('selectedAlgorithm', algorithm);
    
    // Update main title
    const algoName = algorithm === 'rl' ? 'RL Sorting Network' : "Batcher's Odd-Even Mergesort";
    document.getElementById('main-title').textContent = algoName;
    
    // Update other UI elements
    // ...
}
```

### 2. Consistent UI Controls

- Add algorithm selection controls to all tabs
- Ensure visual consistency
- Indicate feature availability per algorithm

### 3. Tab-Specific Adaptations

#### Network Visualization Tab
- Connect existing radio buttons to global state
- Propagate selection to other tabs

#### Execution Demo Tab
- Add algorithm selection
- Update available options based on algorithm selection
- Clearly indicate which algorithms support execution

#### Performance Analysis Tab
- Show comparative charts between algorithms
- Add filtering options to focus on specific algorithms

#### Algorithm Theory Tab
- Connect theory toggle buttons to global state
- Pre-select the correct theory based on the global state

### 4. Backend Considerations

- All API endpoints should accept an `algorithm` parameter
- Implementation for `/execute_network` should handle RL algorithm if available
- Return appropriate error messages for unsupported features

## Implementation Approach

1. **Start with State Management**: Implement the global algorithm state tracker first
2. **Update UI Components**: Modify existing UI elements to react to state changes
3. **Enhance Backend**: Add algorithm parameter support to all endpoints
4. **Add New Features**: Implement algorithm comparison features
5. **Testing**: Test all tabs with both algorithms

## Code Example for Tab Coordination

```javascript
// Listen for algorithm changes on any tab
document.querySelectorAll('input[name="algorithm"]').forEach(radio => {
    radio.addEventListener('change', function() {
        const selectedAlgorithm = this.value;
        updateUIForAlgorithm(selectedAlgorithm);
        
        // Update specific tab elements
        // ...
    });
});

// Initialize UI based on stored preference when page loads
document.addEventListener('DOMContentLoaded', function() {
    updateUIForAlgorithm(currentAlgorithm);
    
    // Select correct radio buttons
    document.querySelectorAll(`input[name="algorithm"][value="${currentAlgorithm}"]`)
        .forEach(radio => { radio.checked = true; });
});
```

## Notes

- The webapp was originally designed for Batcher's algorithm only
- The RL integration is partial and needs to be completed
- Execution demo currently only supports Batcher's algorithm
- The main title changes based on algorithm selection but needs to be consistent across tabs

## Future Enhancements

- Support for additional sorting network algorithms
- More detailed comparative analytics
- Interactive network building features
- Step-by-step algorithm explanation 