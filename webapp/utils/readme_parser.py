"""
README Performance Parser for Sorting Networks Webapp
Extracts authoritative RL performance data from README.md table
"""

import os
import re
import logging
from typing import Dict, Optional, Tuple

class ReadmePerformanceParser:
    """Parse README.md performance table for authoritative RL data"""
    
    def __init__(self, readme_path=None):
        if readme_path is None:
            # Default path relative to webapp directory
            webapp_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.readme_path = os.path.join(webapp_dir, "..", "RLSortingNetworks", "README.md")
        else:
            self.readme_path = readme_path
            
        self.performance_table = None
        self.table_data = {}
        
        # Initialize data extraction
        self._extract_performance_table()
    
    def _extract_performance_table(self):
        """Extract the performance summary table from README"""
        try:
            if not os.path.exists(self.readme_path):
                logging.warning(f"README file not found at {self.readme_path}")
                return False
                
            with open(self.readme_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
            # Look for the performance table section
            # The table starts after "### Performance Summary (Experimental Results)"
            table_start = content.find("### Performance Summary (Experimental Results)")
            if table_start == -1:
                logging.warning("Performance summary table not found in README")
                return False
            
            # Extract the table content
            table_section = content[table_start:]
            
            # Find the actual markdown table
            lines = table_section.split('\n')
            table_lines = []
            in_table = False
            
            for line in lines:
                if '|' in line and ('n' in line or 'max_steps' in line):
                    in_table = True
                    table_lines.append(line.strip())
                elif in_table and '|' in line:
                    table_lines.append(line.strip())
                elif in_table and '|' not in line:
                    break
            
            if table_lines:
                self._parse_table_lines(table_lines)
                return True
            else:
                logging.warning("Could not extract table lines from README")
                return False
                
        except Exception as e:
            logging.error(f"Error reading README file: {e}")
            return False
    
    def _parse_table_lines(self, table_lines):
        """Parse the extracted table lines into structured data"""
        if len(table_lines) < 3:  # Header, separator, data
            logging.warning("Insufficient table lines found")
            return
        
        # Parse header to identify column positions
        header = table_lines[0]
        columns = [col.strip() for col in header.split('|') if col.strip()]
        
        # Find relevant column indices
        col_indices = {}
        for i, col in enumerate(columns):
            col_lower = col.lower()
            if 'n' == col_lower or col_lower.startswith('`n`'):
                col_indices['n'] = i
            elif 'double dqn' in col_lower and 'best size' in col_lower:
                col_indices['double_best_size'] = i
            elif 'double dqn' in col_lower and 'best depth' in col_lower:
                col_indices['double_best_depth'] = i
            elif 'double dqn' in col_lower and 'pruned size' in col_lower:
                col_indices['double_pruned_size'] = i
            elif 'double dqn' in col_lower and 'pruned depth' in col_lower:
                col_indices['double_pruned_depth'] = i
            elif 'classic dqn' in col_lower and 'best size' in col_lower:
                col_indices['classic_best_size'] = i
            elif 'classic dqn' in col_lower and 'best depth' in col_lower:
                col_indices['classic_best_depth'] = i
            elif 'classic dqn' in col_lower and 'pruned size' in col_lower:
                col_indices['classic_pruned_size'] = i
            elif 'classic dqn' in col_lower and 'pruned depth' in col_lower:
                col_indices['classic_pruned_depth'] = i
        
        # Parse data rows (skip header and separator)
        for line in table_lines[2:]:
            if not line.strip() or line.startswith('|---'):
                continue
                
            cells = [cell.strip() for cell in line.split('|') if cell.strip()]
            if len(cells) < len(columns):
                continue
            
            try:
                # Extract n value
                n_str = cells[col_indices.get('n', 0)]
                n = int(n_str)
                
                # Initialize data structure for this n
                self.table_data[n] = {}
                
                # Extract data for each available column
                for col_key, col_idx in col_indices.items():
                    if col_key != 'n' and col_idx < len(cells):
                        cell_value = cells[col_idx]
                        
                        # Parse cell value (handle asterisks, empty cells, etc.)
                        parsed_value = self._parse_cell_value(cell_value)
                        self.table_data[n][col_key] = parsed_value
                        
            except (ValueError, IndexError) as e:
                logging.warning(f"Error parsing table row: {line}, error: {e}")
                continue
        
        logging.info(f"Extracted performance data for n values: {list(self.table_data.keys())}")
    
    def _parse_cell_value(self, cell_value):
        """Parse individual cell value, handling special cases"""
        if not cell_value or cell_value in ['*?*', '-', 'N/A']:
            return None
            
        # Remove asterisks and other markdown formatting
        cleaned = cell_value.replace('*', '').strip()
        
        if not cleaned or cleaned in ['?', '-', 'N/A']:
            return None
            
        try:
            return int(cleaned)
        except ValueError:
            return None
    
    def extract_double_dqn_data(self):
        """Get Double DQN results (both raw and pruned)"""
        double_data = {
            'raw': {'size': {}, 'depth': {}},
            'pruned': {'size': {}, 'depth': {}}
        }
        
        for n, data in self.table_data.items():
            # Raw data (best found during training)
            if 'double_best_size' in data and data['double_best_size'] is not None:
                double_data['raw']['size'][n] = data['double_best_size']
            if 'double_best_depth' in data and data['double_best_depth'] is not None:
                double_data['raw']['depth'][n] = data['double_best_depth']
                
            # Pruned data (optimized)
            if 'double_pruned_size' in data and data['double_pruned_size'] is not None:
                double_data['pruned']['size'][n] = data['double_pruned_size']
            if 'double_pruned_depth' in data and data['double_pruned_depth'] is not None:
                double_data['pruned']['depth'][n] = data['double_pruned_depth']
        
        return double_data
    
    def extract_classic_dqn_data(self):
        """Get Classic DQN results (both raw and pruned)"""
        classic_data = {
            'raw': {'size': {}, 'depth': {}},
            'pruned': {'size': {}, 'depth': {}}
        }
        
        for n, data in self.table_data.items():
            # Raw data (best found during training)
            if 'classic_best_size' in data and data['classic_best_size'] is not None:
                classic_data['raw']['size'][n] = data['classic_best_size']
            if 'classic_best_depth' in data and data['classic_best_depth'] is not None:
                classic_data['raw']['depth'][n] = data['classic_best_depth']
                
            # Pruned data (optimized)
            if 'classic_pruned_size' in data and data['classic_pruned_size'] is not None:
                classic_data['pruned']['size'][n] = data['classic_pruned_size']
            if 'classic_pruned_depth' in data and data['classic_pruned_depth'] is not None:
                classic_data['pruned']['depth'][n] = data['classic_pruned_depth']
        
        return classic_data
    
    def get_algorithm_comparison_data(self):
        """Return structured comparison data for all RL algorithms"""
        return {
            'rl_classic_raw': self.extract_classic_dqn_data()['raw'],
            'rl_classic_pruned': self.extract_classic_dqn_data()['pruned'],
            'rl_double_raw': self.extract_double_dqn_data()['raw'],
            'rl_double_pruned': self.extract_double_dqn_data()['pruned'],
            'data_source': 'README.md performance table',
            'available_n_values': list(self.table_data.keys())
        }
    
    def get_pruning_impact_analysis(self):
        """Calculate the impact of pruning for both Classic and Double DQN"""
        classic_data = self.extract_classic_dqn_data()
        double_data = self.extract_double_dqn_data()
        
        impact_analysis = {
            'classic_dqn': {},
            'double_dqn': {}
        }
        
        # Analyze Classic DQN pruning impact
        for n in self.table_data.keys():
            if (n in classic_data['raw']['size'] and classic_data['raw']['size'][n] is not None and
                n in classic_data['pruned']['size'] and classic_data['pruned']['size'][n] is not None):
                
                raw_size = classic_data['raw']['size'][n]
                pruned_size = classic_data['pruned']['size'][n]
                reduction = raw_size - pruned_size
                reduction_pct = (reduction / raw_size) * 100 if raw_size > 0 else 0
                
                impact_analysis['classic_dqn'][n] = {
                    'size_reduction': reduction,
                    'size_reduction_percent': reduction_pct,
                    'raw_size': raw_size,
                    'pruned_size': pruned_size
                }
        
        # Analyze Double DQN pruning impact
        for n in self.table_data.keys():
            if (n in double_data['raw']['size'] and double_data['raw']['size'][n] is not None and
                n in double_data['pruned']['size'] and double_data['pruned']['size'][n] is not None):
                
                raw_size = double_data['raw']['size'][n]
                pruned_size = double_data['pruned']['size'][n]
                reduction = raw_size - pruned_size
                reduction_pct = (reduction / raw_size) * 100 if raw_size > 0 else 0
                
                impact_analysis['double_dqn'][n] = {
                    'size_reduction': reduction,
                    'size_reduction_percent': reduction_pct,
                    'raw_size': raw_size,
                    'pruned_size': pruned_size
                }
        
        return impact_analysis
    
    def is_data_available(self):
        """Check if README data was successfully extracted"""
        return len(self.table_data) > 0
    
    def get_available_n_values(self):
        """Get list of n values available in the README table"""
        return sorted(list(self.table_data.keys())) 