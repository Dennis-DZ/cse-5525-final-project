import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
import statistics
import os

# Configuration for plots to match report style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("muted")

class DatasetAnalyzer:
    def __init__(self, name, file_path):
        self.name = name
        self.file_path = file_path
        self.examples = []
        self.stats = {}
        self.load_data()

    def load_data(self):
        print(f"Loading {self.name} from {self.file_path}")
        try:
            with open(self.file_path, 'r') as f:
                for line in f:
                    try:
                        ex = json.loads(line.strip())
                        if 'completion' in ex:
                            ex['parsed_json'] = json.loads(ex['completion'])
                            self.examples.append(ex)
                    except json.JSONDecodeError:
                        continue
            print(f"Managed to find {len(self.examples)} valid examples.")
        except FileNotFoundError:
            print(f" can't find the file: {self.file_path}.   ")

    def analyze(self):
        if not self.examples:
            return

        # 1. Overview Stats (Table 1)
        prompt_lengths = [len(ex['prompt'].split()) for ex in self.examples]
        json_lengths = [len(ex['completion']) for ex in self.examples]
        
        self.stats['overview'] = {
            'count': len(self.examples),
            'prompt_len_mean': statistics.mean(prompt_lengths) if prompt_lengths else 0,
            'prompt_len_min': min(prompt_lengths) if prompt_lengths else 0,
            'prompt_len_max': max(prompt_lengths) if prompt_lengths else 0,
            'json_len_mean': statistics.mean(json_lengths) if json_lengths else 0,
            'json_len_min': min(json_lengths) if json_lengths else 0,
            'json_len_max': max(json_lengths) if json_lengths else 0,
        }

        # 2. BC Types (Table 2)
        bc_counts = Counter()
        for ex in self.examples:
            for bc in ex['parsed_json'].get('boundary_conditions', []):
                bc_counts[bc.get('type', 'unknown')] += 1
        self.stats['bc_types'] = bc_counts

        # 3. Load Specs (Table 3)
        load_counts = Counter()
        load_magnitudes = defaultdict(list)
        for ex in self.examples:
            for load in ex['parsed_json'].get('loads', []):
                l_type = load.get('type', 'unknown')
                load_counts[l_type] += 1
                if 'magnitude' in load:
                    load_magnitudes[l_type].append(load['magnitude'])
        
        self.stats['load_types'] = load_counts
        self.stats['load_magnitudes'] = load_magnitudes

        # 4. Geometries (Figure 3)
        geo_counts = Counter()
        hollow_count = 0
        for ex in self.examples:
            geo = ex['parsed_json'].get('geometry', {})
            geo_counts[geo.get('type', 'unknown')] += 1
            # Check for hollow
            dims = geo.get('dimensions', {})
            if 'inner_radius' in dims and dims['inner_radius'] > 0:
                hollow_count += 1
        self.stats['geometries'] = geo_counts
        self.stats['hollow_count'] = hollow_count

        # 5. Materials (Figure 4)
        mat_counts = Counter()
        for ex in self.examples:
            mat = ex['parsed_json'].get('material', {})
            mat_counts[mat.get('type', 'unknown')] += 1
        self.stats['materials'] = mat_counts

        # 6. Location Descriptors (Figure 5)
        loc_counts = Counter()
        for ex in self.examples:
            # Check BCs
            for bc in ex['parsed_json'].get('boundary_conditions', []):
                loc = bc.get('location', {})
                if 'value' in loc: loc_counts[loc['value']] += 1
            # Check Loads
            for load in ex['parsed_json'].get('loads', []):
                loc = load.get('location', {})
                if 'value' in loc: loc_counts[loc['value']] += 1
        self.stats['locations'] = loc_counts

def create_comparison_report(original, new, output_file="comparison_report.txt"):
    with open(output_file, 'w') as f:
        def write(text):
            f.write(text + "\n")
            print(text)

        write("="*60)
        write(" FEA DATASET COMPARISON THINGY")
        write("="*60)
        
        # Table 1: Overview
        write("\n[Table 1] Dataset Statistics")
        write(f"{'Metric':<25} | {'Original':<15} | {'New (Complex)':<15} | {'Combined':<15}")
        write("-" * 76)
        
        metrics = [
            ('Total Examples', 'count', '{:d}'),
            ('Prompt Len (Mean)', 'prompt_len_mean', '{:.1f}'),
            ('Prompt Range', lambda s: f"{s['prompt_len_min']}-{s['prompt_len_max']}", '{}'),
            ('JSON Size (Mean)', 'json_len_mean', '{:.0f}'),
            ('JSON Range', lambda s: f"{s['json_len_min']}-{s['json_len_max']}", '{}'),
        ]
        
        for label, key, fmt in metrics:
            if callable(key):
                v1 = key(original.stats['overview'])
                v2 = key(new.stats['overview'])
                # Combined approximation
                if label == 'Total Examples':
                    v3 = original.stats['overview']['count'] + new.stats['overview']['count']
                else:
                    v3 = "N/A" 
            else:
                v1 = original.stats['overview'][key]
                v2 = new.stats['overview'][key]
                if key == 'count':
                     v3 = v1 + v2
                elif 'mean' in key:
                     # Weighted average
                     c1 = original.stats['overview']['count']
                     c2 = new.stats['overview']['count']
                     v3 = (v1*c1 + v2*c2) / (c1+c2) if (c1+c2) > 0 else 0
                else:
                    v3 = "N/A"

            write(f"{label:<25} | {fmt.format(v1):<15} | {fmt.format(v2):<15} | {str(v3):<15}")

        # Table 2: BC Types
        write("\n[Table 2] Boundary Condition Distribution (%)")
        write(f"{'Type':<20} | {'Original':<10} | {'New':<10}")
        write("-" * 46)
        
        all_bcs = set(original.stats['bc_types'].keys()) | set(new.stats['bc_types'].keys())
        total_bc1 = sum(original.stats['bc_types'].values())
        total_bc2 = sum(new.stats['bc_types'].values())
        
        for bc in sorted(all_bcs):
            p1 = (original.stats['bc_types'][bc] / total_bc1 * 100) if total_bc1 else 0
            p2 = (new.stats['bc_types'][bc] / total_bc2 * 100) if total_bc2 else 0
            write(f"{bc:<20} | {p1:5.1f}%     | {p2:5.1f}%")

        # Table 3: Load Specs
        write("\n[Table 3] Load Specifications")
        write(f"{'Type':<15} | {'Dataset':<10} | {'%':<6} | {'Mean Mag':<10} | {'Range':<15}")
        write("-" * 65)
        
        all_loads = set(original.stats['load_types'].keys()) | set(new.stats['load_types'].keys())
        
        for ld in sorted(all_loads):
            # Original
            c1 = original.stats['load_types'][ld]
            tot1 = sum(original.stats['load_types'].values())
            pct1 = (c1/tot1*100) if tot1 else 0
            mags1 = original.stats['load_magnitudes'][ld]
            mean1 = statistics.mean(mags1) if mags1 else 0
            range1 = f"{min(mags1):.0f}-{max(mags1):.0f}" if mags1 else "-"
            
            write(f"{ld:<15} | {'Orig':<10} | {pct1:5.1f}% | {mean1:<10.1f} | {range1:<15}")
            
            # New
            c2 = new.stats['load_types'][ld]
            tot2 = sum(new.stats['load_types'].values())
            pct2 = (c2/tot2*100) if tot2 else 0
            mags2 = new.stats['load_magnitudes'][ld]
            mean2 = statistics.mean(mags2) if mags2 else 0
            range2 = f"{min(mags2):.0f}-{max(mags2):.0f}" if mags2 else "-"
            
            write(f"{'':<15} | {'New':<10} | {pct2:5.1f}% | {mean2:<10.1f} | {range2:<15}")
            write("-" * 65)

def get_df(original, new, stats_key, top_n=None):
    data = []
    # Original
    for k, v in original.stats[stats_key].items():
        data.append({'Label': k, 'Count': v, 'Dataset': 'Original'})
    # New
    for k, v in new.stats[stats_key].items():
        data.append({'Label': k, 'Count': v, 'Dataset': 'New (Complex)'})
    
    df = pd.DataFrame(data)
    if top_n and not df.empty:
        # Filter to top N overall
        total_counts = df.groupby('Label')['Count'].sum().sort_values(ascending=False)
        top_labels = total_counts.head(top_n).index
        df = df[df['Label'].isin(top_labels)]
    return df

def save_plot(fig, filename):
    plt.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved {filename}. Looks good.")
    plt.close(fig)

def create_individual_plots(original, new, output_dir="plots"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Geometry Distribution (Figure 3)
    df_geo = get_df(original, new, 'geometries')
    if not df_geo.empty:
        fig = plt.figure(figsize=(10, 6))
        sns.barplot(data=df_geo, x='Label', y='Count', hue='Dataset')
        plt.title('Geometry Distribution', fontweight='bold')
        plt.xlabel('')
        plt.xticks(rotation=45)
        save_plot(fig, f"{output_dir}/geometry_distribution.png")

    # 2. Material Distribution (Figure 4)
    df_mat = get_df(original, new, 'materials', top_n=10)
    if not df_mat.empty:
        fig = plt.figure(figsize=(12, 6))
        sns.barplot(data=df_mat, x='Label', y='Count', hue='Dataset')
        plt.title('Material Distribution (Top 10)', fontweight='bold')
        plt.xlabel('')
        plt.xticks(rotation=45)
        save_plot(fig, f"{output_dir}/material_distribution.png")

    # 3. Location Descriptors (Figure 5)
    df_loc = get_df(original, new, 'locations', top_n=15)
    if not df_loc.empty:
        fig = plt.figure(figsize=(10, 8))
        sns.barplot(data=df_loc, x='Count', y='Label', hue='Dataset', orient='h')
        plt.title('Location Descriptor Usage (Top 15)', fontweight='bold')
        plt.ylabel('')
        save_plot(fig, f"{output_dir}/location_descriptors.png")

    # 4. BC Types
    df_bc = get_df(original, new, 'bc_types')
    if not df_bc.empty:
        fig = plt.figure(figsize=(10, 6))
        sns.barplot(data=df_bc, x='Label', y='Count', hue='Dataset')
        plt.title('Boundary Condition Types', fontweight='bold')
        plt.xlabel('')
        save_plot(fig, f"{output_dir}/bc_types.png")

    # 5. Load Types
    df_load = get_df(original, new, 'load_types')
    if not df_load.empty:
        fig = plt.figure(figsize=(10, 6))
        sns.barplot(data=df_load, x='Label', y='Count', hue='Dataset')
        plt.title('Load Types', fontweight='bold')
        plt.xlabel('')
        save_plot(fig, f"{output_dir}/load_types.png")
    
    # 6. Prompt Length Distribution
    p1 = [len(ex['prompt'].split()) for ex in original.examples]
    p2 = [len(ex['prompt'].split()) for ex in new.examples]
    if p1 or p2:
        fig = plt.figure(figsize=(10, 6))
        plt.hist([p1, p2], label=['Original', 'New'], bins=20, alpha=0.7)
        plt.title('Prompt Length Distribution', fontweight='bold')
        plt.xlabel('Words')
        plt.legend()
        save_plot(fig, f"{output_dir}/prompt_lengths.png")

def create_combined_plots(original, new, output_dir="plots"):
    """Generate plots for the combined dataset (Original + New)"""
    print(f"Smashing the data together for plots in {output_dir}...")
    
    # Helper to combine counters
    def combine_stats(key):
        c1 = Counter(original.stats[key])
        c2 = Counter(new.stats[key])
        return c1 + c2

    # 1. Combined Geometry Distribution
    geo_counts = combine_stats('geometries')
    df_geo = pd.DataFrame.from_dict(geo_counts, orient='index', columns=['Count']).reset_index()
    df_geo.columns = ['Geometry Type', 'Count']
    if not df_geo.empty:
        fig = plt.figure(figsize=(10, 6))
        sns.barplot(data=df_geo, x='Geometry Type', y='Count', color='steelblue')
        plt.title('Combined Geometry Distribution', fontweight='bold')
        plt.xlabel('')
        plt.xticks(rotation=45)
        save_plot(fig, f"{output_dir}/combined_geometry_distribution.png")

    # 2. Combined Material Distribution (Top 10)
    mat_counts = combine_stats('materials')
    df_mat = pd.DataFrame.from_dict(mat_counts, orient='index', columns=['Count']).reset_index()
    df_mat.columns = ['Material', 'Count']
    if not df_mat.empty:
        df_mat = df_mat.sort_values('Count', ascending=False).head(10)
        fig = plt.figure(figsize=(12, 6))
        sns.barplot(data=df_mat, x='Material', y='Count', color='mediumseagreen')
        plt.title('Combined Material Distribution (Top 10)', fontweight='bold')
        plt.xlabel('')
        plt.xticks(rotation=45)
        save_plot(fig, f"{output_dir}/combined_material_distribution.png")

    # 3. Combined BC Types
    bc_counts = combine_stats('bc_types')
    df_bc = pd.DataFrame.from_dict(bc_counts, orient='index', columns=['Count']).reset_index()
    df_bc.columns = ['BC Type', 'Count']
    if not df_bc.empty:
        fig = plt.figure(figsize=(10, 6))
        sns.barplot(data=df_bc, x='BC Type', y='Count', color='indianred')
        plt.title('Combined Boundary Condition Types', fontweight='bold')
        plt.xlabel('')
        save_plot(fig, f"{output_dir}/combined_bc_types.png")

    # 4. Combined Load Types
    load_counts = combine_stats('load_types')
    df_load = pd.DataFrame.from_dict(load_counts, orient='index', columns=['Count']).reset_index()
    df_load.columns = ['Load Type', 'Count']
    if not df_load.empty:
        fig = plt.figure(figsize=(10, 6))
        sns.barplot(data=df_load, x='Load Type', y='Count', color='mediumpurple')
        plt.title('Combined Load Types', fontweight='bold')
        plt.xlabel('')
        save_plot(fig, f"{output_dir}/combined_load_types.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', default='TrainingExamples_900.jsonl')
    parser.add_argument('--new', default='Complex_Training_Data.jsonl')
    args = parser.parse_args()

    # Load
    orig_data = DatasetAnalyzer("Original", args.original)
    new_data = DatasetAnalyzer("New", args.new)

    # Analyze
    orig_data.analyze()
    new_data.analyze()

    # Report
    create_comparison_report(orig_data, new_data)
    
    # Plots
    create_individual_plots(orig_data, new_data)
    create_combined_plots(orig_data, new_data)

if __name__ == "__main__":
    main()
