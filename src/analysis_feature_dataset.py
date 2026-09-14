# Goal: Implements both 3-class (perfect match) and 6-class (all activities) analysis

""" 
1) Generate both datasets (3-class and 6-class versions)
2) Run 3-class cross-dataset evaluation FIRST
3) Document baseline performance and degradation
4) Then run 6-class as comparative analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from pathlib import Path

PHASE_TAG = "phase2"

# Set style 
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# ============================================================================
# STEP 1: LOAD UCI HAR DATASET (SOURCE DOMAIN)
# ============================================================================
def load_uci_har_data(data_dir):

    print("=" * 80)
    print("LOADING UCI HAR DATASET (Source Domain)")
    print("=" * 80)
    
    # Load feature matrix
    X_train = pd.read_csv(os.path.join(data_dir, 'train', 'X_train.txt'), sep=r'\s+', header=None)
    
    # Load labels
    y_train = pd.read_csv(os.path.join(data_dir, 'train', 'y_train.txt'), header=None, names=['activity'])
    
    # Load subjects
    subject_train = pd.read_csv(os.path.join(data_dir, 'train', 'subject_train.txt'), header=None, names=['subject'])
    
    # Activity mapping
    activity_labels = {
        1: 'WALKING',
        2: 'WALKING_UPSTAIRS', 
        3: 'WALKING_DOWNSTAIRS',
        4: 'SITTING',
        5: 'STANDING',
        6: 'LAYING'
    }
    
    y_train['activity_name'] = y_train['activity'].map(activity_labels)
    
    print(f"  Loaded UCI HAR training data")
    print(f"  Shape: {X_train.shape}")
    print(f"  Samples per activity:")
    for act, count in y_train['activity_name'].value_counts().sort_index().items():
        print(f"    {act}: {count}")
    print()
    
    return X_train, y_train, subject_train, activity_labels

# ============================================================================
# STEP 2: LOAD WISDM DATASET (TARGET DOMAIN)  
# ============================================================================
def load_wisdm_data(data_dir):

    print("=" * 80)
    print("LOADING WISDM DATASET (Target Domain)")
    print("=" * 80)
    
    # Load feature matrix
    X_wisdm = pd.read_csv(os.path.join(data_dir, 'X_wisdm.txt'), sep=r'\s+', header=None)
    
    # Load labels  
    y_wisdm = pd.read_csv(os.path.join(data_dir, 'y_wisdm.txt'), header=None, names=['activity'])
    
    # Load subjects
    subject_wisdm = pd.read_csv(os.path.join(data_dir, 'subject_wisdm.txt'), header=None, names=['subject'])
    
    # Activity mapping (same as UCI)
    activity_labels = {
        1: 'WALKING',
        2: 'WALKING_UPSTAIRS',
        3: 'WALKING_DOWNSTAIRS', 
        4: 'SITTING',
        5: 'STANDING',
        6: 'LAYING'
    }
    
    y_wisdm['activity_name'] = y_wisdm['activity'].map(activity_labels)
    
    print(f"  Loaded WISDM data")
    print(f"  Shape: {X_wisdm.shape}")
    print(f"  Samples per activity:")
    for act, count in y_wisdm['activity_name'].value_counts().sort_index().items():
        print(f"    {act}: {count}")
    print()
    
    return X_wisdm, y_wisdm, subject_wisdm, activity_labels

# ============================================================================
# STEP 3: FILTER FOR 3-CLASS (PERFECT MATCH) OR 6-CLASS
# ============================================================================
def filter_activities(X, y, subjects, activity_subset='3class'):

    print("=" * 80)
    print(f"FILTERING TO {activity_subset.upper()} ACTIVITIES")
    print("=" * 80)
    
    if activity_subset == '3class':
        # Perfect matches only
        perfect_match_activities = ['WALKING', 'SITTING', 'STANDING']
        mask = y['activity_name'].isin(perfect_match_activities)
        
        print("  Filtering to 3 perfect-match activities:")
        print("  - WALKING (A -> WALKING)")
        print("  - SITTING (D -> SITTING)") 
        print("  - STANDING (E -> STANDING)")
        
    else:  # 6class
        # All activities
        mask = y['activity_name'].notna()
        print("  Using all 6 activities:")
        print("  Perfect matches: WALKING, SITTING, STANDING")
        print("  Approximate matches: WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, LAYING")
    
    X_filtered = X[mask].reset_index(drop=True)
    y_filtered = y[mask].reset_index(drop=True)
    subjects_filtered = subjects[mask].reset_index(drop=True)
    
    # Remap labels to consecutive integers
    unique_activities = sorted(y_filtered['activity_name'].unique())
    activity_to_label = {act: i+1 for i, act in enumerate(unique_activities)}
    y_filtered['activity'] = y_filtered['activity_name'].map(activity_to_label)
    
    print(f"\n  Original samples: {len(y):,}")
    print(f"  Filtered samples: {len(y_filtered):,}")
    print(f"  Retention rate: {100*len(y_filtered)/len(y):.1f}%")
    print(f"\n  Activity distribution after filtering:")
    for act in unique_activities:
        count = (y_filtered['activity_name'] == act).sum()
        pct = 100 * count / len(y_filtered)
        print(f"    {act}: {count:,} ({pct:.1f}%)")
    print()
    
    return X_filtered, y_filtered, subjects_filtered, activity_to_label

# ============================================================================
# STEP 4: KOLMOGOROV-SMIRNOV TEST (Domain Shift Analysis)
# ============================================================================
def perform_ks_tests(X_source, X_target, y_source, y_target, activity_subset='3class', output_dir='results'):

    print("=" * 80)
    print("PERFORMING KOLMOGOROV-SMIRNOV TESTS")
    print("=" * 80)

    # Match feature count (UCI has 561, WISDM has 61)
    n_features = min(X_source.shape[1], X_target.shape[1])
    X_source_matched = X_source.iloc[:, :n_features]
    X_target_matched = X_target.iloc[:, :n_features]

    print(f"Analyzing {n_features} matched features")
    print(f"Source samples: {len(X_source_matched):,}")
    print(f"Target samples: {len(X_target_matched):,}\n")

    # ==== OVERALL K-S TESTS (across all activities) ====
    print("=" * 80)
    print("OVERALL DOMAIN SHIFT (All Activities Combined)")
    print("=" * 80)
    
    ks_results = []
    for col_idx in range(n_features):
        source_feature = X_source_matched.iloc[:, col_idx]
        target_feature = X_target_matched.iloc[:, col_idx]

        # K-S test
        ks_stat, p_value = stats.ks_2samp(source_feature, target_feature)

        ks_results.append({
            'feature_idx': col_idx,
            'ks_statistic': ks_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        })

    ks_df = pd.DataFrame(ks_results).sort_values('ks_statistic', ascending=False)

    # Summary statistics
    significant_count = ks_df['significant'].sum()
    mean_ks = ks_df['ks_statistic'].mean()
    median_ks = ks_df['ks_statistic'].median()

    print(f"\nOverall Domain Shift Summary:")
    print(f"  Significant differences (p < 0.05): {significant_count}/{n_features} ({100*significant_count/n_features:.1f}%)")
    print(f"  Mean K-S statistic: {mean_ks:.4f}")
    print(f"  Median K-S statistic: {median_ks:.4f}")

    print(f"\nTop 15 features with largest domain shift:")
    print(ks_df.head(15)[['feature_idx', 'ks_statistic', 'p_value', 'significant']].to_string(index=False))

    # Save overall results
    os.makedirs(output_dir, exist_ok=True)
    ks_path = os.path.join(output_dir, f'ks_test_overall_{activity_subset}.csv')
    ks_df.to_csv(ks_path, index=False)
    print(f"\n Saved: {ks_path}")

    # ==== PER-ACTIVITY K-S TESTS ====
    print("\n" + "=" * 80)
    print("PER-ACTIVITY DOMAIN SHIFT ANALYSIS")
    print("=" * 80)
    
    unique_activities = sorted(y_source['activity_name'].unique())
    per_activity_results = []

    for activity in unique_activities:
        print(f"\n--- {activity} ---")

        # Filter to this activity only
        source_mask = y_source['activity_name'] == activity
        target_mask = y_target['activity_name'] == activity
        
        X_source_activity = X_source_matched[source_mask]
        X_target_activity = X_target_matched[target_mask]
        
        print(f"Source samples: {len(X_source_activity):,}")
        print(f"Target samples: {len(X_target_activity):,}")
        
        if len(X_source_activity) == 0 or len(X_target_activity) == 0:
            print(" Skipping - insufficient samples")
            continue

        activity_ks_results = []
        for col_idx in range(n_features):
            source_feature = X_source_activity.iloc[:, col_idx]
            target_feature = X_target_activity.iloc[:, col_idx]
            
            ks_stat, p_value = stats.ks_2samp(source_feature, target_feature)
            
            activity_ks_results.append({
                'activity': activity,
                'feature_idx': col_idx,
                'ks_statistic': ks_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            })

        activity_ks_df = pd.DataFrame(activity_ks_results)
        
        # Summary for this activity
        sig_count = activity_ks_df['significant'].sum()
        mean_ks_act = activity_ks_df['ks_statistic'].mean()
        
        print(f"Significant features: {sig_count}/{n_features} ({100*sig_count/n_features:.1f}%)")
        print(f"Mean K-S statistic: {mean_ks_act:.4f}")
        
        per_activity_results.append(activity_ks_df)

     # Combine all per-activity results
    if per_activity_results:
        all_activity_ks = pd.concat(per_activity_results, ignore_index=True)
        activity_ks_path = os.path.join(output_dir, f'ks_test_per_activity_{activity_subset}.csv')
        all_activity_ks.to_csv(activity_ks_path, index=False)
        print(f"\n Saved: {activity_ks_path}")
    
    # ==== VISUALIZATION ====
    plot_ks_distribution(ks_df, activity_subset, output_dir)
    plot_per_activity_ks(per_activity_results, activity_subset, output_dir)
    
    print("\n" + "=" * 80)
    
    return ks_df, per_activity_results

# ============================================================================
# VISUALIZATION: K-S DISTRIBUTION
# ============================================================================
def plot_ks_distribution(ks_df, activity_subset, output_dir):
    """Plot distribution of K-S statistics"""
     
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

     # Histogram of K-S statistics
    axes[0].hist(ks_df['ks_statistic'], bins=30, edgecolor='black', alpha=0.7)
    axes[0].axvline(ks_df['ks_statistic'].median(), color='red', 
                    linestyle='--', label=f'Median: {ks_df["ks_statistic"].median():.3f}')
    axes[0].set_xlabel('K-S Statistic')
    axes[0].set_ylabel('Number of Features')
    axes[0].set_title(f'Distribution of K-S Statistics ({activity_subset.upper()})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Significance breakdown
    sig_counts = ks_df['significant'].value_counts()
    colors = ['#d62728' if x else '#2ca02c' for x in [True, False]]
    axes[1].bar(['Significant\n(p < 0.05)', 'Not Significant\n(p ≥ 0.05)'], 
                [sig_counts.get(True, 0), sig_counts.get(False, 0)],
                color=colors, alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('Number of Features')
    axes[1].set_title('Statistical Significance of Domain Shift')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add percentages on bars
    total = len(ks_df)
    for i, (val, count) in enumerate([(True, sig_counts.get(True, 0)), 
                                       (False, sig_counts.get(False, 0))]):
        pct = 100 * count / total
        axes[1].text(i, count + total*0.02, f'{count}\n({pct:.1f}%)', 
                    ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'ks_distribution_{activity_subset}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f" Saved: {save_path}")
    plt.close()




def plot_per_activity_ks(per_activity_results, activity_subset, output_dir):
    """Plot per-activity K-S statistics comparison"""

    if not per_activity_results:
        return
    
    # Calculate mean K-S per activity
    activity_means = []
    for df in per_activity_results:
        activity = df['activity'].iloc[0]
        mean_ks = df['ks_statistic'].mean()
        sig_pct = 100 * df['significant'].sum() / len(df)
        activity_means.append({
            'activity': activity,
            'mean_ks': mean_ks,
            'sig_percentage': sig_pct
        })
    
    summary_df = pd.DataFrame(activity_means).sort_values('mean_ks', ascending=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Mean K-S statistic per activity
    axes[0].barh(summary_df['activity'], summary_df['mean_ks'], 
                 color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Mean K-S Statistic')
    axes[0].set_title(f'Domain Shift by Activity ({activity_subset.upper()})')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Percentage of significant features per activity
    axes[1].barh(summary_df['activity'], summary_df['sig_percentage'],
                 color='coral', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('% Features with Significant Shift (p < 0.05)')
    axes[1].set_title('Statistical Significance by Activity')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'ks_per_activity_{activity_subset}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f" Saved: {save_path}")
    plt.close()


# ============================================================================
# STEP 5: SAVE FILTERED DATASETS FOR NEXT STAGE
# ============================================================================
def save_filtered_datasets(X, y, subjects, activity_subset, output_dir):
    """Save filtered datasets for later model training"""

    print("=" * 80)
    print(f"SAVING FILTERED {activity_subset.upper()} DATASET")
    print("=" * 80)

    subset_dir = os.path.join(output_dir, activity_subset)
    os.makedirs(subset_dir, exist_ok=True)

    # Save features, labels, subjects
    X.to_csv(os.path.join(subset_dir, 'X_filtered.txt'), sep=' ', index=False, header=False)
    y[['activity']].to_csv(os.path.join(subset_dir, 'y_filtered.txt'), index=False, header=False)
    subjects.to_csv(os.path.join(subset_dir, 'subject_filtered.txt'), index=False, header=False)

    # Save activity mapping
    unique_activities = sorted(y['activity_name'].unique())
    activity_mapping = {act: (y[y['activity_name']==act]['activity'].iloc[0]) 
                       for act in unique_activities}
    
    mapping_df = pd.DataFrame([
        {'label': label, 'activity': act} 
        for act, label in sorted(activity_mapping.items(), key=lambda x: x[1])
    ])
    mapping_df.to_csv(os.path.join(subset_dir, 'activity_mapping.csv'), index=False)
    
    print(f"  Saved filtered dataset to: {subset_dir}/")
    print(f"  Files: X_filtered.txt, y_filtered.txt, subject_filtered.txt, activity_mapping.csv")
    print()

# ============================================================================
# MAIN PIPELINE (DATA PREP & K-S TESTS ONLY)
# ============================================================================
def run_data_prep_and_ks_tests(uci_dir, wisdm_dir, activity_subset='3class', output_dir='results'):
    """
    Stage 1 & 2: Data preparation and K-S tests
    NO MODEL TRAINING YET
    """
    print("\n")
    print("+" + "=" * 78 + "+")
    print("|" + " " * 15 + "DATA PREPARATION & K-S TESTING" + " " * 32 + "|")
    print("|" + " " * 25 + f"Mode: {activity_subset.upper()}" + " " * (52 - len(activity_subset)) + "|")
    print("+" + "=" * 78 + "+")
    print("\n")

    # Load data
    X_uci, y_uci, subj_uci, labels = load_uci_har_data(uci_dir)
    X_wisdm, y_wisdm, subj_wisdm, _ = load_wisdm_data(wisdm_dir)

    # Filter to desired activities
    X_uci_filt, y_uci_filt, subj_uci_filt, act_map_uci = filter_activities(X_uci, y_uci, subj_uci, activity_subset)
    X_wisdm_filt, y_wisdm_filt, subj_wisdm_filt, act_map_wisdm = filter_activities(X_wisdm, y_wisdm, subj_wisdm, activity_subset)

    # K-S tests
    ks_overall, ks_per_activity = perform_ks_tests(X_uci_filt, X_wisdm_filt, y_uci_filt, y_wisdm_filt,
        activity_subset, output_dir=os.path.join(output_dir, activity_subset))
    
    # Save filtered datasets for next stage
    save_filtered_datasets(X_uci_filt, y_uci_filt, subj_uci_filt, f'{activity_subset}_uci_{PHASE_TAG}', output_dir)
    save_filtered_datasets(X_wisdm_filt, y_wisdm_filt, subj_wisdm_filt, f'{activity_subset}_wisdm_{PHASE_TAG}', output_dir)

    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Configuration: {activity_subset.upper()}")
    print(f"UCI HAR (source): {len(X_uci_filt):,} samples")
    print(f"WISDM (target): {len(X_wisdm_filt):,} samples")
    print(f"\nDomain Shift Analysis:")
    sig_count = ks_overall['significant'].sum()
    print(f"  Significant features: {sig_count}/{len(ks_overall)} ({100*sig_count/len(ks_overall):.1f}%)")
    print(f"  Mean K-S statistic: {ks_overall['ks_statistic'].mean():.4f}")
    print(f"\n Datasets prepared and saved for model training (Stage 3)")
    print("=" * 80 + "\n")
    
    return ks_overall, ks_per_activity

# ============================================================================
# USAGE
# ============================================================================
if __name__ == "__main__":
    # Paths anchored to the repo root (this file lives in <repo_root>/src/).
    REPO_ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = REPO_ROOT / "data"

    # Place the raw UCI HAR dataset here (see README "Data" section).
    UCI_HAR_DIR = str(DATA_DIR / "raw" / "uci_har")
    # Output of feature_extraction_WISDM.py. Must match PHASE_TAG (line ~17 in this
    # file) and the PHASE_TAG that generated data/interim/wisdm_{PHASE_TAG}/.
    WISDM_DIR = str(DATA_DIR / "interim" / f"wisdm_{PHASE_TAG}")
    OUTPUT_DIR = str(DATA_DIR / "interim" / "Filtered_datasets_and_KS_results")

    # ==== PRIMARY: 3-CLASS ANALYSIS ====
    print("\n" + "= " * 40)
    print("PRIMARY ANALYSIS: 3-CLASS (PERFECT MATCHES)")
    print("= " * 40 + "\n")
    
    ks_3class, per_act_3class = run_data_prep_and_ks_tests(
        uci_dir=UCI_HAR_DIR,
        wisdm_dir=WISDM_DIR,
        activity_subset='3class',
        output_dir=OUTPUT_DIR
    )
    
    # ==== SECONDARY: 6-CLASS ANALYSIS ====
    print("\n" + "=" * 40)
    print("SECONDARY ANALYSIS: 6-CLASS (ALL ACTIVITIES)")
    print("= " * 40 + "\n")
    
    ks_6class, per_act_6class = run_data_prep_and_ks_tests(
        uci_dir=UCI_HAR_DIR,
        wisdm_dir=WISDM_DIR,
        activity_subset='6class',
        output_dir=OUTPUT_DIR
    )
    
    print("\n" + "=" * 80)
    print("DATA PREPARATION COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print(f"  {OUTPUT_DIR}/3class/ - 3-class datasets & K-S results")
    print(f"  {OUTPUT_DIR}/6class/ - 6-class datasets & K-S results")
    print("\nNext: Use these filtered datasets for model training (Stage 3)")
    print("=" * 80)