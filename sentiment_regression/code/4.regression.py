import pandas as pd
import numpy as np
import warnings
import os
import matplotlib.pyplot as plt
import seaborn as sns


from pymer4.models import Lmer


warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class MultiMethodAnalysis:
    def __init__(self, data_dir="../result/"):
        """Initialize multi-method analysis"""
        self.data_dir = data_dir
        self.results_df = pd.DataFrame()
        self.log_content = []

    def log_print(self, message):
        """Print and log message"""
        print(message)
        self.log_content.append(message)

    def load_all_sentiment_data(self):
        """Load the combined sentiment analysis result file"""
        self.log_print("Loading combined sentiment analysis results...")
        
        # Load the combined data file
        combined_file = os.path.join(self.data_dir, "3.sentiment_data_all_methods.csv")
        if not os.path.exists(combined_file):
            raise ValueError(f"Combined sentiment data file not found: {combined_file}")
        
        # Load the data
        self.raw_data = pd.read_csv(combined_file)
        self.log_print(f"Loaded combined dataset: {len(self.raw_data)} records")
        
        # Define sentiment methods and their corresponding columns
        self.sentiment_methods = {
            'Erlangshen': 'sentiment_erlangshen',
            'Cemotion': 'sentiment_cemotion', 
            'SnowNLP': 'sentiment_snownlp',
            'Baidu': 'sentiment_baidu',
            'Aliyun': 'sentiment_aliyun',
            'OpenAI': 'sentiment_openai',
            'Tencent': 'sentiment_tencent'
        }
        
        # Create separate datasets for each method
        all_data = []
        methods = []
        
        for method_name, sentiment_col in self.sentiment_methods.items():
            if sentiment_col in self.raw_data.columns:
                # Create a copy of the data for this method
                method_data = self.raw_data.copy()
                
                # Select only rows where this method has results (not null)
                method_data = method_data[method_data[sentiment_col].notna()].copy()
                
                if len(method_data) > 0:
                    # Rename the sentiment column to 'sentiment' for consistency
                    method_data['sentiment'] = method_data[sentiment_col]
                    method_data['method'] = method_name
                    
                    # Select relevant columns
                    base_cols = ['text', 'group', 'role', 'model', 'TTR', 'total_tokens', 
                               'total_tokens_scaled', 'stm_topic', 'stm_topic_probability']
                    final_cols = base_cols + ['sentiment', 'method']
                    
                    # Only keep columns that exist
                    existing_cols = [col for col in final_cols if col in method_data.columns]
                    method_data = method_data[existing_cols]
                    
                    all_data.append(method_data)
                    methods.append(method_name)
                    
                    self.log_print(f"Loaded {method_name}: {len(method_data)} records")
                else:
                    self.log_print(f"Warning: No data found for {method_name}")
            else:
                self.log_print(f"Warning: Column {sentiment_col} not found for {method_name}")
        
        if not all_data:
            raise ValueError("No valid sentiment data found in any methods")
        
        # Combine all data
        self.combined_data = pd.concat(all_data, ignore_index=True)
        self.methods = methods
        
        self.log_print(f"Combined dataset: {len(self.combined_data)} records from {len(methods)} methods")
        self.log_print(f"Available methods: {methods}")
        
        # Show role distribution
        role_dist = self.combined_data['role'].value_counts()
        self.log_print(f"Role distribution: {role_dist.to_dict()}")
        
        # Show sentiment distribution by method
        for method in methods:
            method_data = self.combined_data[self.combined_data['method'] == method]
            sentiment_dist = method_data['sentiment'].value_counts()
            self.log_print(f"{method} sentiment distribution: {sentiment_dist.to_dict()}")
        
        return self.combined_data

    def prepare_analysis_data(self, df):
        """Prepare data for analysis"""
        # Create binary variables
        df['is_positive'] = (df['sentiment'] == '正面').astype(int)
        df['is_negative'] = (df['sentiment'] == '负面').astype(int)
        
        self.log_print(f"  is_positive cases: {df['is_positive'].sum()}")
        self.log_print(f"  is_negative cases: {df['is_negative'].sum()}")
        
        # Show key distributions for this dataset
        self.log_print("  is_positive by group distribution:")
        self.log_print(f"  {pd.crosstab(df['group'], df['is_positive'])}")
        
        self.log_print("  is_negative by group distribution:")
        self.log_print(f"  {pd.crosstab(df['group'], df['is_negative'])}")
        
        # Scale continuous variables
        if 'total_tokens' in df.columns:
            df['total_tokens_scaled'] = (df['total_tokens'] - df['total_tokens'].mean()) / df['total_tokens'].std()

        return df

    def run_mixed_effects_analysis(self):
        """Run mixed effects analysis for all methods and roles"""
        
        self.log_print("\n=== Mixed Effects Analysis for All Methods ===")
        
        results = []
        
        for method in self.methods:
            self.log_print(f"\n{'='*60}")
            self.log_print(f"ANALYZING METHOD: {method.upper()}")
            self.log_print(f"{'='*60}")
            
            # Filter data for this method
            method_data = self.combined_data[self.combined_data['method'] == method].copy()

            # Check if we have both roles
            roles = method_data['role'].unique()
            self.log_print(f"Available roles: {roles}")
            
            if len(roles) < 2:
                self.log_print(f"  ⚠️ Warning: Only {roles} available for {method}, skipping")
                continue
            
            # Run analysis for each role and outcome combination
            for role in ['user', 'assistant']:
                if role not in method_data['role'].values:
                    self.log_print(f"  ⚠️ Role '{role}' not found in {method}")
                    continue
                
                self.log_print(f"\n--- Role: {role.upper()} ---")
                role_data = method_data[method_data['role'] == role].copy()
                self.log_print(f"Sample size: {len(role_data)}")
                
                # Prepare data and show distributions
                role_data = self.prepare_analysis_data(role_data)
                
                for outcome in ['is_positive', 'is_negative']:
                    self.log_print(f"\n=== {outcome.upper()} Analysis for {role} ===")
                    
                    try:
                        result = self._run_single_mixed_model(role_data, outcome, method, role)
                        if result:
                            results.append(result)
                    except Exception as e:
                        error_msg = str(e)
                        self.log_print(f"  ❌ Error in {method}-{role}-{outcome}: {error_msg}")
                        
                        # Check if it's a singular matrix error
                        is_singular = "Downdated VtV is not positive definite" in error_msg
                        
                        if is_singular:
                            self.log_print(f"  🔧 Detected singular matrix, setting odds_ratio=0")
                            # Add singular matrix result
                            results.append({
                                'method': method,
                                'role': role,
                                'outcome': outcome,
                                'odds_ratio': 0,
                                'p_value': np.nan,
                                'significant': False,
                                'error': 'singular_matrix'
                            })
                        else:
                            # Add regular failed result
                            results.append({
                                'method': method,
                                'role': role,
                                'outcome': outcome,
                                'odds_ratio': np.nan,
                                'p_value': np.nan,
                                'significant': False,
                                'error': error_msg
                            })
        
        self.results_df = pd.DataFrame(results)
        self._print_overall_summary()
        return self.results_df

    def _run_single_mixed_model(self, data, outcome, method, role):
        """Run single mixed effects model"""
        # Check if we have model variable for random effects
        has_random_effect = False
        grouping_var = None
        
        if 'model' in data.columns and data['model'].nunique() > 1:
            grouping_var = 'model'
            has_random_effect = True
            self.log_print(f"Using random effect: (1|{grouping_var})")
            self.log_print(f"Number of {grouping_var} groups: {data[grouping_var].nunique()}")
        else:
            self.log_print("No suitable random effect variable found, using fixed effects only")
            if len(data) < 50:
                self.log_print(f"  ⚠️ Sample size too small ({len(data)}), skipping")
                return None
        
        # Prepare variables
        reference_group = 'they' if outcome == 'is_positive' else 'we'
        data['group_ref'] = data['group'].replace({
            reference_group: 0, 
            'we' if reference_group == 'they' else 'they': 1
        })
        
        self.log_print(f"Reference group: '{reference_group}'")

        # Build formula
        formula = f"{outcome} ~ group_ref"
        if 'total_tokens_scaled' in data.columns:
            formula += " + total_tokens_scaled"
        if 'TTR' in data.columns:
            formula += " + TTR"
        # if 'stm_topic' in data.columns:
        #     formula += " + C(stm_topic)"

        if has_random_effect:
            formula += f" + (1 | model)"
        
        self.log_print(f"Formula: {formula}")

        try:
            model = Lmer(formula, data=data, family='binomial')
            fitted_model = model.fit()
            
            # Extract and print results
            result = self._extract_and_print_results(fitted_model, outcome, method, role, has_random_effect)
            return result

        except Exception as e:
            self.log_print(f"❌ Model fitting failed: {str(e)}")
            raise e

    def _extract_and_print_results(self, fitted_model, outcome, method, role, has_random_effect):
        """Extract results and print detailed information"""
        # Get coefficient table
        coef_table = fitted_model.coefs if hasattr(fitted_model, 'coefs') else fitted_model

        self.log_print(f"Model fitted successfully {'(Mixed Effects)' if has_random_effect else '(Fixed Effects)'}")

        # Find group coefficient
        group_row = None
        if 'group_ref' in coef_table.index:
            group_row = coef_table.loc[['group_ref']]
        elif len(coef_table) > 1:
            group_row = coef_table.iloc[1:2]
        
        if group_row is not None and not group_row.empty:
            # Get coefficient
            coef = None
            for col in ['Estimate', 'Coef', 'coef']:
                if col in group_row.columns:
                    coef = group_row[col].iloc[0]
                    break
            
            # Get standard error
            se = None
            for col in ['SE', 'Std.Error', 'se']:
                if col in group_row.columns:
                    se = group_row[col].iloc[0]
                break

            # Get p-value
            p_val = None
            for col in ['P-val', 'P>|z|', 'pvalue']:
                if col in group_row.columns:
                    p_val = group_row[col].iloc[0]
                    break

            if coef is not None:
                odds_ratio = np.exp(coef)

                # Print detailed results
                self.log_print(f"Coefficient: {coef:.4f}")
                if se is not None:
                    ci_lower = np.exp(coef - 1.96 * se)
                    ci_upper = np.exp(coef + 1.96 * se)
                    self.log_print(f"Odds Ratio: {odds_ratio:.4f} [95% CI: {ci_lower:.4f}-{ci_upper:.4f}]")
                else:
                    self.log_print(f"Odds Ratio: {odds_ratio:.4f}")

                if p_val is not None:
                    self.log_print(f"P-value: {p_val:.4f}")

                    significant = p_val < 0.05
                    if significant:
                        direction = "higher" if coef > 0 else "lower"
                        sentiment_type = "positive" if outcome == 'is_positive' else "negative"
                        self.log_print(f"✅ Result: Significant {direction} {sentiment_type} sentiment (p < 0.05)")
                    else:
                        self.log_print(f"✅ Result: No significant difference (p = {p_val:.3f})")
                else:
                    self.log_print("⚠️ P-value not available")
                    significant = False

                return {
                    'method': method,
                    'role': role,
                    'outcome': outcome,
                    'odds_ratio': odds_ratio,
                    'p_value': p_val,
                    'significant': significant,
                    'coefficient': coef,
                    'se': se,
                    'error': None
                }
            else:
                self.log_print("❌ Coefficient not found in results")
        else:
            self.log_print("❌ No group coefficient found in model")

        return None

    def _print_overall_summary(self):
        """Print overall summary of all results"""
        if self.results_df.empty:
            return

        self.log_print("\n" + "=" * 80)
        self.log_print("OVERALL SUMMARY - ALL METHODS AND ROLES")
        self.log_print("=" * 80)

        # Summary by outcome
        for outcome in ['is_positive', 'is_negative']:
            outcome_data = self.results_df[self.results_df['outcome'] == outcome]
            if outcome_data.empty:
                continue

            outcome_label = "INGROUP SOLIDARITY (Positive)" if outcome == 'is_positive' else "OUTGROUP HOSTILITY (Negative)"
            self.log_print(f"\n{outcome_label}:")
            self.log_print("-" * 50)

            for _, row in outcome_data.iterrows():
                method = row['method']
                role = row['role']
                or_val = row['odds_ratio']
                p_val = row['p_value']
                error = row.get('error', None)

                if error == 'singular_matrix':
                    self.log_print(f"{method:20} {role:10} : SINGULAR MATRIX")
                elif pd.isna(or_val):
                    self.log_print(f"{method:20} {role:10} : FAILED")
                else:
                    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                    self.log_print(f"{method:20} {role:10} : OR = {or_val:.3f}, p = {p_val:.3f} {sig}")

        # Success rate summary
        total_analyses = len(self.results_df)
        successful = len(self.results_df[~pd.isna(self.results_df['odds_ratio'])])
        significant = len(self.results_df[self.results_df['significant'] == True])
        singular_matrices = len(self.results_df[self.results_df['error'] == 'singular_matrix'])

        self.log_print(f"\nANALYSIS SUMMARY:")
        self.log_print(f"Total analyses: {total_analyses}")
        self.log_print(f"Successful: {successful} ({successful/total_analyses*100:.1f}%)")
        self.log_print(f"Significant results: {significant} ({significant/total_analyses*100:.1f}%)")
        self.log_print(f"Singular matrices: {singular_matrices} ({singular_matrices/total_analyses*100:.1f}%)")

    def create_heatmap_visualization(self):
        """Create heatmap visualization like the provided image"""
        if self.results_df.empty:
            self.log_print("No results to visualize")
            return

        # Create separate plots for positive and negative outcomes
        for outcome in ['is_positive', 'is_negative']:
            outcome_data = self.results_df[self.results_df['outcome'] == outcome].copy()

            if outcome_data.empty:
                continue

            # Create pivot table
            pivot_data = outcome_data.pivot(index='role', columns='method', values='odds_ratio')

            # Sort columns (methods) for better display
            method_order = ['Aliyun', 'Baidu', 'Erlangshen', 'Cemotion', 'OpenAI', 'SnowNLP', 'Tencent']
            available_methods = [method for method in method_order if method in pivot_data.columns]
            other_methods = [method for method in pivot_data.columns if method not in method_order]
            final_method_order = available_methods + other_methods

            pivot_data = pivot_data.reindex(columns=final_method_order)

            # Create significance mask (True where p < 0.05)
            significance = outcome_data.pivot(index='role', columns='method', values='significant')
            significance = significance.reindex(columns=final_method_order)

            # Create error status matrix to check for singular matrices
            error_status = outcome_data.pivot(index='role', columns='method', values='error')
            error_status = error_status.reindex(columns=final_method_order)
            
            # Create annotation matrix - only show values where p < 0.05
            annot_matrix = pivot_data.copy()
            for i in range(len(annot_matrix.index)):
                for j in range(len(annot_matrix.columns)):
                    # Check if it's a singular matrix error
                    is_singular = (not pd.isna(error_status.iloc[i, j]) and 
                                 error_status.iloc[i, j] == 'singular_matrix')
                    
                    if is_singular:
                        annot_matrix.iloc[i, j] = "Singular Matrix"  # Mark singular matrix
                    elif pd.isna(significance.iloc[i, j]) or not significance.iloc[i, j]:
                        annot_matrix.iloc[i, j] = ""  # Empty string for non-significant
                    else:
                        annot_matrix.iloc[i, j] = f"{pivot_data.iloc[i, j]:.2f}"  # Show value for significant

            # Create the plot
            plt.figure(figsize=(14, 6))

            # Create heatmap with custom annotations
            sns.heatmap(pivot_data,
                       annot=annot_matrix,  # Use custom annotation matrix
                       fmt='',  # Don't format numbers since we already formatted them
                       cmap='Reds',
                       center=1,
                       vmin=0,
                       vmax=4,
                       cbar_kws={'label': 'Odds Ratio'},
                       linewidths=0.5,
                       annot_kws={'fontsize': 11, 'fontweight': 'bold'})

            # Customize plot
            title = "Ingroup Solidarity" if outcome == 'is_positive' else "Outgroup Hostility"
            plt.title(f"{chr(97 + ['is_positive', 'is_negative'].index(outcome))} {title}",
                     fontsize=16, fontweight='bold', pad=20)

            # Format labels
            plt.xlabel('')
            plt.ylabel('')

            # Rotate x-axis labels and make them more readable
            plt.xticks(rotation=45, ha='right', fontsize=12)

            # Map role names to more readable format and center them
            role_labels = []
            for role in pivot_data.index:
                if role == 'user':
                    role_labels.append('User')
                elif role == 'assistant':
                    role_labels.append('Model')
                else:
                    role_labels.append(role.title())

            # Set y-axis labels with centering
            plt.yticks(range(len(role_labels)), role_labels, rotation=0, fontsize=12, va='center')
            
            # Adjust y-axis tick positions to center labels in cells
            ax = plt.gca()
            ax.set_yticks([i + 0.5 for i in range(len(role_labels))])
            ax.set_yticklabels(role_labels, rotation=0, fontsize=12, va='center')

            plt.tight_layout()
            
            # Save plot
            output_path = f"../result/heatmap_{outcome}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            self.log_print(f"Saved heatmap: {output_path}")
            
        plt.show()

    def save_results(self):
        """Save detailed results to CSV"""
        if not self.results_df.empty:
            output_path = "../result/regression_results.csv"
            self.results_df.to_csv(output_path, index=False)
            self.log_print(f"Results saved: {output_path}")
        
        # Save log
        log_path = "../result/multi_method_analysis.txt"
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("MULTI-METHOD SENTIMENT ANALYSIS REGRESSION\n")
            f.write("=" * 50 + "\n\n")
            for line in self.log_content:
                f.write(line + "\n")
        self.log_print(f"Log saved: {log_path}")


def main():
    """Main analysis function"""
    print("=== Multi-Method Group Effect Analysis ===")

    analysis = MultiMethodAnalysis()
    
    # Load all sentiment data
    analysis.load_all_sentiment_data()

    # Run mixed effects analysis
    results = analysis.run_mixed_effects_analysis()
    
    if results is not None and not results.empty:
        # Create visualization
        analysis.create_heatmap_visualization()
        
        # Save results
        analysis.save_results()

        # Print summary
        print("\n=== Results Summary ===")
        print(results.groupby(['outcome', 'role'])['significant'].sum())

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
