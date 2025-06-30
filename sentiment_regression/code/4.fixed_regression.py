import pandas as pd
import numpy as np
import warnings
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class MultiMethodFixedAnalysis:
    def __init__(self, data_dir="../result/"):
        """Initialize multi-method fixed effects analysis"""
        self.data_dir = data_dir
        self.results_df = pd.DataFrame()
        self.log_content = []

    def log_print(self, message):
        """Print and log message"""
        print(message)
        self.log_content.append(message)

    def _simplify_method_name(self, method_name):
        """Simplify method names for display"""
        method_lower = method_name.lower()
        
        # Define mapping based on keywords in filename
        if "aliyun" in method_lower:
            return "Aliyun"
        elif "baidu" in method_lower:
            return "Baidu"
        elif "erlangshen" in method_lower:
            return "Erlangshen"
        elif "cemotion" in method_lower:
            return "Cemotion"
        elif "openai" in method_lower:
            return "OpenAI"
        elif "snownlp" in method_lower:
            return "SnowNLP"
        elif "tencent" in method_lower:
            return "Tencent"
        else:
            return method_name  # Keep original if no match

    def load_all_sentiment_data(self):
        """Load all sentiment analysis result files"""
        self.log_print("Loading all sentiment analysis results...")
        
        # Find all sentiment data files
        pattern = os.path.join(self.data_dir, "3.sentiment_data_*.csv")
        files = glob.glob(pattern)
        
        all_data = []
        methods = []
        method_mapping = {}  # Store original -> simplified mapping
        
        for file_path in files:
            try:
                # Extract method name from filename
                filename = os.path.basename(file_path)
                original_method = filename.replace("3.sentiment_data_", "").replace(".csv", "")
                simplified_method = self._simplify_method_name(original_method)
                
                methods.append(simplified_method)
                method_mapping[original_method] = simplified_method
                
                # Load data (sample first to check structure)
                df = pd.read_csv(file_path, nrows=1000)  # Sample first for speed
                
                # Check if required columns exist
                if 'role' not in df.columns:
                    self.log_print(f"Warning: {simplified_method} missing 'role' column, skipping")
                    continue
                if 'sentiment' not in df.columns:
                    self.log_print(f"Warning: {simplified_method} missing 'sentiment' column, skipping")
                    continue
                
                # Load full data
                df = pd.read_csv(file_path)
                df['method'] = simplified_method  # Use simplified name
                all_data.append(df)
                
                self.log_print(f"Loaded {simplified_method}: {len(df)} records")
                
            except Exception as e:
                self.log_print(f"Error loading {file_path}: {str(e)}")
                continue
        
        if not all_data:
            raise ValueError("No valid sentiment data files found")
        
        # Combine all data
        self.combined_data = pd.concat(all_data, ignore_index=True)
        self.methods = methods
        self.method_mapping = method_mapping
        
        self.log_print(f"Combined dataset: {len(self.combined_data)} records from {len(methods)} methods")
        self.log_print(f"Simplified method names: {methods}")
        
        # Show role distribution
        role_dist = self.combined_data['role'].value_counts()
        self.log_print(f"Role distribution: {role_dist.to_dict()}")
        
        return self.combined_data

    def prepare_analysis_data(self, df):
        """Prepare data for analysis"""
        # Create binary variables
        df['is_positive'] = (df['sentiment'] == '正面').astype(int)
        df['is_negative'] = (df['sentiment'] == '负面').astype(int)
        
        self.log_print(f"  is_positive cases: {df['is_positive'].sum()}")
        self.log_print(f"  is_negative cases: {df['is_negative'].sum()}")
        
        # Show key distributions for this dataset
        self.log_print("  is_positive by source distribution:")
        self.log_print(f"  {pd.crosstab(df['source'], df['is_positive'])}")
        
        self.log_print("  is_negative by source distribution:")
        self.log_print(f"  {pd.crosstab(df['source'], df['is_negative'])}")
        
        # Scale continuous variables
        if 'total_tokens' in df.columns:
            df['total_tokens_scaled'] = (df['total_tokens'] - df['total_tokens'].mean()) / df['total_tokens'].std()

        return df

    def run_fixed_effects_analysis(self):
        """Run fixed effects analysis for all methods and roles"""
        self.log_print("\n=== Fixed Effects Analysis for All Methods ===")
        
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
                        result = self._run_single_fixed_model(role_data, outcome, method, role)
                        if result:
                            results.append(result)
                    except Exception as e:
                        self.log_print(f"  ❌ Error in {method}-{role}-{outcome}: {str(e)}")
                        # Add failed result
                        error_type = 'singular_matrix' if ("singular matrix" in str(e).lower() or "LinAlgError" in str(e)) else 'other_error'
                        results.append({
                            'method': method,
                            'role': role,
                            'outcome': outcome,
                            'odds_ratio': 0.0 if error_type == 'singular_matrix' else np.nan,
                            'p_value': np.nan,
                            'significant': False,
                            'error': error_type
                        })
        
        self.results_df = pd.DataFrame(results)
        self._print_overall_summary()
        return self.results_df

    def _run_single_fixed_model(self, data, outcome, method, role):
        """Run single fixed effects logistic regression model"""
        self.log_print(f"Using fixed effects logistic regression")
        
        if len(data) < 50:
            self.log_print(f"  ⚠️ Sample size too small ({len(data)}), skipping")
            return None
        
        # Prepare variables
        reference_group = 'they' if outcome == 'is_positive' else 'we'
        
        # Prepare categorical variable with reference group
        data['source_cat'] = pd.Categorical(data['source'], categories=['we', 'they'])
        data['source_cat'] = data['source_cat'].cat.set_categories(
            [reference_group] + [cat for cat in ['we', 'they'] if cat != reference_group])
        
        self.log_print(f"Reference group: '{reference_group}'")

        # Build formula
        formula = f"{outcome} ~ C(source_cat)"
        if 'total_tokens_scaled' in data.columns:
            formula += " + total_tokens_scaled"
        if 'TTR' in data.columns:
            formula += " + TTR"
        if 'stm_topic' in data.columns:
            formula += " + C(stm_topic)"
        
        self.log_print(f"Formula: {formula}")

        try:
            model = smf.logit(formula, data=data).fit(disp=0)
            
            # Extract and print results
            result = self._extract_and_print_fixed_results(model, outcome, method, role)
            return result

        except Exception as e:
            self.log_print(f"❌ Model fitting failed: {str(e)}")
            
            # Check if it's a singular matrix error
            if "singular matrix" in str(e).lower() or "LinAlgError" in str(e):
                self.log_print("⚠️ Singular matrix detected - returning OR=0 for visualization")
                return {
                    'method': method,
                    'role': role,
                    'outcome': outcome,
                    'odds_ratio': 0.0,  # Set to 0 for singular matrix
                    'p_value': np.nan,
                    'significant': False,
                    'coefficient': np.nan,
                    'se': np.nan,
                    'error': 'singular_matrix'
                }
            else:
                # For other errors, still raise
                raise e

    def _extract_and_print_fixed_results(self, model, outcome, method, role):
        """Extract results and print detailed information for fixed effects"""
        self.log_print(f"Model fitted successfully (Fixed Effects)")
        
        # Find source coefficient
        source_param = None
        for param in model.params.index:
            if 'source_cat' in param and 'Intercept' not in param:
                source_param = param
                break
        
        if source_param:
            coef = model.params[source_param]
            se = model.bse[source_param]
            p_val = model.pvalues[source_param]
            odds_ratio = np.exp(coef)
            ci = np.exp(model.conf_int().loc[source_param])

            # Print detailed results
            self.log_print(f"Coefficient: {coef:.4f}")
            self.log_print(f"Odds Ratio: {odds_ratio:.4f} [95% CI: {ci[0]:.4f}-{ci[1]:.4f}]")
            self.log_print(f"P-value: {p_val:.4f}")

            significant = p_val < 0.05
            if significant:
                direction = "higher" if coef > 0 else "lower"
                sentiment_type = "positive" if outcome == 'is_positive' else "negative"
                self.log_print(f"✅ Result: Significant {direction} {sentiment_type} sentiment (p < 0.05)")
            else:
                self.log_print(f"✅ Result: No significant difference (p = {p_val:.3f})")

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
            self.log_print("❌ No source coefficient found in model")
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

                if pd.isna(or_val):
                    self.log_print(f"{method:20} {role:10} : FAILED")
                elif or_val == 0.0:
                    self.log_print(f"{method:20} {role:10} : SINGULAR MATRIX")
                else:
                    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                    self.log_print(f"{method:20} {role:10} : OR = {or_val:.3f}, p = {p_val:.3f} {sig}")

        # Success rate summary
        total_analyses = len(self.results_df)
        successful = len(self.results_df[~pd.isna(self.results_df['odds_ratio'])])
        singular_matrix = len(self.results_df[self.results_df['odds_ratio'] == 0.0])
        significant = len(self.results_df[self.results_df['significant'] == True])

        self.log_print(f"\nANALYSIS SUMMARY:")
        self.log_print(f"Total analyses: {total_analyses}")
        self.log_print(f"Successful: {successful} ({successful/total_analyses*100:.1f}%)")
        self.log_print(f"Singular matrix: {singular_matrix} ({singular_matrix/total_analyses*100:.1f}%)")
        self.log_print(f"Significant results: {significant} ({significant/total_analyses*100:.1f}%)")

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

            # Create annotation matrix - show values where p < 0.05 or mark singular matrix
            annot_matrix = pivot_data.copy()
            
            # Check for singular matrix cases
            singular_matrix_cases = outcome_data[outcome_data['error'] == 'singular_matrix']
            
            for i in range(len(annot_matrix.index)):
                for j in range(len(annot_matrix.columns)):
                    role = annot_matrix.index[i]
                    method = annot_matrix.columns[j]
                    
                    # Check if this is a singular matrix case
                    is_singular = not singular_matrix_cases[
                        (singular_matrix_cases['role'] == role) & 
                        (singular_matrix_cases['method'] == method)
                    ].empty
                    
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
                       annot_kws={'fontsize': 9, 'fontweight': 'bold'})

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
            output_path = f"../result/heatmap_fixed_{outcome}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            self.log_print(f"Saved heatmap: {output_path}")
            
        plt.show()

    def save_results(self):
        """Save detailed results to CSV"""
        if not self.results_df.empty:
            output_path = "../result/regression_results_fixed.csv"
            self.results_df.to_csv(output_path, index=False)
            self.log_print(f"Results saved: {output_path}")
        
        # Save log
        log_path = "../result/multi_method_analysis_fixed.txt"
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("MULTI-METHOD SENTIMENT ANALYSIS REGRESSION (FIXED EFFECTS)\n")
            f.write("=" * 60 + "\n\n")
            for line in self.log_content:
                f.write(line + "\n")
        self.log_print(f"Log saved: {log_path}")


def main():
    """Main analysis function"""
    print("=== Multi-Method Fixed Effects Analysis ===")

    analysis = MultiMethodFixedAnalysis()
    
    # Load all sentiment data
    analysis.load_all_sentiment_data()

    # Run fixed effects analysis
    results = analysis.run_fixed_effects_analysis()
    
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