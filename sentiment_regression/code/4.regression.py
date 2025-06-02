import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import warnings
import os
import sys
from contextlib import redirect_stdout
import io

warnings.filterwarnings('ignore')


class GroupEffectAnalysis:
    def __init__(self, data_path="../result/3.sentiment_data_bert.csv"):
        """
        Initialize group effect analysis - focused on odds ratios
        """
        self.data_path = data_path
        self.df = None
        self.results = {}

        # Extract method and model info from file path
        self.file_suffix = self._extract_file_suffix()

        # Initialize log storage
        self.log_content = []

    def _extract_file_suffix(self):
        """Extract method and model information from file path for output naming"""
        filename = os.path.basename(self.data_path)

        # Remove file extension
        base_name = os.path.splitext(filename)[0]

        # Extract suffix after "3.sentiment_data_"
        if "3.sentiment_data_" in base_name:
            suffix = base_name.replace("3.sentiment_data_", "")
            return suffix
        else:
            return "unknown"

    def log_print(self, *args, **kwargs):
        """Print to console and save to log"""
        # Convert all arguments to string and join them
        message = ' '.join(str(arg) for arg in args)

        # Print to console
        print(message, **kwargs)

        # Save to log
        self.log_content.append(message)

    def save_log(self, output_path=None):
        """Save all logged content to txt file"""
        if output_path is None:
            output_path = f"../result/4.regression_analysis_{self.file_suffix}.txt"

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("GROUP EFFECT REGRESSION ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Data file: {self.data_path}\n")
            f.write(f"Analysis method/model: {self.file_suffix}\n")
            f.write("=" * 60 + "\n\n")

            for line in self.log_content:
                f.write(line + "\n")

        print(f"Analysis report saved to: {output_path}")
        return output_path

    def load_data(self):
        """Load sentiment analysis results"""
        self.log_print("Loading data...")
        self.df = pd.read_csv(self.data_path)
        self.log_print(f"Loaded {len(self.df)} records")
        return self.df

    def prepare_data(self):
        """Prepare data for analysis"""
        self.log_print(f"Group distribution: {self.df['source'].value_counts().to_dict()}")

        # Check for data quality issues
        self.diagnose_data()

        # Convert topic to categorical if it exists
        if 'topic' in self.df.columns:
            self.df['topic'] = self.df['topic'].astype('category')
            self.log_print(f"Topic distribution: {sorted(self.df['topic'].unique())}")

            topic_counts = self.df['topic'].value_counts()
            self.log_print(f"Number of topics: {len(topic_counts)}")
            self.log_print(f"Topic sample size range: {topic_counts.min()} - {topic_counts.max()}")

            if len(topic_counts) > 50:
                self.log_print(
                    f"⚠️  Large number of topics ({len(topic_counts)}), will auto-simplify model if singular matrix occurs")

        return self.df

    def diagnose_data(self):
        """Diagnose potential data issues that could cause singular matrix"""
        self.log_print("\n=== Data Diagnosis ===")

        # Check for perfect separation
        for outcome in ['pos', 'neg']:
            if outcome in self.df.columns:
                crosstab = pd.crosstab(self.df['source'], self.df[outcome])
                self.log_print(f"\n{outcome.upper()} by source distribution:")
                self.log_print(str(crosstab))

                # Check for zero cells (perfect separation)
                if (crosstab == 0).any().any():
                    self.log_print(f"⚠️  Warning: Zero cells detected in {outcome} crosstab")

                # Check for very low cell counts
                min_count = crosstab.min().min()
                if min_count < 5:
                    self.log_print(f"⚠️  Warning: Very low cell counts in {outcome} (min: {min_count})")

        # Check variable availability
        available_vars = []
        if 'total_tokens_scaled' in self.df.columns:
            available_vars.append('total_tokens_scaled')
        if 'TTR' in self.df.columns:
            available_vars.append('TTR')
        if 'topic' in self.df.columns:
            available_vars.append('topic')

        self.log_print(f"\nAvailable control variables: {available_vars}")

        # Check for multicollinearity among topics if they exist
        if 'topic' in self.df.columns:
            topic_counts = self.df['topic'].value_counts()
            self.log_print(f"Topic statistics: {len(topic_counts)} unique topics")
            self.log_print(
                f"Topic distribution: min={topic_counts.min()}, max={topic_counts.max()}, mean={topic_counts.mean():.1f}")

    def run_group_model(self, outcome='pos'):
        """
        Run logistic regression for group effects
        Progressive model fitting strategy, auto-downgrade when singular matrix occurs
        """
        self.log_print(f"\n=== {outcome.upper()} Sentiment Analysis ===")

        # Set reference group according to R logic
        if outcome == 'pos':
            reference_group = 'they'
            self.log_print("Reference group: 'they' (outgroup)")
        else:
            reference_group = 'we'
            self.log_print("Reference group: 'we' (ingroup)")

        # Create categorical source variable with specified reference
        self.df['source_cat'] = pd.Categorical(self.df['source'],
                                               categories=['we', 'they'])
        self.df['source_cat'] = self.df['source_cat'].cat.set_categories(
            [reference_group] + [cat for cat in ['we', 'they'] if cat != reference_group]
        )


        model_configs = [
            {
                'name': 'Full model',
                'formula': self._build_full_formula(outcome),
                'description': 'including all available variables'
            },
            {
                'name': 'No-topic model',
                'formula': self._build_no_topic_formula(outcome),
                'description': 'excluding topic variables to avoid excessive parameters'
            },
            {
                'name': 'Basic model',
                'formula': f"{outcome} ~ source_cat",
                'description': 'only including group variables'
            }
        ]

        for config in model_configs:
            try:
                self.log_print(f"\nTrying {config['name']}: {config['description']}")
                self.log_print(f"Model formula: {config['formula']}")

                model = smf.logit(config['formula'], data=self.df).fit(disp=0)
                self.results[outcome] = model

                self.log_print(f"✅ {config['name']} fitted successfully")
                self._print_model_results(model, outcome, reference_group)
                return model

            except np.linalg.LinAlgError as e:
                self.log_print(f"❌ {config['name']} singular matrix error: {str(e)[:100]}...")
                self.log_print(f"   Continuing to next model...")
                continue

            except Exception as e:
                self.log_print(f"❌ {config['name']} other error: {str(e)[:100]}...")
                self.log_print(f"   Continuing to next model...")
                continue

        self.log_print("❌ All models failed to fit, please check data quality")
        return None

    def _build_full_formula(self, outcome):
        """Build complete model formula including all variables"""
        formula = f"{outcome} ~ source_cat"

        if 'total_tokens_scaled' in self.df.columns:
            formula += " + total_tokens_scaled"
        if 'TTR' in self.df.columns:
            formula += " + TTR"
        if 'topic' in self.df.columns:
            formula += " + C(topic)"

        return formula

    def _build_no_topic_formula(self, outcome):
        """Build model formula without topic variables"""
        formula = f"{outcome} ~ source_cat"

        if 'total_tokens_scaled' in self.df.columns:
            formula += " + total_tokens_scaled"
        if 'TTR' in self.df.columns:
            formula += " + TTR"

        return formula

    def _print_model_results(self, model, outcome, reference_group):
        """Print model results"""
        # Find source coefficient
        source_param = None
        for param in model.params.index:
            if 'source_cat' in param and param != 'Intercept':
                source_param = param
                break

        if source_param:
            coef = model.params[source_param]
            se = model.bse[source_param]
            p_val = model.pvalues[source_param]
            odds_ratio = np.exp(coef)
            ci = np.exp(model.conf_int().loc[source_param])

            self.log_print(f"Coefficient: {coef:.4f} (SE: {se:.4f})")
            self.log_print(f"Odds Ratio: {odds_ratio:.4f} [95% CI: {ci[0]:.4f}-{ci[1]:.4f}]")
            self.log_print(f"P-value: {p_val:.4f}")

            # Interpret according to reference group
            if outcome == 'pos':
                if p_val < 0.05:
                    direction = "higher" if coef > 0 else "lower"
                    self.log_print(f"Result: Ingroup has {direction} positive sentiment than outgroup (p < 0.05)")
                else:
                    self.log_print(f"Result: No significant group difference (p = {p_val:.3f})")
            else:
                if p_val < 0.05:
                    direction = "higher" if coef > 0 else "lower"
                    self.log_print(f"Result: Outgroup has {direction} negative sentiment than ingroup (p < 0.05)")
                else:
                    self.log_print(f"Result: No significant group difference (p = {p_val:.3f})")
        else:
            self.log_print("⚠️  No source coefficient found in model")

    def print_summary(self):
        """Print simple summary of all results"""
        self.log_print("\n" + "=" * 50)
        self.log_print("SUMMARY: Group Effect Analysis")
        self.log_print("=" * 50)

        for outcome in ['pos', 'neg']:
            if outcome in self.results:
                model = self.results[outcome]

                # Find source coefficient
                source_param = None
                for param in model.params.index:
                    if 'source_cat' in param and param != 'Intercept':
                        source_param = param
                        break

                if source_param:
                    coef = model.params[source_param]
                    p_val = model.pvalues[source_param]
                    odds_ratio = np.exp(coef)

                    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                    self.log_print(f"{outcome.upper()}: OR = {odds_ratio:.3f}, p = {p_val:.3f} {sig}")

    def simple_plot(self, save_plot=True):
        """Create simple group comparison plot"""
        self.log_print("\nCreating visualization...")

        # Simple bar plot of sentiment rates
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # Positive sentiment
        pos_rates = self.df.groupby('source')['pos'].mean()
        ax1.bar(pos_rates.index, pos_rates.values, alpha=0.7, color=['skyblue', 'lightcoral'])
        ax1.set_title('Positive Sentiment Rate by Group')
        ax1.set_ylabel('Rate')
        ax1.set_ylim(0, 1)

        # Negative sentiment
        neg_rates = self.df.groupby('source')['neg'].mean()
        ax2.bar(neg_rates.index, neg_rates.values, alpha=0.7, color=['skyblue', 'lightcoral'])
        ax2.set_title('Negative Sentiment Rate by Group')
        ax2.set_ylabel('Rate')
        ax2.set_ylim(0, 1)

        plt.tight_layout()
        if save_plot:
            plot_path = f'../result/4.group_effects_{self.file_suffix}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.log_print(f"Plot saved to: {plot_path}")
        plt.show()


def main():
    """Main analysis function"""
    print("=== Group Effect Analysis ===")

    # Initialize - update this path to match your actual file
    analysis = GroupEffectAnalysis(
        data_path="../result/3.sentiment_data_cemotion_cemotion-chinese-2class_threshold_0_5.csv")

    # Run analysis
    analysis.load_data()
    analysis.prepare_data()

    # Test group effects for both outcomes
    analysis.run_group_model('pos')
    analysis.run_group_model('neg')

    # Summary and save log
    analysis.print_summary()
    analysis.simple_plot()
    analysis.save_log()

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
