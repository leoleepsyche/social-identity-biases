import pandas as pd
import numpy as np
from bertopic import BERTopic
import matplotlib.pyplot as plt

# Configure matplotlib to support Chinese characters (optional)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class TopicModeling:
    def __init__(self, data_path="../result/1.group_data.csv"):
        """Initialize the topic modeling class"""
        self.data_path = data_path
        self.df = None
        self.topic_model = None
        self.topics = None
        self.probs = None

    def load_data(self):
        """Load the prepared data"""
        print("Loading data...")
        self.df = pd.read_csv(self.data_path)
        print(f"Data loaded. Total records: {len(self.df)}")
        return self.df

    def perform_topic_modeling(self, language="chinese (simplified)", min_topic_size=10):
        print("Performing topic modeling...")
        if self.df is None:
            self.load_data()

        # Create a BERTopic model
        self.topic_model = BERTopic(
            language=language,
            min_topic_size=min_topic_size,
            calculate_probabilities=True
        )

        # Fit the model and extract topics
        texts = self.df['text'].tolist()
        self.topics, self.probs = self.topic_model.fit_transform(texts)

        # Add topics to the DataFrame
        self.df['topic'] = self.topics
        self.df['topic_probability'] = [max(prob) for prob in self.probs]

        print(f"Topic modeling completed. Total topics identified: {len(set(self.topics))}")
        return self.topics, self.probs

    def save_enhanced_data(self, output_path="../result/2.topic_data.csv"):
        """Save the DataFrame with topic information"""
        if self.df is None or 'topic' not in self.df.columns:
            raise ValueError("Please run topic modeling first.")

        # Keep only rows with valid (non-noise) topics (topic > 0)
        valid_df = self.df[self.df['topic'] > 0].copy()
        valid_df.to_csv(output_path, index=False, encoding='utf-8')

        print(f"Enhanced data saved to: {output_path}")
        print(f"Number of valid records: {len(valid_df)} (excluding noise topics)")

        return output_path


def main():
    """Main function - run topic modeling and update data"""
    # Create an instance of the topic modeling class
    tm = TopicModeling()
    # Execute the topic modeling workflow
    tm.load_data()
    tm.perform_topic_modeling()
    # Save results
    tm.save_enhanced_data()

    print("\nTopic modeling analysis completed!")


if __name__ == "__main__":
    main()
