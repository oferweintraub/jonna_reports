import pandas as pd
from datetime import datetime
import os
import re

class TweetCleaner:
    def __init__(self, min_words=7, remove_mentions=True, remove_urls=True):
        """
        Initialize TweetCleaner
        Args:
            min_words (int): Minimum number of words required in a tweet
            remove_mentions (bool): Whether to remove @mentions from tweets
            remove_urls (bool): Whether to remove URLs from tweets
        """
        self.min_words = min_words
        self.remove_mentions = remove_mentions
        self.remove_urls = remove_urls
        
        # Compile regex patterns for efficiency
        self.mention_pattern = re.compile(r'@\w+')
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    
    def _clean_text(self, text):
        """
        Clean a single text string by removing mentions and URLs if specified
        Args:
            text (str): Text to clean
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
            
        cleaned_text = text
        
        if self.remove_urls:
            cleaned_text = self.url_pattern.sub('', cleaned_text)
            
        if self.remove_mentions:
            cleaned_text = self.mention_pattern.sub('', cleaned_text)
        
        # Remove extra whitespace
        cleaned_text = ' '.join(cleaned_text.split())
        
        return cleaned_text
    
    def clean_tweets(self, input_data, period_label=None):
        """
        Clean tweets from input DataFrame or CSV file
        Args:
            input_data: Either a pandas DataFrame or a path to a CSV file
            period_label: Optional period label (e.g., 'pre_war', 'post_war') for organizing output
        Returns:
            DataFrame with cleaned tweets
        """
        # Handle input data
        if isinstance(input_data, str):
            print(f"Loading tweets from: {input_data}")
            df = pd.read_csv(input_data)
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            raise ValueError("Input must be either a DataFrame or a path to a CSV file")
        
        # Store original counts
        original_counts = df['author_username'].value_counts()
        
        # Clean text content first
        print("\nCleaning tweets...")
        if self.remove_urls:
            print("- Removing URLs")
        if self.remove_mentions:
            print("- Removing @mentions")
        
        df['cleaned_text'] = df['text'].apply(self._clean_text)
        
        # Count words after cleaning
        df['word_count'] = df['cleaned_text'].str.split().str.len()
        
        # Filter by minimum word count
        print(f"- Filtering tweets with less than {self.min_words} words")
        df_cleaned = df[df['word_count'] >= self.min_words].copy()
        
        # Replace original text with cleaned text
        df_cleaned['text'] = df_cleaned['cleaned_text']
        df_cleaned = df_cleaned.drop(['cleaned_text', 'word_count'], axis=1)
        
        # Get cleaned counts
        cleaned_counts = df_cleaned['author_username'].value_counts()
        
        # Print summary
        print("\nTweet counts before and after cleaning:")
        print("-" * 70)
        
        all_users = sorted(set(original_counts.index) | set(cleaned_counts.index))
        for user in all_users:
            orig = original_counts.get(user, 0)
            cleaned = cleaned_counts.get(user, 0)
            removed = orig - cleaned
            if orig > 0:
                removal_percent = (removed / orig) * 100
            else:
                removal_percent = 0
            print(f"{user:<20} - original: {orig:>4}, cleaned: {cleaned:>4} (removed: {removed:>4}, {removal_percent:>6.1f}%)")
        
        # Print overall statistics
        total_original = len(df)
        total_cleaned = len(df_cleaned)
        total_removed = total_original - total_cleaned
        total_percent = (total_removed / total_original * 100) if total_original > 0 else 0
        
        print("-" * 70)
        print(f"Total tweets - original: {total_original}, after cleaning: {total_cleaned}")
        print(f"Total removed: {total_removed} ({total_percent:.1f}%)")
        
        # Save cleaned data with period-specific organization
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Determine save directory based on period_label
        if period_label:
            save_dir = os.path.join('data', 'cleaned', period_label)
        else:
            save_dir = os.path.join('data', 'cleaned')
            
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Create filename with period label if provided
        if period_label:
            filename = f'cleaned_{period_label}_{timestamp}.csv'
        else:
            filename = f'cleaned_tweets_{timestamp}.csv'
            
        # Save the cleaned data
        filepath = os.path.join(save_dir, filename)
        df_cleaned.to_csv(filepath, index=False)
        print(f"\nSaved cleaned tweets to: {filepath}")
        
        return df_cleaned 