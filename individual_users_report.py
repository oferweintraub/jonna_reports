import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown
import numpy as np

class UserAnalysisReport:
    def __init__(self):
        # Set basic style for seaborn
        sns.set_theme()
        
    def generate_report(self, period_results, test_users, batch_size=80):
        """
        Generate a comprehensive analysis report for users across different periods.
        
        Args:
            period_results (dict): Dictionary containing DataFrames with analysis results for each period
            test_users (list): List of usernames to analyze
            batch_size (int): Size of tweet batches used in analysis (not used after merging)
        """
        # Create the report
        report = []
        report.append("# Cross-Period Analysis Report\n")
        report.append("## Overview\n")
        report.append(f"Analysis period: Pre-war vs Post-war\n")
        report.append(f"Users analyzed: {', '.join(test_users)}\n")

        # Ensure the user_analysis directory exists
        os.makedirs('data/user_analysis', exist_ok=True)

        # Generate visualizations
        viz_path = self._generate_visualizations(period_results, test_users)
        
        # Generate detailed user analysis
        self._add_user_analysis(report, period_results, test_users)
        
        # Save and display report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join('data', 'user_analysis', f'analysis_report_{timestamp}.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        # Display the report in the notebook
        display(Markdown('\n'.join(report)))
        
        print(f"\nReport and visualizations saved to:")
        print(f"- Report: {report_path}")
        print(f"- Visualizations: {viz_path}")
        
        return report_path, viz_path

    def _generate_visualizations(self, period_results, test_users):
        """Generate and save visualization plots."""
        plt.figure(figsize=(15, 10))
        
        # 1. Toxicity Level Comparison
        plt.subplot(2, 2, 1)
        toxicity_data = []
        for period, df in period_results.items():
            for username in test_users:
                user_data = df[df['username'] == username]
                if not user_data.empty:
                    toxicity_data.append({
                        'Username': username,
                        'Period': period.replace('_', ' ').title(),
                        'Toxicity': user_data['toxicity_level'].mean()
                    })
        
        toxicity_df = pd.DataFrame(toxicity_data)
        sns.barplot(data=toxicity_df, x='Username', y='Toxicity', hue='Period')
        plt.title('Average Toxicity Levels by Period')
        plt.xticks(rotation=45)
        
        # 2. Tweet Volume Comparison (using tweet_count if available, otherwise just showing 1 per analysis)
        plt.subplot(2, 2, 2)
        volume_data = []
        for period, df in period_results.items():
            for username in test_users:
                user_data = df[df['username'] == username]
                if not user_data.empty:
                    # Use tweet_count if available, otherwise count rows
                    tweet_count = user_data['tweet_count'].iloc[0] if 'tweet_count' in user_data.columns else len(user_data)
                    volume_data.append({
                        'Username': username,
                        'Period': period.replace('_', ' ').title(),
                        'Tweets': tweet_count
                    })
        
        volume_df = pd.DataFrame(volume_data)
        sns.barplot(data=volume_df, x='Username', y='Tweets', hue='Period')
        plt.title('Number of Tweets by Period')
        plt.xticks(rotation=45)
        
        # Adjust layout and save
        plt.tight_layout()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        viz_path = os.path.join('data', 'user_analysis', f'analysis_visualization_{timestamp}.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        
        return viz_path

    def _add_user_analysis(self, report, period_results, test_users):
        """Add detailed analysis for each user to the report."""
        report.append("\n## Detailed Analysis by User\n")
        
        for username in test_users:
            report.append(f"\n### {username}\n")
            
            # Toxicity analysis
            pre_war_tox = period_results['pre_war'][period_results['pre_war']['username'] == username]['toxicity_level'].mean()
            post_war_tox = period_results['post_war'][period_results['post_war']['username'] == username]['toxicity_level'].mean()
            tox_change = post_war_tox - pre_war_tox
            
            report.append("#### Toxicity Analysis\n")
            report.append(f"- Pre-war average toxicity: {pre_war_tox:.2f}\n")
            report.append(f"- Post-war average toxicity: {post_war_tox:.2f}\n")
            report.append(f"- Change in toxicity: {tox_change:+.2f}\n")
            
            # Most toxic tweets analysis
            report.append("\n#### Most Toxic Tweets\n")
            for period in ['pre_war', 'post_war']:
                period_display = period.replace('_', ' ').title()
                user_data = period_results[period][period_results[period]['username'] == username]
                
                if not user_data.empty:
                    toxic_tweets = user_data.nlargest(2, 'toxicity_level')
                    report.append(f"\n**{period_display} Period Most Toxic Tweets:**\n")
                    for idx, (_, tweet) in enumerate(toxic_tweets.iterrows(), 1):
                        report.append(f"\n{idx}. **Toxicity Level: {tweet['toxicity_level']:.2f}**\n")
                        if 'tweet_text' in tweet:
                            report.append(f"   Tweet: {tweet['tweet_text']}\n")
                        if 'toxic_examples' in tweet and isinstance(tweet['toxic_examples'], str):
                            examples = eval(tweet['toxic_examples'])
                            if examples:
                                report.append(f"   Examples: {examples[0]}\n")
                        report.append("\n")  # Add space between tweets
            
            # Volume analysis
            pre_war_data = period_results['pre_war'][period_results['pre_war']['username'] == username]
            post_war_data = period_results['post_war'][period_results['post_war']['username'] == username]
            
            # Get total tweets analyzed from the original data
            pre_war_tweets = pre_war_data['num_tweets'].iloc[0] if 'num_tweets' in pre_war_data.columns else pre_war_data['tweet_count'].iloc[0]
            post_war_tweets = post_war_data['num_tweets'].iloc[0] if 'num_tweets' in post_war_data.columns else post_war_data['tweet_count'].iloc[0]
            
            report.append("\n#### Tweet Volume Analysis\n")
            report.append(f"- Pre-war tweets analyzed: {int(pre_war_tweets):,}\n")  # Format with commas for readability
            report.append(f"- Post-war tweets analyzed: {int(post_war_tweets):,}\n")
            report.append(f"- Total tweets analyzed: {int(pre_war_tweets + post_war_tweets):,}\n")
            
            # Narrative analysis
            self._add_narrative_analysis(report, period_results, username)

    def _add_narrative_analysis(self, report, period_results, username):
        """Add narrative analysis for a user to the report."""
        report.append("\n#### Narrative Analysis\n")
        for period in ['pre_war', 'post_war']:
            period_display = period.replace('_', ' ').title()
            user_data = period_results[period][period_results[period]['username'] == username]
            
            report.append(f"\n**{period_display} Period:**\n")
            
            if not user_data.empty:
                narratives = user_data['narratives'].iloc[0]
                attacked = user_data['attacked_entities'].iloc[0]
                protected = user_data['protected_entities'].iloc[0]
                
                report.append("*Main Narratives:*\n")
                if isinstance(narratives, str):
                    narratives = eval(narratives)
                for narr in narratives:
                    report.append(f"- {narr}\n")
                
                report.append("\n*Attacked Entities:*\n")
                if isinstance(attacked, str):
                    attacked = eval(attacked)
                for entity in attacked:
                    report.append(f"- {entity}\n")
                
                report.append("\n*Protected Entities:*\n")
                if isinstance(protected, str):
                    protected = eval(protected)
                for entity in protected:
                    report.append(f"- {entity}\n") 