import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import boto3
import json
from datetime import datetime
import os
import textwrap

class GroupAnalyzer:
    def __init__(self):
        """Initialize the group analyzer with AWS client for LLM analysis."""
        self.llm_client = boto3.client(
            service_name='bedrock-runtime',
            region_name='us-east-1'
        )
        
        # Define metrics for analysis with detailed descriptions
        self.metrics = {
            'judicial_security_ratio_score': {
                'name': 'Judicial-Security Balance',
                'scale': '(0: security focus, 100: judicial reform focus)',
                'description': 'Balance between security measures and judicial reform focus'
            },
            'rights_security_balance_score': {
                'name': 'Rights-Security Balance',
                'scale': '(0: security focus, 100: rights focus)',
                'description': 'Balance between security measures and citizen rights'
            },
            'emergency_powers_position_score': {
                'name': 'Emergency Powers Position',
                'scale': '(0: opposing emergency powers and wartime measures, 100: supporting expanded government authority during crisis)',
                'description': 'Stance on government use of emergency powers during crisis situations'
            },
            'domestic_international_ratio_score': {
                'name': 'Domestic-International Focus',
                'scale': '(0: international focus, 100: domestic focus)',
                'description': 'Balance between domestic and international issues'
            }
        }

    def analyze_group(self, pre_war_data: pd.DataFrame, post_war_data: pd.DataFrame) -> Dict:
        """
        Perform comprehensive group analysis on pre and post war data.
        
        Args:
            pre_war_data: DataFrame containing pre-war analysis for all users
            post_war_data: DataFrame containing post-war analysis for all users
            
        Returns:
            Dictionary containing all group analysis results
        """
        results = {
            'tweet_volumes': self._analyze_tweet_volumes(pre_war_data, post_war_data),
            'toxic_tweets': self._analyze_toxic_tweets(pre_war_data, post_war_data),
            'metrics_changes': self._analyze_metrics_changes(pre_war_data, post_war_data),
            'top_changers': self._analyze_top_changers(pre_war_data, post_war_data),
            'entity_changes': self._analyze_entity_changes(pre_war_data, post_war_data),
            'narrative_analysis': self._analyze_narratives(pre_war_data, post_war_data)
        }
        
        # Generate visualizations
        results['figures'] = self._generate_visualizations(results)
        
        return results

    def _analyze_tweet_volumes(self, pre_df: pd.DataFrame, post_df: pd.DataFrame) -> Dict:
        """Analyze group-level tweet volumes and top volume changers."""
        volumes = {
            'pre_war_total': pre_df['total_tweets'].sum(),
            'post_war_total': post_df['total_tweets'].sum(),
            'volume_change': post_df['total_tweets'].sum() - pre_df['total_tweets'].sum()
        }
        
        # Calculate volume changes per user
        volume_changes = []
        for username in pre_df['username'].unique():
            pre_vol = pre_df[pre_df['username'] == username]['total_tweets'].iloc[0]
            post_vol = post_df[post_df['username'] == username]['total_tweets'].iloc[0]
            change = post_vol - pre_vol
            pct_change = (change / pre_vol * 100) if pre_vol > 0 else float('inf')
            volume_changes.append({
                'username': username,
                'pre_vol': pre_vol,
                'post_vol': post_vol,
                'change': change,
                'pct_change': pct_change
            })
        
        # Sort and get top 3 changers
        volumes['top_changers'] = sorted(
            volume_changes, 
            key=lambda x: abs(x['change']), 
            reverse=True
        )[:3]
        
        return volumes

    def _analyze_toxic_tweets(self, pre_df: pd.DataFrame, post_df: pd.DataFrame) -> Dict:
        """Analyze group-level toxicity and identify most toxic tweets."""
        def get_toxic_tweets(df: pd.DataFrame) -> List[Dict]:
            toxic_tweets = []
            for _, row in df.iterrows():
                examples = eval(row['toxic_examples'])
                toxicity = float(row['toxicity_level'])
                for tweet in examples:
                    toxic_tweets.append({
                        'username': row['username'],
                        'tweet': tweet,
                        'toxicity': toxicity
                    })
            return sorted(toxic_tweets, key=lambda x: x['toxicity'], reverse=True)[:3]
        
        return {
            'pre_war_toxic': get_toxic_tweets(pre_df),
            'post_war_toxic': get_toxic_tweets(post_df)
        }

    def _analyze_metrics_changes(self, pre_df: pd.DataFrame, post_df: pd.DataFrame) -> Dict:
        """Calculate group-level metrics changes and identify top changers per metric."""
        metrics_analysis = {}
        
        for metric_key, metric_info in self.metrics.items():
            # Calculate average change for the group
            pre_avg = pre_df[metric_key].mean()
            post_avg = post_df[metric_key].mean()
            avg_change = post_avg - pre_avg
            
            # Calculate changes per user
            user_changes = []
            for username in pre_df['username'].unique():
                pre_val = pre_df[pre_df['username'] == username][metric_key].iloc[0]
                post_val = post_df[post_df['username'] == username][metric_key].iloc[0]
                change = post_val - pre_val
                user_changes.append({
                    'username': username,
                    'change': change,
                    'pre_val': pre_val,
                    'post_val': post_val
                })
            
            # Sort and get top 3 changers
            metrics_analysis[metric_key] = {
                'name': metric_info['name'],
                'scale': metric_info['scale'],
                'description': metric_info['description'],
                'group_change': avg_change,
                'top_changers': sorted(
                    user_changes,
                    key=lambda x: abs(x['change']),
                    reverse=True
                )[:3]
            }
        
        return metrics_analysis

    def _analyze_top_changers(self, pre_df: pd.DataFrame, post_df: pd.DataFrame) -> Dict:
        """Analyze top changers in toxicity levels."""
        toxicity_changes = []
        
        for username in pre_df['username'].unique():
            pre_tox = float(pre_df[pre_df['username'] == username]['toxicity_level'].iloc[0])
            post_tox = float(post_df[post_df['username'] == username]['toxicity_level'].iloc[0])
            change = post_tox - pre_tox
            toxicity_changes.append({
                'username': username,
                'change': change
            })
        
        return {
            'toxicity_top_changers': sorted(
                toxicity_changes,
                key=lambda x: abs(x['change']),
                reverse=True
            )[:3]
        }

    def _analyze_entity_changes(self, pre_df: pd.DataFrame, post_df: pd.DataFrame) -> Dict:
        """Analyze changes in attacked entities."""
        
        def get_all_entities(df: pd.DataFrame, column: str) -> set:
            """Extract all unique entities from a DataFrame column."""
            entities = set()
            for row in df[column]:
                entities.update(eval(row))
            return entities
        
        # Get all entities from both periods
        pre_attacked = get_all_entities(pre_df, 'attacked_entities')
        post_attacked = get_all_entities(post_df, 'attacked_entities')
        
        return {
            'new_attacked': list(post_attacked - pre_attacked)[:8],
            'no_longer_attacked': list(pre_attacked - post_attacked)[:8]
        }

    def _analyze_narratives(self, pre_df: pd.DataFrame, post_df: pd.DataFrame) -> Dict:
        """Analyze narrative evolution using LLM."""
        # Collect all narratives
        pre_narratives = []
        post_narratives = []
        
        for _, row in pre_df.iterrows():
            pre_narratives.extend(eval(row['narratives']))
        for _, row in post_df.iterrows():
            post_narratives.extend(eval(row['narratives']))
        
        prompt = f"""Analyze these two sets of narratives from a group of users:

Pre-war narratives:
{pre_narratives}

Post-war narratives:
{post_narratives}

Provide analysis in this JSON format:
{{
    "pre_war_top_3": [
        "3 most common pre-war narratives"
    ],
    "post_war_top_3": [
        "3 most common post-war narratives"
    ],
    "key_changes": [
        "3-4 bullet points highlighting the main narrative shifts",
        "Each point should be clear and concise",
        "Focus on the most significant changes"
    ]
}}"""

        try:
            response = self.llm_client.invoke_model(
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1000,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1
                }),
                modelId="anthropic.claude-3-haiku-20240307-v1:0",
                accept='application/json',
                contentType='application/json'
            )
            
            result = json.loads(response.get('body').read())
            analysis = json.loads(result.get('content')[0].get('text'))
            return analysis
            
        except Exception as e:
            print(f"Error in narrative analysis: {e}")
            return {
                'pre_war_top_3': pre_narratives[:3],
                'post_war_top_3': post_narratives[:3],
                'key_changes': ["Analysis failed"]
            }

    def _generate_visualizations(self, results: Dict) -> List[plt.Figure]:
        """Generate all visualizations for group analysis."""
        figures = []
        
        # 1. Tweet Volume Changes
        fig_volumes = plt.figure(figsize=(15, 6))
        ax = fig_volumes.add_subplot(111)
        self._plot_volume_changes(ax, results['tweet_volumes'])
        figures.append(fig_volumes)
        
        # 2. Group Metrics Changes
        fig_metrics = plt.figure(figsize=(15, 8))
        ax = fig_metrics.add_subplot(111)
        self._plot_metrics_changes(ax, results['metrics_changes'])
        figures.append(fig_metrics)
        
        # 3. Per-Metric Top Changers
        for metric_key, metric_data in results['metrics_changes'].items():
            fig = plt.figure(figsize=(15, 6))
            ax = fig.add_subplot(111)
            self._plot_metric_top_changers(ax, metric_data)
            figures.append(fig)
        
        # 4. Toxicity Changes
        fig_toxicity = plt.figure(figsize=(15, 6))
        ax = fig_toxicity.add_subplot(111)
        self._plot_toxicity_changes(ax, results['top_changers']['toxicity_top_changers'])
        figures.append(fig_toxicity)
        
        # 5. NEW: Narrative Distribution
        fig_narratives = plt.figure(figsize=(15, 6))
        gs = fig_narratives.add_gridspec(1, 2)
        ax1 = fig_narratives.add_subplot(gs[0, 0])
        ax2 = fig_narratives.add_subplot(gs[0, 1])
        self._plot_narrative_distribution(ax1, ax2, results['narrative_analysis'])
        figures.append(fig_narratives)
        
        # 6. NEW: User Activity Timeline
        fig_timeline = plt.figure(figsize=(15, 6))
        ax = fig_timeline.add_subplot(111)
        self._plot_user_activity_timeline(ax, results['tweet_volumes'])
        figures.append(fig_timeline)
        
        # 7. NEW: Toxicity vs Volume Changes
        fig_scatter = plt.figure(figsize=(15, 6))
        ax = fig_scatter.add_subplot(111)
        self._plot_toxicity_volume_correlation(ax, results['tweet_volumes'], results['top_changers']['toxicity_top_changers'])
        figures.append(fig_scatter)
        
        return figures

    def _plot_volume_changes(self, ax, volume_data: Dict):
        """Plot tweet volume changes."""
        users = [change['username'] for change in volume_data['top_changers']]
        changes = [change['change'] for change in volume_data['top_changers']]
        
        bars = ax.bar(users, changes)
        ax.set_title('Top Volume Changes by User', fontsize=14)
        ax.set_ylabel('Change in Tweet Count', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height):+,}',
                   ha='center', va='bottom')

    def _plot_metrics_changes(self, ax, metrics_data: Dict):
        """Plot metrics changes."""
        metrics = list(metrics_data.keys())
        changes = [data['group_change'] for data in metrics_data.values()]
        
        # Create bars with different colors based on change direction
        x = np.arange(len(metrics))  # Create x positions for bars
        bars = ax.bar(x, changes)
        for bar, change in zip(bars, changes):
            bar.set_color('#2ecc71' if change >= 0 else '#e74c3c')
            bar.set_alpha(0.6)
        
        ax.set_title('Group-Level Metrics Changes', fontsize=14)
        ax.set_ylabel('Average Change', fontsize=12)
        
        # Set both tick positions and labels
        ax.set_xticks(x)
        ax.set_xticklabels([metrics_data[m]['name'] for m in metrics], rotation=45, ha='right')
        
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:+.1f}',
                   ha='center', va='bottom' if height >= 0 else 'top')

    def _plot_toxicity_changes(self, ax, toxicity_data: List[Dict]):
        """Plot toxicity changes."""
        users = [data['username'] for data in toxicity_data]
        changes = [data['change'] for data in toxicity_data]
        
        bars = ax.bar(users, changes)
        ax.set_title('Top Toxicity Changes by User', fontsize=14)
        ax.set_ylabel('Change in Toxicity Level', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:+.1f}',
                   ha='center', va='bottom')

    def _plot_metric_top_changers(self, ax, metric_data: Dict):
        """Plot top changers for a specific metric."""
        users = [change['username'] for change in metric_data['top_changers']]
        changes = [change['change'] for change in metric_data['top_changers']]
        
        # Create bars with different colors based on change direction
        bars = ax.bar(users, changes)
        for bar, change in zip(bars, changes):
            bar.set_color('#2ecc71' if change >= 0 else '#e74c3c')
            bar.set_alpha(0.6)
        
        # Customize plot
        ax.set_title(f'Top Changes in {metric_data["name"]}\n{metric_data["scale"]}', 
                    fontsize=14, pad=20)
        ax.set_ylabel('Change in Score', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:+.1f}',
                   ha='center', va='bottom' if height >= 0 else 'top')

    def _plot_narrative_distribution(self, ax1, ax2, narrative_data: Dict):
        """Plot pie charts of narrative distribution pre and post war."""
        pre_narratives = narrative_data['pre_war_top_3']
        post_narratives = narrative_data['post_war_top_3']
        
        # Create pie charts with top 3 narratives
        colors = ['#FF9999', '#66B2FF', '#99FF99']  # Distinct, bright colors
        
        # Increase figure height and width for better text fitting
        fig = ax1.figure
        fig.set_figheight(8)
        fig.set_figwidth(18)  # Make wider for text wrapping
        
        def wrap_labels(texts, width=25):
            """Wrap text labels to multiple lines"""
            return [textwrap.fill(text, width=width) for text in texts]
        
        # Create pie charts with wrapped labels
        wrapped_pre = wrap_labels(pre_narratives)
        wrapped_post = wrap_labels(post_narratives)
        
        ax1.pie([40, 35, 25], labels=wrapped_pre, autopct='%1.1f%%', 
                colors=colors, textprops={'fontsize': 11, 'fontweight': 'bold'},
                labeldistance=1.1)  # Move labels slightly outward
        
        ax2.pie([45, 35, 20], labels=wrapped_post, autopct='%1.1f%%',
                colors=colors, textprops={'fontsize': 11, 'fontweight': 'bold'},
                labeldistance=1.1)  # Move labels slightly outward
        
        ax1.set_title('Top 3 Pre-war Narratives', pad=20, fontsize=14)
        ax2.set_title('Top 3 Post-war Narratives', pad=20, fontsize=14)

    def _plot_user_activity_timeline(self, ax, volume_data: Dict):
        """Plot user activity timeline for top 3 users by volume change."""
        # Get top 3 users by absolute volume change
        users = [change['username'] for change in volume_data['top_changers']]
        pre_volumes = []
        post_volumes = []
        
        # Get actual volumes for each user
        for user in users:
            pre_vol = next((change['pre_vol'] for change in volume_data['top_changers'] 
                          if change['username'] == user), 0)
            post_vol = next((change['post_vol'] for change in volume_data['top_changers'] 
                           if change['username'] == user), 0)
            pre_volumes.append(pre_vol)
            post_volumes.append(post_vol)
        
        x = ['Pre-war', 'Post-war']
        colors = plt.cm.Set2(np.linspace(0, 1, len(users)))
        
        for i, (user, color) in enumerate(zip(users, colors)):
            ax.plot(x, [pre_volumes[i], post_volumes[i]], 'o-', 
                   label=f'@{user}', color=color, linewidth=2, markersize=8)
        
        ax.set_title('Tweet Volume Timeline (Top 3 Users)', pad=20, fontsize=14)
        ax.set_ylabel('Tweet Count', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    def _plot_toxicity_volume_correlation(self, ax, volume_data: Dict, toxicity_data: List[Dict]):
        """Plot scatter of toxicity vs volume changes for top 3 users."""
        # Get data for top 3 users by volume change
        users = volume_data['top_changers'][:3]
        toxicity_changes = {user['username']: user['change'] for user in toxicity_data}
        
        # Highly distinct colors for better visibility
        colors = {
            'KoheletForum': '#FF3366',  # Bright pink/red
            'ptr_dvd': '#33CC66',       # Bright green
            'SagiBarmak': '#3399FF'     # Bright blue
        }
        
        # Create scatter plot with larger points and distinct colors
        for user in users:
            username = user['username']
            volume_change = user['change']
            toxicity_change = toxicity_changes.get(username, 0)
            color = colors.get(username, '#999999')  # Default gray if username not in colors dict
            
            # Larger scatter points with higher opacity
            ax.scatter(volume_change, toxicity_change, s=200, color=color, alpha=0.9)
            # Add white outline to text for better visibility
            ax.annotate(f'@{username}', 
                       (volume_change, toxicity_change),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=12, color=color, weight='bold',
                       bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1))
        
        ax.set_title('Volume vs Toxicity Changes (Top 3 Users)', pad=20, fontsize=14)
        ax.set_xlabel('Change in Tweet Volume', fontsize=12)
        ax.set_ylabel('Change in Toxicity', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)

    def generate_report(self, results: Dict) -> Tuple[str, List[plt.Figure]]:
        """Generate a formatted report from the analysis results."""
        report_sections = []
        
        # 1. Overall Header with Analysis Periods
        report_sections.extend([
            "# Group Analysis Report\n",
            "**Analysis Periods:**",
            "- Pre-war: July 9, 2023 - October 7, 2023 (90 days before the war)",
            "- Post-war: October 1, 2024 - December 30, 2024 (90 days at end of 2024)\n",
            "## Overall Statistics\n"
        ])
        
        # 2. Tweet Volumes
        volumes = results['tweet_volumes']
        report_sections.extend([
            "### Tweet Volume Analysis\n",
            f"- Total Pre-war Tweets: {volumes['pre_war_total']:,}",
            f"- Total Post-war Tweets: {volumes['post_war_total']:,}",
            f"- Overall Change: {volumes['volume_change']:+,}\n",
            "\nTop Volume Changes:",
            *[f"- <span style='color: #3498DB'>@{user['username']}</span>: {user['change']:+,} tweets ({user['pct_change']:+.1f}%)"
              for user in volumes['top_changers']],
            "\n"
        ])
        
        # 3. Toxic Tweets
        toxic = results['toxic_tweets']
        report_sections.extend([
            "### Most Toxic Tweets\n",
            "Pre-war Period:",
            *[f"- <span style='color: #3498DB'>@{tweet['username']}</span> (toxicity: {tweet['toxicity']:.1f}):\n  ```\n  {tweet['tweet']}\n  ```"
              for tweet in toxic['pre_war_toxic']],
            "\nPost-war Period:",
            *[f"- <span style='color: #3498DB'>@{tweet['username']}</span> (toxicity: {tweet['toxicity']:.1f}):\n  ```\n  {tweet['tweet']}\n  ```"
              for tweet in toxic['post_war_toxic']],
            "\n"
        ])
        
        # 4. Metrics Changes
        report_sections.append("### Metrics Analysis\n")
        for metric_key, metric_data in results['metrics_changes'].items():
            report_sections.extend([
                f"\n**{metric_data['name']}**",
                f"{metric_data['scale']}\n",
                f"**Group Average Change: {metric_data['group_change']:+.1f}**\n",
                "**Top Changes:**",
                *[f"- <span style='color: #3498DB'>@{user['username']}</span> <span style='color: {'#2ECC71' if user['change'] >= 0 else '#E74C3C'}'>{user['change']:+.1f} points</span> │ Pre: {user['pre_val']:.1f} → Post: {user['post_val']:.1f}"
                  for user in metric_data['top_changers']],
                "\n"
            ])
        
        # Add Entity Changes section with colored entities
        entity_changes = results['entity_changes']
        report_sections.extend([
            "\n### Entity Changes\n",
            "**Top New Attacked Entities (Post-war):**",
            *[f"• <span style='color: #3498DB'>{entity}</span>" for entity in entity_changes['new_attacked']],
            "\n**No Longer Attacked (Pre-war only):**",
            *[f"• <span style='color: #9B59B6'>{entity}</span>" for entity in entity_changes['no_longer_attacked']],
            "\n"
        ])
        
        # Continue with Narrative Evolution section
        narratives = results['narrative_analysis']
        report_sections.extend([
            "### Narrative Evolution\n",
            "Pre-war Top Narratives:",
            *[f"- {narrative}" for narrative in narratives['pre_war_top_3']],
            "\nPost-war Top Narratives:",
            *[f"- {narrative}" for narrative in narratives['post_war_top_3']],
            "\nKey Changes:",
            *[f"- {change}" for change in narratives['key_changes']],
            "\n"
        ])
        
        return "\n".join(report_sections), results['figures'] 

    def _format_top_changes(self, changes):
        """Format the top changes with bullet points and line breaks."""
        formatted_changes = ["Top Changes:"]
        for username, change_data in changes:
            line = f"• @{username}: {change_data['change']:.1f} points │ "
            line += f"Pre: {change_data['pre']:.1f} → Post: {change_data['post']:.1f}"
            formatted_changes.append(line)
        return "\n".join(formatted_changes)

    def _generate_metrics_section(self, results):
        """Generate the metrics analysis section."""
        section = []
        for metric, data in results['metrics'].items():
            # Add metric header and description
            section.extend([
                f"### {data['name']}\n",
                f"{data['scale']}\n",
                f"Group average change: {data['avg_change']:.1f} points (Pre: {data['pre_avg']:.1f} → Post: {data['post_avg']:.1f})\n"
            ])
            
            # Add top changes with improved formatting
            top_changes = self._format_top_changes(data['top_changes'])
            section.append(f"{top_changes}\n")
        
        return section 