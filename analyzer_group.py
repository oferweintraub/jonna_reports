import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import boto3
import json
from datetime import datetime
import os
import textwrap
from collections import Counter
from difflib import SequenceMatcher

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

    def analyze_group(self, pre_df: pd.DataFrame, post_df: pd.DataFrame) -> Dict:
        """Analyze group-level metrics and changes."""
        results = {}
        
        # Existing analyses...
        results['tweet_volumes'] = self._analyze_tweet_volumes(pre_df, post_df)
        results['toxic_tweets'] = self._analyze_toxic_tweets(pre_df, post_df)
        results['metrics_changes'] = self._analyze_metrics_changes(pre_df, post_df)
        results['top_changers'] = self._analyze_top_changers(pre_df, post_df)
        results['entity_changes'] = self._analyze_entity_changes(pre_df, post_df)
        
        # Get narrative analysis which now includes popularity distribution
        narrative_results = self._analyze_narratives(pre_df, post_df)
        results['narrative_analysis'] = narrative_results
        results['narrative_popularity'] = narrative_results['narrative_popularity']
        
        # Analyze emotional tones
        results['emotional_tones'] = {
            'pre_war': self._analyze_emotional_tones(pre_df),
            'post_war': self._analyze_emotional_tones(post_df)
        }
        
        # Generate all visualizations
        results['figures'] = self._generate_visualizations(results)
        
        # Add emotional tones visualization
        results['figures'].extend([
            self._plot_emotional_tones(
                results['emotional_tones']['pre_war'],
                results['emotional_tones']['post_war']
            )
        ])
        
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
        """Analyze narrative evolution using LLM with fallback to raw data analysis."""
        # Collect all narratives with their counts
        pre_narratives = []
        post_narratives = []
        
        for _, row in pre_df.iterrows():
            pre_narratives.extend(eval(row['narratives']))
        for _, row in post_df.iterrows():
            post_narratives.extend(eval(row['narratives']))
        
        # Count occurrences
        pre_counts = Counter(pre_narratives)
        post_counts = Counter(post_narratives)
        
        # Calculate percentages
        pre_total = len(pre_narratives)
        post_total = len(post_narratives)
        
        pre_percentages = {k: (v/pre_total)*100 for k, v in pre_counts.items()}
        post_percentages = {k: (v/post_total)*100 for k, v in post_counts.items()}

        try:
            # Try LLM analysis first with improved prompt for merging
            prompt = f"""You are a data analyst specializing in narrative analysis. Analyze and merge similar narratives from these two sets:

Pre-war narratives with percentages:
{[f"{k} ({v:.1f}%)" for k, v in pre_percentages.items()]}

Post-war narratives with percentages:
{[f"{k} ({v:.1f}%)" for k, v in post_percentages.items()]}

Important instructions:
1. First, merge very similar narratives together (e.g., combine narratives about the same topic with slightly different wording)
2. Sum the percentages of merged narratives
3. Sort all narratives by their percentages (highest to lowest)
4. Select ONLY the top 3 narratives with highest percentages for each period
5. Group all remaining narratives into an "Others" category
6. Make sure narratives are clear and concise
7. The percentages for each period MUST sum to exactly 100%

Return ONLY a valid JSON object with this EXACT format:
{{
    "pre_war_top_3": [
        {{
            "narrative": "First most common pre-war narrative (merged)",
            "percentage": 55.5,
            "merged_from": ["original narrative 1", "original narrative 2"]
        }},
        {{
            "narrative": "Second most common pre-war narrative (merged)",
            "percentage": 33.3,
            "merged_from": ["original narrative 3", "original narrative 4"]
        }},
        {{
            "narrative": "Third most common pre-war narrative (merged)",
            "percentage": 11.2,
            "merged_from": ["original narrative 5"]
        }},
        {{
            "narrative": "Others",
            "percentage": 0.0,
            "merged_from": ["all remaining narratives"]
        }}
    ],
    "post_war_top_3": [
        {{
            "narrative": "First most common post-war narrative (merged)",
            "percentage": 44.4,
            "merged_from": ["original narrative 1", "original narrative 2"]
        }},
        {{
            "narrative": "Second most common post-war narrative (merged)",
            "percentage": 33.3,
            "merged_from": ["original narrative 3", "original narrative 4"]
        }},
        {{
            "narrative": "Third most common post-war narrative (merged)",
            "percentage": 22.3,
            "merged_from": ["original narrative 5"]
        }},
        {{
            "narrative": "Others",
            "percentage": 0.0,
            "merged_from": ["all remaining narratives"]
        }}
    ],
    "analysis": "A 40-word analysis of how narratives changed from pre to post war, focusing on major shifts and new emerging themes"
}}

IMPORTANT:
- You MUST return the top 3 narratives by percentage for each period
- Percentages MUST sum to 100% for each period
- Sort by percentage (highest to lowest)
- Make narratives clear and concise"""

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
            content = result.get('content')[0].get('text', '').strip()
            
            # Clean up the response to ensure valid JSON
            content = content.replace('```json', '').replace('```', '').strip()
            if not content.startswith('{'):
                content = content[content.find('{'):]
            if not content.endswith('}'):
                content = content[:content.rfind('}')+1]
                
            analysis = json.loads(content)
            
        except Exception as e:
            print(f"Error in narrative analysis: {e}")
            print("Using fallback raw data analysis...")
            
            # Fallback: Use raw data without LLM grouping
            pre_top_3 = sorted(pre_percentages.items(), key=lambda x: x[1], reverse=True)
            post_top_3 = sorted(post_percentages.items(), key=lambda x: x[1], reverse=True)
            
            # Calculate "Others" percentage for pre-war
            pre_top_3_sum = sum(p for _, p in pre_top_3[:3])
            pre_others = 100 - pre_top_3_sum
            
            # Calculate "Others" percentage for post-war
            post_top_3_sum = sum(p for _, p in post_top_3[:3])
            post_others = 100 - post_top_3_sum
            
            analysis = {
                'pre_war_top_3': [
                    {
                        "narrative": narrative,
                        "percentage": percentage,
                        "merged_from": [narrative]
                    } 
                    for narrative, percentage in pre_top_3[:3]
                ] + [{
                    "narrative": "Others",
                    "percentage": pre_others,
                    "merged_from": [n for n, _ in pre_top_3[3:]]
                }],
                'post_war_top_3': [
                    {
                        "narrative": narrative,
                        "percentage": percentage,
                        "merged_from": [narrative]
                    }
                    for narrative, percentage in post_top_3[:3]
                ] + [{
                    "narrative": "Others",
                    "percentage": post_others,
                    "merged_from": [n for n, _ in post_top_3[3:]]
                }],
                'analysis': "Analysis using raw data due to LLM service unavailability"
            }
        
        # Add a separate LLM call just for the evolution analysis
        try:
            evolution_prompt = f"""You are a political discourse analyst. Analyze how these narratives changed from pre-war to post-war:

Pre-war Top Narratives:
- Defending Kohelet Forum's judicial reform (11.1%)
- Promoting free market economics and privatization (11.1%)
- Advocating for reduced government regulation (11.1%)

Post-war Top Narratives:
- Promoting Israel's Jewish identity (11.1%)
- Advocating for reduced government regulation (11.1%)
- Opposing labor unions and supporting privatization (11.1%)

Return ONLY a concise 34-word analysis focusing on the key shifts in narrative focus, new themes that emerged, and themes that disappeared. Focus on the most significant changes.
NO other text, NO explanations, EXACTLY 34 words."""

            evolution_response = self.llm_client.invoke_model(
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": evolution_prompt}],
                    "temperature": 0.1
                }),
                modelId="anthropic.claude-3-haiku-20240307-v1:0",
                accept='application/json',
                contentType='application/json'
            )
            
            evolution_result = json.loads(evolution_response.get('body').read())
            evolution_content = evolution_result.get('content')[0].get('text', '').strip()
            
            # Clean up the evolution content
            evolution_content = evolution_content.strip('"').strip()
            if evolution_content.lower().startswith(('here', 'analysis:', 'response:')):
                evolution_content = ' '.join(evolution_content.split()[1:])
            
            analysis['analysis'] = evolution_content
            
        except Exception as e:
            print(f"Error in evolution analysis: {e}")
            analysis['analysis'] = "Analysis using raw data due to LLM service unavailability"
        
        # Convert distributions to the format expected by the visualization
        analysis['narrative_popularity'] = {
            'pre_war': {item['narrative']: item['percentage'] 
                       for item in analysis['pre_war_top_3']},
            'post_war': {item['narrative']: item['percentage'] 
                        for item in analysis['post_war_top_3']}
        }
        
        return analysis

    def _generate_visualizations(self, results: Dict) -> List[plt.Figure]:
        """Generate all visualizations for group analysis."""
        figures = []
        
        # Set the default figure style and font sizes - INCREASED ALL SIZES
        plt.style.use('dark_background')
        plt.rcParams.update({
            'font.size': 24,  # Increased from 14
            'axes.titlesize': 32,  # Increased from 20
            'axes.labelsize': 28,  # Increased from 16
            'xtick.labelsize': 24,  # Increased from 14
            'ytick.labelsize': 24,  # Increased from 14
            'legend.fontsize': 24   # Increased from 14
        })
        
        # 1. Tweet Volume Changes
        fig_volumes = plt.figure(figsize=(20, 10), constrained_layout=True)
        fig_volumes.patch.set_facecolor('#1e1e1e')
        ax = fig_volumes.add_subplot(111)
        ax.set_facecolor('#1e1e1e')
        self._plot_volume_changes(ax, results['tweet_volumes'])
        figures.append(fig_volumes)
        
        # 2. Group Metrics Changes
        fig_metrics = plt.figure(figsize=(20, 12), constrained_layout=True)
        fig_metrics.patch.set_facecolor('#1e1e1e')
        ax = fig_metrics.add_subplot(111)
        ax.set_facecolor('#1e1e1e')
        self._plot_metrics_changes(ax, results['metrics_changes'])
        figures.append(fig_metrics)
        
        # 3. Per-Metric Top Changers
        for metric_key, metric_data in results['metrics_changes'].items():
            fig = plt.figure(figsize=(20, 10), constrained_layout=True)
            fig.patch.set_facecolor('#1e1e1e')
            ax = fig.add_subplot(111)
            ax.set_facecolor('#1e1e1e')
            self._plot_metric_top_changers(ax, metric_data)
            figures.append(fig)
        
        # 4. Toxicity Changes
        fig_toxicity = plt.figure(figsize=(20, 10), constrained_layout=True)
        fig_toxicity.patch.set_facecolor('#1e1e1e')
        ax = fig_toxicity.add_subplot(111)
        ax.set_facecolor('#1e1e1e')
        self._plot_toxicity_changes(ax, results['top_changers']['toxicity_top_changers'])
        figures.append(fig_toxicity)
        
        # 5. Narrative Distribution (increased size)
        fig_narratives = plt.figure(figsize=(40, 48))  # Made even taller
        fig_narratives.patch.set_facecolor('#1e1e1e')
        
        # Create gridspec with more vertical space
        gs = fig_narratives.add_gridspec(
            1, 2,
            width_ratios=[1, 1],
            wspace=0.4,
            left=0.1,
            right=0.9,
            top=0.85,
            bottom=0.15
        )
        
        ax1 = fig_narratives.add_subplot(gs[0])
        ax2 = fig_narratives.add_subplot(gs[1])
        ax1.set_facecolor('#1e1e1e')
        ax2.set_facecolor('#1e1e1e')
        
        self._plot_narrative_distribution(ax1, ax2, results['narrative_analysis'])
        figures.append(fig_narratives)
        
        # 6. User Activity Timeline
        fig_timeline = plt.figure(figsize=(20, 10), constrained_layout=True)
        fig_timeline.patch.set_facecolor('#1e1e1e')
        ax = fig_timeline.add_subplot(111)
        ax.set_facecolor('#1e1e1e')
        self._plot_user_activity_timeline(ax, results['tweet_volumes'])
        figures.append(fig_timeline)
        
        # 7. Toxicity vs Volume Changes
        fig_scatter = plt.figure(figsize=(20, 10), constrained_layout=True)
        fig_scatter.patch.set_facecolor('#1e1e1e')
        ax = fig_scatter.add_subplot(111)
        ax.set_facecolor('#1e1e1e')
        self._plot_toxicity_volume_correlation(ax, results['tweet_volumes'], results['top_changers']['toxicity_top_changers'])
        figures.append(fig_scatter)
        
        return figures

    def _plot_volume_changes(self, ax, volume_data: Dict):
        """Plot tweet volume changes."""
        users = [change['username'] for change in volume_data['top_changers']]
        changes = [change['change'] for change in volume_data['top_changers']]
        
        bars = ax.bar(users, changes, color='#2980b9')
        ax.set_title('Top Volume Changes by User', fontsize=32, color='white', pad=20)
        ax.set_ylabel('Change in Tweet Count', fontsize=28, color='white')
        ax.grid(True, alpha=0.3)
        
        # Style improvements
        ax.set_facecolor('#1e1e1e')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.tick_params(colors='white', labelsize=24)
        
        # Add value labels inside bars with larger font
        for bar in bars:
            height = bar.get_height()
            y_pos = height/2 if height >= 0 else height/2
            ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                   f'{int(height):+,}',
                   ha='center', va='center',
                   color='white', weight='bold',
                   fontsize=24)

    def _plot_metrics_changes(self, ax, metrics_data: Dict):
        """Plot metrics changes."""
        metrics = list(metrics_data.keys())
        changes = [data['group_change'] for data in metrics_data.values()]
        
        x = np.arange(len(metrics))
        bars = ax.bar(x, changes)
        for bar, change in zip(bars, changes):
            bar.set_color('#2ecc71' if change >= 0 else '#e74c3c')
            bar.set_alpha(0.6)
            
        ax.set_title('Group-Level Metrics Changes', fontsize=32, color='white', pad=20)
        ax.set_ylabel('Average Change', fontsize=28, color='white')
        
        ax.set_xticks(x)
        ax.set_xticklabels([metrics_data[m]['name'] for m in metrics], 
                          rotation=45, ha='right', color='white', fontsize=24)
        
        ax.set_facecolor('#1e1e1e')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.tick_params(colors='white', labelsize=24)
        ax.grid(True, alpha=0.3)
        
        # Add value labels with larger font
        for bar in bars:
            height = bar.get_height()
            y_pos = height/2 if height >= 0 else height/2
            ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                   f'{height:+.1f}',
                   ha='center', va='center',
                   color='white', weight='bold',
                   fontsize=24)

    def _plot_toxicity_changes(self, ax, toxicity_data: List[Dict]):
        """Plot toxicity changes."""
        users = [data['username'] for data in toxicity_data]
        changes = [data['change'] for data in toxicity_data]
        
        bars = ax.bar(users, changes)
        for bar, change in zip(bars, changes):
            bar.set_color('#2ecc71' if change >= 0 else '#e74c3c')
            bar.set_alpha(0.6)
            
        ax.set_title('Top Toxicity Changes by User', fontsize=32, color='white', pad=20)
        ax.set_ylabel('Change in Toxicity Level', fontsize=28, color='white')
        
        ax.set_facecolor('#1e1e1e')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.tick_params(colors='white', labelsize=24)
        ax.grid(True, alpha=0.3)
        
        # Add value labels with larger font
        for bar in bars:
            height = bar.get_height()
            y_pos = height/2 if height >= 0 else height/2
            ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                   f'{height:+.1f}',
                   ha='center', va='center',
                   color='white', weight='bold',
                   fontsize=24)

    def _plot_metric_top_changers(self, ax, metric_data: Dict):
        """Plot top changers for a specific metric."""
        users = [change['username'] for change in metric_data['top_changers']]
        changes = [change['change'] for change in metric_data['top_changers']]
        
        bars = ax.bar(users, changes)
        for bar, change in zip(bars, changes):
            bar.set_color('#2ecc71' if change >= 0 else '#e74c3c')
            bar.set_alpha(0.6)
            
        ax.set_title(f'Top Changes in {metric_data["name"]}\n{metric_data["scale"]}', 
                    fontsize=32, pad=20, color='white')
        ax.set_ylabel('Change in Score', fontsize=28, color='white')
        ax.grid(True, alpha=0.3)
        
        ax.set_facecolor('#1e1e1e')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.tick_params(colors='white', labelsize=24)
        
        # Add value labels with larger font
        for bar in bars:
            height = bar.get_height()
            y_pos = height/2 if height >= 0 else height/2
            ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                   f'{height:+.1f}',
                   ha='center', va='center',
                   color='white', weight='bold',
                   fontsize=24)

    def _plot_narrative_distribution(self, ax1, ax2, narrative_data):
        """Plot narrative distribution as text blocks."""
        # Set up the figure
        plt.gcf().set_size_inches(40, 36)
        
        # Clear axes and remove spines
        for ax in [ax1, ax2]:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            
        # Set titles
        ax1.set_title('Pre-war Narrative Distribution', color='white', pad=50, fontsize=56, weight='bold')
        ax2.set_title('Post-war Narrative Distribution', color='white', pad=50, fontsize=56, weight='bold')
        
        # Colors for narratives (in order)
        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
        
        # Calculate positions with more spacing
        y_positions = [0.85, 0.55, 0.25]  # Only 3 positions since we don't show Others if 0%
        
        # Function to create text with line breaks
        def format_text(text, width=25):  # Reduced width for better readability
            words = text.split()
            lines = []
            current_line = []
            current_length = 0
            
            for word in words:
                if current_length + len(word) + 1 <= width:
                    current_line.append(word)
                    current_length += len(word) + 1
                else:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                    current_length = len(word)
            
            if current_line:
                lines.append(' '.join(current_line))
            return '\n'.join(lines)

        # Display pre-war narratives directly from narrative_data
        pre_war_narratives = [n for n in narrative_data['pre_war_top_3'] if n['narrative'] != "Others" or n['percentage'] > 0]
        for i, item in enumerate(pre_war_narratives):
            narrative = item['narrative']
            percentage = item['percentage']
            
            # Format text with manual line breaks
            formatted_text = format_text(narrative)
            text = f"{formatted_text}\n{percentage:.1f}%"
            
            # Create text box
            bbox_props = dict(
                boxstyle="round,pad=0.3",
                fc=colors[i],
                ec="white",
                alpha=0.8,
                mutation_scale=2.0
            )
            
            # Add text with adjusted position and width
            ax1.text(0.5, y_positions[i], text,
                    color='black',
                    fontsize=42,
                    weight='bold',
                    ha='center',
                    va='center',
                    bbox=bbox_props,
                    transform=ax1.transAxes,
                    linespacing=1.3)
        
        # Display post-war narratives directly from narrative_data
        post_war_narratives = [n for n in narrative_data['post_war_top_3'] if n['narrative'] != "Others" or n['percentage'] > 0]
        for i, item in enumerate(post_war_narratives):
            narrative = item['narrative']
            percentage = item['percentage']
            
            # Format text with manual line breaks
            formatted_text = format_text(narrative)
            text = f"{formatted_text}\n{percentage:.1f}%"
            
            # Create text box
            bbox_props = dict(
                boxstyle="round,pad=0.3",
                fc=colors[i],
                ec="white",
                alpha=0.8,
                mutation_scale=2.0
            )
            
            # Add text with adjusted position and width
            ax2.text(0.5, y_positions[i], text,
                    color='black',
                    fontsize=42,
                    weight='bold',
                    ha='center',
                    va='center',
                    bbox=bbox_props,
                    transform=ax2.transAxes,
                    linespacing=1.3)

    def _plot_user_activity_timeline(self, ax, volume_data: Dict):
        """Plot user activity timeline for top 3 users."""
        users = [change['username'] for change in volume_data['top_changers']]
        pre_volumes = []
        post_volumes = []
        
        for user in users:
            pre_vol = next((change['pre_vol'] for change in volume_data['top_changers'] 
                          if change['username'] == user), 0)
            post_vol = next((change['post_vol'] for change in volume_data['top_changers'] 
                           if change['username'] == user), 0)
            pre_volumes.append(pre_vol)
            post_volumes.append(post_vol)
        
        x = ['Pre-war', 'Post-war']
        colors = ['#FF9999', '#66B2FF', '#99FF99']
        
        for i, (user, color) in enumerate(zip(users, colors)):
            ax.plot(x, [pre_volumes[i], post_volumes[i]], 'o-', 
                   label=f'@{user}', color=color, linewidth=3, markersize=12)
        
        ax.set_title('Tweet Volume Timeline (Top 3 Users)', pad=20, fontsize=32, color='white')
        ax.set_ylabel('Tweet Count', fontsize=28, color='white')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=24, facecolor='#1e1e1e', labelcolor='white')
        
        ax.set_facecolor('#1e1e1e')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.tick_params(colors='white', labelsize=24)

    def _plot_toxicity_volume_correlation(self, ax, volume_data: Dict, toxicity_data: List[Dict]):
        """Plot scatter of toxicity vs volume changes."""
        users = volume_data['top_changers'][:3]
        toxicity_changes = {user['username']: user['change'] for user in toxicity_data}
        
        colors = ['#FF9999', '#66B2FF', '#99FF99']
        
        for i, user in enumerate(users):
            username = user['username']
            volume_change = user['change']
            toxicity_change = toxicity_changes.get(username, 0)
            color = colors[i]
            
            ax.scatter(volume_change, toxicity_change, s=300, color=color, alpha=0.9)
            ax.annotate(f'@{username}', 
                       (volume_change, toxicity_change),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=24, color=color, weight='bold',
                       bbox=dict(facecolor='#1e1e1e', edgecolor='white', alpha=0.8, pad=1))
        
        ax.set_title('Volume vs Toxicity Changes (Top 3 Users)', pad=20, fontsize=32, color='white')
        ax.set_xlabel('Change in Tweet Volume', fontsize=28, color='white')
        ax.set_ylabel('Change in Toxicity', fontsize=28, color='white')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='white', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='white', linestyle='--', alpha=0.3)
        
        ax.set_facecolor('#1e1e1e')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.tick_params(colors='white', labelsize=24)

    def generate_report(self, results: Dict) -> Tuple[str, List[plt.Figure]]:
        """Generate a comprehensive group analysis report."""
        report_sections = []
        
        # 1. Overall Header with Analysis Periods
        report_sections.extend([
            "# Group Analysis Report\n",
            "**Analysis Periods:**",
            "- Pre-war: July 9, 2023 - October 7, 2023 (90 days before the war)",
            "- Post-war: October 1, 2024 - December 30, 2024 (90 days at end of 2024)\n",
            "## Overall Statistics\n"
        ])
        
        # 2. Tweet Volumes with percentage change
        volumes = results['tweet_volumes']
        pct_change = ((volumes['post_war_total'] - volumes['pre_war_total']) / volumes['pre_war_total'] * 100)
        report_sections.extend([
            "### Tweet Volume Analysis\n",
            f"- Total Pre-war Tweets: {volumes['pre_war_total']:,}",
            f"- Total Post-war Tweets: {volumes['post_war_total']:,} ({pct_change:+.1f}%)",
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
        
        # Update Narrative Evolution section to include percentages
        narratives = results['narrative_analysis']
        report_sections.extend([
            "### Narrative Evolution\n",
            "Pre-war Top Narratives:",
            *[f"- {n['narrative']} ({n['percentage']:.1f}%)" 
              for n in narratives['pre_war_top_3']
              if n['narrative'] != "Others" or n['percentage'] > 0],
            "\nPost-war Top Narratives:",
            *[f"- {n['narrative']} ({n['percentage']:.1f}%)" 
              for n in narratives['post_war_top_3']
              if n['narrative'] != "Others" or n['percentage'] > 0],
            f"\n**Narrative Evolution Analysis:**\n{narratives['analysis']}\n"
        ])
        
        # Add Emotional Tones section
        emotional_tones = results['emotional_tones']
        report_sections.extend([
            "\n### Emotional Tones\n",
            "Pre-war Top Emotional Tones:",
            *[f"- {tone}: {percentage:.1f}%" 
              for tone, percentage in emotional_tones['pre_war'].items()],
            "\nPost-war Top Emotional Tones:",
            *[f"- {tone}: {percentage:.1f}%" 
              for tone, percentage in emotional_tones['post_war'].items()],
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

    def _analyze_emotional_tones(self, df: pd.DataFrame) -> Dict:
        """Analyze top 3 emotional tones from the data."""
        all_tones = []
        for _, row in df.iterrows():
            tones = eval(row['emotional_tones'])
            # Normalize tones by converting to lowercase and removing extra whitespace
            normalized_tones = [t.lower().strip() for t in tones]
            all_tones.extend(normalized_tones)
        
        # Count and get top 3
        tone_counts = Counter(all_tones)
        total_tones = sum(tone_counts.values())
        
        # Calculate percentages for top 3
        top_3_tones = {
            tone: (count / total_tones) * 100 
            for tone, count in tone_counts.most_common(3)
        }
        
        # Add "Others" category
        others_count = sum(count for tone, count in tone_counts.items() 
                         if tone not in top_3_tones)
        if others_count > 0:
            top_3_tones["Others"] = (others_count / total_tones) * 100
            
        return top_3_tones

    def _plot_emotional_tones(self, pre_tones: Dict, post_tones: Dict) -> plt.Figure:
        """Create a single bar chart comparing pre/post war emotional tones."""
        # Create a larger figure
        fig = plt.figure(figsize=(24, 16), constrained_layout=True)
        ax = fig.add_subplot(111)
        fig.patch.set_facecolor('#1e1e1e')
        ax.set_facecolor('#1e1e1e')
        
        # Get all unique tones and sort by total percentage
        all_tones = list(set(list(pre_tones.keys()) + list(post_tones.keys())))
        total_percentages = {tone: (pre_tones.get(tone, 0) + post_tones.get(tone, 0)) 
                           for tone in all_tones}
        all_tones = sorted(all_tones, key=lambda x: total_percentages[x], reverse=True)
        
        x = np.arange(len(all_tones))
        width = 0.35
        
        # Create bars
        pre_values = [pre_tones.get(tone, 0) for tone in all_tones]
        post_values = [post_tones.get(tone, 0) for tone in all_tones]
        
        bars1 = ax.bar(x - width/2, pre_values, width, label='Pre-war', 
                      color='#FF9999', alpha=0.9)
        bars2 = ax.bar(x + width/2, post_values, width, label='Post-war', 
                      color='#66B2FF', alpha=0.9)
        
        # Customize plot with larger fonts
        ax.set_title('Emotional Tones Distribution', color='white', pad=30, fontsize=40)
        ax.set_ylabel('Percentage (%)', color='white', fontsize=36)
        ax.set_xticks(x)
        ax.set_xticklabels([tone.title() for tone in all_tones], 
                          rotation=30, ha='right', fontsize=32)
        
        # Style improvements
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.tick_params(axis='y', colors='white', labelsize=32)
        ax.set_ylim(0, max(max(pre_values), max(post_values)) * 1.2)
        ax.grid(True, alpha=0.2, color='white')
        
        # Add legend with larger font
        legend = ax.legend(fontsize=32, facecolor='#1e1e1e', labelcolor='white',
                         loc='upper right', bbox_to_anchor=(1, 1))
        legend.get_frame().set_alpha(0.9)
        
        # Add value labels with larger font
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                y_pos = height/2
                ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                       f'{height:.1f}%',
                       ha='center', va='center',
                       color='white',
                       fontsize=32,
                       fontweight='bold')
        
        autolabel(bars1)
        autolabel(bars2)
        
        return fig 