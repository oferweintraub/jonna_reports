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
            region_name='us-west-2'
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
        # Filter users with at least 30 tweets in each period
        users_pre_30 = set(pre_df[pre_df['total_tweets'] >= 30]['username'])
        users_post_30 = set(post_df[post_df['total_tweets'] >= 30]['username'])
        users_with_30_tweets = users_pre_30.intersection(users_post_30)
        
        # Filter dataframes to only include users with sufficient tweets
        pre_df = pre_df[pre_df['username'].isin(users_with_30_tweets)]
        post_df = post_df[post_df['username'].isin(users_with_30_tweets)]
        
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
        
        # Get all unique usernames from both periods
        all_usernames = set(pre_df['username'].unique()) | set(post_df['username'].unique())
        
        for username in all_usernames:
            # Get pre-war volume (0 if user not present)
            pre_vol = pre_df[pre_df['username'] == username]['total_tweets'].iloc[0] if username in pre_df['username'].values else 0
            
            # Get post-war volume (0 if user not present)
            post_vol = post_df[post_df['username'] == username]['total_tweets'].iloc[0] if username in post_df['username'].values else 0
            
            change = post_vol - pre_vol
            # Calculate percentage change, handling division by zero
            if pre_vol > 0:
                pct_change = (change / pre_vol * 100)
            elif post_vol > 0:
                pct_change = float('inf')  # Infinite increase from 0
            else:
                pct_change = 0  # No change if both volumes are 0
                
            volume_changes.append({
                'username': username,
                'pre_vol': pre_vol,
                'post_vol': post_vol,
                'change': change,
                'pct_change': pct_change
            })
        
        # Sort and get top 5 changers by absolute change
        volumes['top_changers'] = sorted(
            volume_changes, 
            key=lambda x: abs(x['change']), 
            reverse=True
        )[:5]
        
        return volumes

    def _analyze_toxic_tweets(self, pre_df: pd.DataFrame, post_df: pd.DataFrame) -> Dict:
        """Analyze group-level toxicity and identify most toxic tweets."""
        def get_toxic_tweets(df: pd.DataFrame) -> List[Dict]:
            toxic_tweets = []
            for _, row in df.iterrows():
                try:
                    examples = eval(row['toxic_examples'])
                    toxicity = float(row['toxicity_level'])
                    for tweet in examples:
                        toxic_tweets.append({
                            'username': row['username'],
                            'tweet': tweet,
                            'toxicity': toxicity
                        })
                except (ValueError, SyntaxError) as e:
                    print(f"Warning: Could not process toxic tweets for user {row['username']}: {e}")
                    continue
            
            # Sort by toxicity and return top 3, or fewer if less than 3 available
            sorted_tweets = sorted(toxic_tweets, key=lambda x: x['toxicity'], reverse=True)
            return sorted_tweets[:3] if sorted_tweets else []
        
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
            
            # Get all unique usernames from both periods
            all_usernames = set(pre_df['username'].unique()) | set(post_df['username'].unique())
            
            for username in all_usernames:
                try:
                    # Get pre-war value (0 if user not present)
                    pre_val = pre_df[pre_df['username'] == username][metric_key].iloc[0] if username in pre_df['username'].values else 0
                    
                    # Get post-war value (0 if user not present)
                    post_val = post_df[post_df['username'] == username][metric_key].iloc[0] if username in post_df['username'].values else 0
                    
                    change = post_val - pre_val
                    user_changes.append({
                        'username': username,
                        'change': change,
                        'pre_val': pre_val,
                        'post_val': post_val
                    })
                except (IndexError, KeyError) as e:
                    print(f"Warning: Could not process metric {metric_key} for user {username}: {e}")
                    continue
            
            # Sort and get top 5 changers by absolute change
            top_changers = sorted(
                user_changes,
                key=lambda x: abs(x['change']),
                reverse=True
            )[:5] if user_changes else []
            
            metrics_analysis[metric_key] = {
                'name': metric_info['name'],
                'scale': metric_info['scale'],
                'description': metric_info['description'],
                'group_change': avg_change,
                'top_changers': top_changers
            }
        
        return metrics_analysis

    def _analyze_top_changers(self, pre_df: pd.DataFrame, post_df: pd.DataFrame) -> Dict:
        """Analyze top changers in toxicity levels."""
        toxicity_changes = []
        
        # Get all unique usernames from both periods
        all_usernames = set(pre_df['username'].unique()) | set(post_df['username'].unique())
        
        for username in all_usernames:
            try:
                # Get pre-war toxicity (0 if user not present)
                pre_tox = float(pre_df[pre_df['username'] == username]['toxicity_level'].iloc[0]) if username in pre_df['username'].values else 0.0
                
                # Get post-war toxicity (0 if user not present)
                post_tox = float(post_df[post_df['username'] == username]['toxicity_level'].iloc[0]) if username in post_df['username'].values else 0.0
                
                change = post_tox - pre_tox
                toxicity_changes.append({
                    'username': username,
                    'change': change
                })
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not process toxicity for user {username}: {e}")
                continue
        
        # Sort and get top 5 changers by absolute change
        top_changers = sorted(
                toxicity_changes,
                key=lambda x: abs(x['change']),
                reverse=True
        )[:5] if toxicity_changes else []
        
        return {
            'toxicity_top_changers': top_changers
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
            print("\nAttempting LLM narrative analysis...")
            
            prompt = f"""You are a data analyst specializing in narrative analysis. First, understand this important context:

Context about Kohelet Forum:
Kohelet Forum is an influential Israeli think tank established in 2012, self-defined as "non-partisan" but widely associated with Jewish nationalism and free-market principles. It gained significant attention for:
- Leading role in designing and promoting the 2023 judicial reform, which sparked massive protests
- Promoting free-market economics and limited government intervention
- Facing controversy over transparency and foreign funding
- Experiencing major changes in 2024 including loss of funding and staff reductions
- Being criticized for its stance on public housing, workers' rights, and minimum wage

Analyze these two sets of narratives to identify which remained consistent and which changed between periods, considering Kohelet Forum's role and evolution:

Pre-war narratives with percentages:
{[f"{k} ({v:.1f}%)" for k, v in pre_percentages.items()]}

Post-war narratives with percentages:
{[f"{k} ({v:.1f}%)" for k, v in post_percentages.items()]}

Important instructions:
1. First, merge very similar narratives together (e.g., combine narratives about the same topic with slightly different wording)
2. Sum the percentages of merged narratives
3. Identify:
   a) The 2 most common narratives that remained consistent between pre and post war (similar themes/focus)
   b) The 3 most significant narratives that changed (either disappeared, emerged, or significantly transformed)
4. Make sure narratives are clear and concise
5. Consider how narratives relate to Kohelet Forum's evolution and role
6. The percentages for each period MUST sum to exactly 100%

Return ONLY a valid JSON object with this EXACT format:
{{
    "consistent_narratives": [
        {{
            "narrative": "First consistent narrative across both periods",
            "pre_war_percentage": 33.3,
            "post_war_percentage": 30.0,
            "merged_from": ["original narrative 1", "original narrative 2"]
        }},
        {{
            "narrative": "Second consistent narrative across both periods",
            "pre_war_percentage": 22.2,
            "post_war_percentage": 25.0,
            "merged_from": ["original narrative 3", "original narrative 4"]
        }}
    ],
    "changed_narratives": [
        {{
            "narrative": "First changed/new/disappeared narrative",
            "pre_war_percentage": 25.5,
            "post_war_percentage": 0.0,
            "change_type": "disappeared",
            "merged_from": ["original narrative 5", "original narrative 6"]
        }},
        {{
            "narrative": "Second changed/new/disappeared narrative",
            "pre_war_percentage": 0.0,
            "post_war_percentage": 30.0,
            "change_type": "emerged",
            "merged_from": ["original narrative 7"]
        }},
        {{
            "narrative": "Third changed/new/disappeared narrative",
            "pre_war_percentage": 19.0,
            "post_war_percentage": 15.0,
            "change_type": "transformed",
            "merged_from": ["original narrative 8", "original narrative 9"]
        }}
    ],
    "analysis": "A 34-word analysis of how narratives evolved, focusing on the persistence of consistent themes and the nature of narrative changes in light of Kohelet Forum's role and evolution."
}}

IMPORTANT:
- You MUST identify exactly 2 consistent narratives and 3 changed narratives
- Percentages MUST sum to 100% for each period
- For consistent narratives, focus on those with similar percentages in both periods
- For changed narratives, clearly indicate if they disappeared, emerged, or transformed
- Consider how narratives reflect Kohelet Forum's evolving role and challenges
- Make narratives clear and concise"""

            response = self.llm_client.invoke_model(
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 4096,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1
                }),
                modelId="anthropic.claude-3-5-haiku-20241022-v1:0",
                accept='application/json',
                contentType='application/json'
            )
            
            result = json.loads(response.get('body').read())
            content = result.get('content')[0].get('text', '').strip()
            
            # Clean up the response
            content = content.replace('```json', '').replace('```', '').strip()
            start_idx = content.find('{')
            end_idx = content.rfind('}')
            
            if start_idx == -1 or end_idx == -1:
                raise ValueError("No valid JSON object found in response")
                
            content = content[start_idx:end_idx + 1]
            analysis = json.loads(content)
            
        except Exception as e:
            # Fallback to raw data analysis
            if not pre_percentages and not post_percentages:
                analysis = {
                    'consistent_narratives': [],
                    'changed_narratives': [],
                    'analysis': "No narratives found in the data",
                    'narrative_popularity': {
                        'pre_war': {},
                        'post_war': {}
                    }
                }
            else:
                # Use raw data without LLM grouping
                pre_top_3 = sorted(pre_percentages.items(), key=lambda x: x[1], reverse=True)
                post_top_3 = sorted(post_percentages.items(), key=lambda x: x[1], reverse=True)
                
                # Initialize with empty lists
                analysis = {
                    'consistent_narratives': [],
                    'changed_narratives': [],
                    'analysis': "Analysis using raw data due to LLM service unavailability",
                    'narrative_popularity': {
                        'pre_war': {},
                        'post_war': {}
                    }
                }
                
                # Find narratives that appear in both periods
                common_narratives = set(dict(pre_top_3).keys()) & set(dict(post_top_3).keys())
                
                # Sort common narratives by average percentage
                common_sorted = sorted(
                    [(n, pre_percentages[n], post_percentages[n]) for n in common_narratives],
                    key=lambda x: (x[1] + x[2])/2,
                    reverse=True
                )
                
                # Add top 2 consistent narratives
                for narrative, pre_pct, post_pct in common_sorted[:2]:
                    entry = {
                        "narrative": narrative,
                        "pre_war_percentage": pre_pct,
                        "post_war_percentage": post_pct,
                        "merged_from": [narrative]
                    }
                    analysis['consistent_narratives'].append(entry)
                    analysis['narrative_popularity']['pre_war'][narrative] = pre_pct
                    analysis['narrative_popularity']['post_war'][narrative] = post_pct
                
                # Find narratives that disappeared (in pre-war but not post-war)
                disappeared = [(n, p, 0) for n, p in pre_top_3 if n not in dict(post_top_3)]
                
                # Find narratives that emerged (in post-war but not pre-war)
                emerged = [(n, 0, p) for n, p in post_top_3 if n not in dict(pre_top_3)]
                
                # Find narratives that transformed (significant change in percentage)
                transformed = [(n, pre_percentages[n], post_percentages[n]) 
                             for n in common_narratives 
                             if abs(pre_percentages[n] - post_percentages[n]) > 5
                             and n not in [x['narrative'] for x in analysis['consistent_narratives']]]
                
                # Combine all changed narratives and sort by the magnitude of change
                all_changes = (
                    [(n, pre_p, post_p, "disappeared") for n, pre_p, post_p in disappeared] +
                    [(n, pre_p, post_p, "emerged") for n, pre_p, post_p in emerged] +
                    [(n, pre_p, post_p, "transformed") for n, pre_p, post_p in transformed]
                )
                
                all_changes.sort(key=lambda x: abs(x[1] - x[2]), reverse=True)
                
                # Add top 3 changed narratives
                for narrative, pre_pct, post_pct, change_type in all_changes[:3]:
                    entry = {
                        "narrative": narrative,
                        "pre_war_percentage": pre_pct,
                        "post_war_percentage": post_pct,
                        "change_type": change_type,
                        "merged_from": [narrative]
                    }
                    analysis['changed_narratives'].append(entry)
                    if pre_pct > 0:
                        analysis['narrative_popularity']['pre_war'][narrative] = pre_pct
                    if post_pct > 0:
                        analysis['narrative_popularity']['post_war'][narrative] = post_pct
                
                # Calculate remaining percentages
                pre_accounted = sum(n['pre_war_percentage'] for n in analysis['consistent_narratives']) + \
                              sum(n['pre_war_percentage'] for n in analysis['changed_narratives'])
                post_accounted = sum(n['post_war_percentage'] for n in analysis['consistent_narratives']) + \
                               sum(n['post_war_percentage'] for n in analysis['changed_narratives'])
                
                # Adjust percentages to sum to 100%
                if pre_accounted < 100:
                    scale_factor = 100 / pre_accounted
                    for n in analysis['consistent_narratives'] + analysis['changed_narratives']:
                        n['pre_war_percentage'] *= scale_factor
                        if n['narrative'] in analysis['narrative_popularity']['pre_war']:
                            analysis['narrative_popularity']['pre_war'][n['narrative']] *= scale_factor
                
                if post_accounted < 100:
                    scale_factor = 100 / post_accounted
                    for n in analysis['consistent_narratives'] + analysis['changed_narratives']:
                        n['post_war_percentage'] *= scale_factor
                        if n['narrative'] in analysis['narrative_popularity']['post_war']:
                            analysis['narrative_popularity']['post_war'][n['narrative']] *= scale_factor
        
        # Add narrative popularity data structure
            analysis['narrative_popularity'] = {
            'pre_war': {},
            'post_war': {}
        }
        
        # Add consistent narratives to popularity
        for item in analysis.get('consistent_narratives', []):
            analysis['narrative_popularity']['pre_war'][item['narrative']] = item['pre_war_percentage']
            analysis['narrative_popularity']['post_war'][item['narrative']] = item['post_war_percentage']
        
        # Add changed narratives to popularity
        for item in analysis.get('changed_narratives', []):
            if item['pre_war_percentage'] > 0:
                analysis['narrative_popularity']['pre_war'][item['narrative']] = item['pre_war_percentage']
            if item['post_war_percentage'] > 0:
                analysis['narrative_popularity']['post_war'][item['narrative']] = item['post_war_percentage']
        
        # Add a separate LLM call just for the evolution analysis
        try:
            print("\nAttempting evolution analysis with LLM...")
            evolution_prompt = f"""You are a political discourse analyst. Analyze how these narratives changed from pre-war to post-war:

Pre-war Top Narratives:
{[f"- {n['narrative']} ({n['pre_war_percentage']:.1f}%)" for n in analysis['consistent_narratives'] + analysis['changed_narratives'] if n['pre_war_percentage'] > 0]}

Post-war Top Narratives:
{[f"- {n['narrative']} ({n['post_war_percentage']:.1f}%)" for n in analysis['consistent_narratives'] + analysis['changed_narratives'] if n['post_war_percentage'] > 0]}

Return ONLY a concise 40-word analysis focusing on the key shifts in narrative focus, new themes that emerged, and themes that disappeared. Focus on the most significant changes.
NO other text, NO explanations, EXACTLY 40 words."""

            print("\nEvolution prompt:", evolution_prompt)
            
            evolution_response = self.llm_client.invoke_model(
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 200,
                    "messages": [{"role": "user", "content": evolution_prompt}],
                    "temperature": 0.1
                }),
                modelId="anthropic.claude-3-5-haiku-20241022-v1:0",
                accept='application/json',
                contentType='application/json'
            )
            
            evolution_result = json.loads(evolution_response.get('body').read())
            evolution_content = evolution_result.get('content')[0].get('text', '').strip()
            
            print("\nEvolution response:", evolution_content)
            
            # Clean up the evolution content
            evolution_content = evolution_content.strip('"').strip()
            if evolution_content.lower().startswith(('here', 'analysis:', 'response:')):
                evolution_content = ' '.join(evolution_content.split()[1:])
            
            analysis['analysis'] = evolution_content
            
        except Exception as e:
            print(f"\nError in evolution analysis: {str(e)}")
            analysis['analysis'] = "Analysis using raw data due to LLM service unavailability"
        
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
        
        # 5. Narrative Distribution
        if 'narrative_analysis' in results:
            fig_narratives = plt.figure(figsize=(40, 48))
            fig_narratives.patch.set_facecolor('#1e1e1e')
            
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
            
            try:
                self._plot_narrative_distribution(ax1, ax2, results['narrative_analysis'])
                figures.append(fig_narratives)
            except Exception as e:
                plt.close(fig_narratives)
        
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

    def _plot_metrics_changes(self, ax, metrics_data):
        """Plot metrics changes."""
        metrics = []
        changes = []
        
        for metric_key, metric_info in metrics_data.items():
            metrics.append(metric_info['name'])
            changes.append(float(metric_info.get('group_change', 0)))
        
        # Create bars
        x = np.arange(len(metrics))
        bars = ax.bar(x, changes, width=0.6)
        
        # Customize plot
        ax.set_ylabel('Change After Oct 7', fontsize=28, color='white')
        ax.set_title('Key Metrics Changes', fontsize=32, pad=40, color='white')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=30, ha='right', fontsize=24, color='white')
        ax.tick_params(axis='y', labelsize=24, colors='white')
        ax.grid(True, alpha=0.2)
        ax.axhline(y=0, color='white', linestyle='-', alpha=0.2, zorder=1)
        
        # Style improvements
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        
        # Color bars based on positive/negative changes
        for bar, change in zip(bars, changes):
            bar.set_color('#e74c3c' if change < 0 else '#2ecc71')
            bar.set_alpha(0.8)
            
            # Add value labels inside bars
            height = bar.get_height()
            label_height = height/2 if height >= 0 else height/2
            ax.text(bar.get_x() + bar.get_width()/2., label_height,
                   f'{change:+.1f}',
                   ha='center', va='center', fontsize=24,
                   color='white', fontweight='bold')

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
        """Plot the narrative distribution for pre-war and post-war periods."""
        # Clear any existing plots
        ax1.clear()
        ax2.clear()
        
        # Set background colors
        ax1.set_facecolor('#1e1e1e')
        ax2.set_facecolor('#1e1e1e')
        
        # Define y-positions for narratives
        y_positions = [0.8, 0.6, 0.4, 0.2]  # Positions for up to 4 narratives
        
        # Function to wrap text
        def wrap_text(text, width=25):
            return textwrap.fill(text, width=width)
        
        # Plot consistent narratives (blue)
        consistent_color = '#3498db'  # Blue
        for i, narrative in enumerate(narrative_data.get('consistent_narratives', [])):
            # Pre-war
            ax1.text(0.02, y_positions[i], f"{wrap_text(narrative['narrative'])} ({narrative['pre_war_percentage']:.1f}%)",
                    color='white', fontsize=42, bbox=dict(facecolor=consistent_color, alpha=0.7, edgecolor='none', pad=10))
            
            # Post-war
            ax2.text(0.02, y_positions[i], f"{wrap_text(narrative['narrative'])} ({narrative['post_war_percentage']:.1f}%)",
                    color='white', fontsize=42, bbox=dict(facecolor=consistent_color, alpha=0.7, edgecolor='none', pad=10))
        
        # Plot changed narratives with different colors based on change type
        colors = {
            'disappeared': '#e74c3c',  # Red
            'emerged': '#2ecc71',      # Green
            'transformed': '#f1c40f'   # Yellow
        }
        
        for i, narrative in enumerate(narrative_data.get('changed_narratives', []), start=2):
            change_type = narrative.get('change_type', 'transformed')
            color = colors.get(change_type, '#95a5a6')  # Default gray if type not found
            
            # Pre-war (if percentage > 0)
            if narrative['pre_war_percentage'] > 0:
                ax1.text(0.02, y_positions[i], f"{wrap_text(narrative['narrative'])} ({narrative['pre_war_percentage']:.1f}%)",
                        color='white', fontsize=42, bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', pad=10))
            
            # Post-war (if percentage > 0)
            if narrative['post_war_percentage'] > 0:
                ax2.text(0.02, y_positions[i], f"{wrap_text(narrative['narrative'])} ({narrative['post_war_percentage']:.1f}%)",
                        color='white', fontsize=42, bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', pad=10))
        
        # Set titles
        ax1.set_title("Pre-war Narrative Distribution", color='white', fontsize=44, pad=20, weight='bold')
        ax2.set_title("Post-war Narrative Distribution", color='white', fontsize=44, pad=20, weight='bold')
        
        # Remove axes
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        
        # Add legend for narrative types
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor=consistent_color, alpha=0.7, label='Consistent'),
            plt.Rectangle((0, 0), 1, 1, facecolor=colors['disappeared'], alpha=0.7, label='Disappeared'),
            plt.Rectangle((0, 0), 1, 1, facecolor=colors['emerged'], alpha=0.7, label='Emerged'),
            plt.Rectangle((0, 0), 1, 1, facecolor=colors['transformed'], alpha=0.7, label='Transformed')
        ]
        
        # Add legend to both plots
        for ax in [ax1, ax2]:
            ax.legend(handles=legend_elements, loc='upper right', 
                     bbox_to_anchor=(0.98, 0.98), fontsize=36, 
                     facecolor='#1e1e1e', edgecolor='none',
                     labelcolor='white')
            
            # If no narratives, display a message
        if not narrative_data.get('consistent_narratives', []) and not narrative_data.get('changed_narratives', []):
            for ax in [ax1, ax2]:
                ax.text(0.5, 0.5, "No narratives available",
                       color='white',
                       fontsize=42,
                       ha='center',
                       va='center',
                       transform=ax.transAxes)

    def _plot_user_activity_timeline(self, ax, volume_data: Dict):
        """Plot user activity timeline for top 5 users."""
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
        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC']  # Added two more colors
        
        for i, (user, color) in enumerate(zip(users, colors)):
            ax.plot(x, [pre_volumes[i], post_volumes[i]], 'o-', 
                   label=f'@{user}', color=color, linewidth=3, markersize=12)
        
        ax.set_title('Tweet Volume Timeline (Top 5 Users)', pad=20, fontsize=32, color='white')
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
        """Plot correlation between toxicity and volume changes."""
        # Get top 5 users by volume change
        top_users = [change['username'] for change in volume_data['top_changers']]
        
        # Get volume changes for these users
        volume_changes = [change['change'] for change in volume_data['top_changers']]
        
        # Get toxicity changes for these users
        toxicity_changes = []
        for user in top_users:
            tox_change = next((change['change'] for change in toxicity_data if change['username'] == user), 0)
            toxicity_changes.append(tox_change)
        
        # Create scatter plot with larger points
        for user, vol, tox in zip(top_users, volume_changes, toxicity_changes):
            ax.scatter(vol, tox, s=400, alpha=0.7, label=f'@{user}')  # Increased point size from 200 to 400
            ax.annotate(f'@{user}', (vol, tox), 
                       xytext=(5, 5), textcoords='offset points',
                       color='white', fontsize=24)
        
        ax.set_title('Volume vs Toxicity Changes (Top 5 Users)', pad=20, fontsize=32, color='white')
        ax.set_xlabel('Change in Tweet Volume', fontsize=28, color='white')
        ax.set_ylabel('Change in Toxicity', fontsize=28, color='white')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=24, facecolor='#1e1e1e', labelcolor='white')
        
        # Add zero lines
        ax.axhline(y=0, color='white', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='white', linestyle='--', alpha=0.3)
        
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
                f"**Group Average Change: {metric_data['group_change']:.1f}**\n",
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
        narrative_data = results['narrative_analysis']
        if 'consistent_narratives' in narrative_data:
        report_sections.extend([
                "### Narrative Evolution Analysis\n",
                "#### Consistent Narratives",
                *[f"- **{n['narrative']}**\n  - Pre-war: {n['pre_war_percentage']:.1f}%\n  - Post-war: {n['post_war_percentage']:.1f}%"
                  for n in narrative_data['consistent_narratives']],
                "\n#### Changed Narratives",
                *[f"- **{n['narrative']}** ({n['change_type'].title()})\n  - Pre-war: {n['pre_war_percentage']:.1f}%\n  - Post-war: {n['post_war_percentage']:.1f}%"
                  for n in narrative_data['changed_narratives']],
                f"\n**Evolution Analysis:** {narrative_data.get('analysis', '')}\n"
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
        for metric, data in results['metrics_changes'].items():
            # Add metric header and description
            section.extend([
                f"### {data['name']}\n",
                f"{data['scale']}\n",
                f"Group Average Change: {data['group_change']:.1f}\n",
            ])
            
            # Add top changes with improved formatting
            if data['top_changers']:
                section.append("Top Changes:\n")
                for user in data['top_changers']:
                    section.append(f"@{user['username']} {user['change']:+.1f} points │ Pre: {user['pre_val']:.1f} → Post: {user['post_val']:.1f}")
                section.append("\n")
        
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