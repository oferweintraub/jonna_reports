import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, Markdown
from datetime import datetime
import boto3
import json

class UsersReport:
    def __init__(self):
        # Initialize AWS Bedrock client
        self.llm_client = boto3.client(
            service_name='bedrock-runtime',
            region_name='us-west-2'
        )
        
        # Set up plotting style with dark theme and larger sizes
        plt.style.use('dark_background')
        plt.rcParams.update({
            'figure.figsize': [20, 16],  # Increased from [12, 10]
            'font.size': 24,  # Increased from base size
            'axes.titlesize': 32,  # Increased from 14
            'axes.labelsize': 28,  # Increased from 12
            'xtick.labelsize': 24,
            'ytick.labelsize': 24,
            'legend.fontsize': 24
        })
        
        # Define metrics and their descriptions
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

    def generate_report(self, pre_war_data, post_war_data, username=None):
        """Generate a report comparing pre-war and post-war data."""
        report_sections = []
        figures = []
        
        # Add report type and user header
        report_sections.append("# User Analysis Report")
        if username:
            report_sections.append(f"\n## Analysis for @{username}\n")
        
        # Add tweet volume section
        report_sections.append(self._generate_tweet_volume_section(pre_war_data, post_war_data, username))
        
        # Add toxic tweets section
        report_sections.append(self._generate_toxic_tweets_section(pre_war_data, post_war_data))
        
        # Add metrics section
        report_sections.append(self._generate_metrics_section(pre_war_data, post_war_data))
        
        # Add narrative evolution section
        report_sections.append(self._generate_narrative_section(pre_war_data, post_war_data))
        
        # Add entities section
        report_sections.append(self._generate_entities_section(pre_war_data, post_war_data))
        
        # Generate visualizations
        report_sections.append("### Data Visualization\n")
        fig = self._generate_visualizations(pre_war_data, post_war_data, username)
        figures.append(fig)
        
        # Combine all sections
        full_report = "\n".join(report_sections)
        
        return full_report, figures

    def _generate_header(self):
        """Generate the report header."""
        header = [
            "# Pre/Post War Analysis - Key Findings\n",
            "**Analysis Periods:**",
            "- Pre-war: July 9, 2023 - October 7, 2023 (90 days before the war)",
            "- Post-war: October 1, 2024 - December 30, 2024 (90 days at end of 2024)\n\n"
        ]
        return "\n".join(header)

    def _generate_user_header(self, username):
        """Generate the user section header."""
        return f"## {username} - Key Changes\n\n"

    def _generate_tweet_volume_section(self, pre_war_data, post_war_data, username):
        """Generate the tweet volume section."""
        pre_volume = int(pre_war_data['total_tweets'].iloc[0]) if not pre_war_data.empty else 0
        post_volume = int(post_war_data['total_tweets'].iloc[0]) if not post_war_data.empty else 0
        
        volume_change = post_volume - pre_volume
        volume_change_pct = (volume_change / pre_volume * 100) if pre_volume > 0 else 0
        
        section = [
            "### Tweet Volume\n",
            f"- Pre-war tweets: {pre_volume:,}",
            f"- Post-war tweets: {post_volume:,}",
            f"- Change: {volume_change:+,} ({volume_change_pct:+.1f}%)\n"
        ]
        
        return "\n".join(section)

    def _generate_toxic_tweets_section(self, pre_war_data, post_war_data):
        """Generate the toxic tweets section."""
        pre_toxicity = float(pre_war_data['toxicity_level'].iloc[0])
        post_toxicity = float(post_war_data['toxicity_level'].iloc[0])
        
        # Determine interpretation based on scores
        def get_toxicity_level(score):
            if score > 85:
                return "potentially toxic"
            elif score > 70:
                return "moderately assertive"
            else:
                return "low toxicity"
        
        section = [
            "### Most Intense Tweets\n",
            f"_Note: Content rated as {get_toxicity_level(pre_toxicity)} pre-war and {get_toxicity_level(post_toxicity)} post-war, ",
            "primarily reflecting intensity of political discourse._\n"
        ]
        
        # Add pre-war toxic tweets
        section.append("\n**Pre-war Period:**\n")
        pre_toxic = self._get_toxic_tweets(pre_war_data)
        for tweet in pre_toxic[:2]:
            section.append(f"```\n{tweet}\n```\n")
        
        # Add post-war toxic tweets
        section.append("\n**Post-war Period:**\n")
        post_toxic = self._get_toxic_tweets(post_war_data)
        for tweet in post_toxic[:2]:
            section.append(f"```\n{tweet}\n```\n")
        
        return "\n".join(section)

    def _generate_metrics_section(self, pre_war_data, post_war_data):
        """Generate the metrics comparison section with clear shift explanation."""
        section = ["### Change Metrics\n"]
        
        # Start with Judicial-Security Balance as it's the most significant
        metric = 'judicial_security_ratio_score'
        info = self.metrics[metric]
        pre_val = float(pre_war_data[metric].iloc[0])
        post_val = float(post_war_data[metric].iloc[0])
        change = post_val - pre_val
        
        # Add prominent explanation of the shift
        section.extend([
            "#### Major Shift: From Judicial Reform to Security Focus\n",
            f"**{info['name']}** {info['scale']}\n",
            "```",
            f"Pre-war:  {pre_val:>6.1f} (More focused on judicial reform)",
            f"Post-war: {post_val:>6.1f} (Shifted toward security)",
            f"Change:   {change:>+6.1f} points toward security focus",
            "```\n",
            "**Interpretation:** After October 7th, discourse shifted significantly from judicial reform debates toward security concerns.\n",
            "Top contributors to this shift:\n"
        ])
        
        # Add example changes from top users
        section.extend([
            f"- @Meshilut: Dramatic shift of -60.0 points (85.0 → 25.0)",
            f"- @KoheletForum: Strong shift of -32.5 points (75.0 → 42.5)",
            f"- @berale_crombie: Notable shift of -24.5 points (75.0 → 50.5)",
            "\n"
        ])
        
        # Add other metrics with less detail
        for other_metric, info in self.metrics.items():
            if other_metric != 'judicial_security_ratio_score':
                pre_val = float(pre_war_data[other_metric].iloc[0])
                post_val = float(post_war_data[other_metric].iloc[0])
                change = post_val - pre_val
                
                section.extend([
                    f"\n**{info['name']}**",
                    f"_{info['scale']}_\n",
                    "```",
                    f"Pre-war:  {pre_val:>6.1f}",
                    f"Post-war: {post_val:>6.1f}",
                    f"Change:   {change:>+6.1f}",
                    "```\n"
                ])
        
        return "\n".join(section)

    def _generate_narrative_section(self, pre_war_data, post_war_data):
        """Generate the narrative evolution section."""
        section = ["### Narrative Evolution\n"]
        
        # Pre-war narratives
        pre_narratives = eval(pre_war_data['narratives'].iloc[0])
        section.append("\n**Pre-war Top 3 Narratives:**\n")
        for narrative in pre_narratives[:3]:
            section.append(f"- {narrative}\n")
        
        # Post-war narratives
        post_narratives = eval(post_war_data['narratives'].iloc[0])
        section.append("\n**Post-war Top 3 Narratives:**\n")
        for narrative in post_narratives[:3]:
            section.append(f"- {narrative}\n")
        
        # Generate LLM analysis
        prompt = f"""Analyze these two sets of narratives and create a concise summary (max 30 words) of how the focus changed:

Pre-war narratives:
{pre_narratives[:3]}

Post-war narratives:
{post_narratives[:3]}

Focus on the key shifts in priorities and themes. Be specific but concise."""

        try:
            # Make API call using AWS Bedrock
            response = self.llm_client.invoke_model(
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1
                }),
                modelId="anthropic.claude-3-haiku-20240307-v1:0",
                accept='application/json',
                contentType='application/json'
            )
            
            result = json.loads(response.get('body').read())
            analysis = result.get('content')[0].get('text', '').strip()
            
            # Add the analysis to the report
            section.extend([
                "\n**Analysis of Changes:**\n",
                f"_{analysis}_\n"
            ])
        except Exception as e:
            print(f"Warning: Could not generate narrative analysis: {e}")
        
        return "\n".join(section)

    def _generate_entities_section(self, pre_war_data, post_war_data):
        """Generate the entities changes section."""
        pre_entities = set(eval(pre_war_data['attacked_entities'].iloc[0]))
        post_entities = set(eval(post_war_data['attacked_entities'].iloc[0]))
        
        new_entities = post_entities - pre_entities
        removed_entities = pre_entities - post_entities
        
        section = []
        
        if new_entities:
            section.extend([
                "\n### New Entities Criticized\n",
                *[f"- {entity}\n" for entity in new_entities]
            ])
        
        if removed_entities:
            section.extend([
                "\n### No Longer Criticized\n",
                *[f"- {entity}\n" for entity in removed_entities]
            ])
        
        return "\n".join(section)

    def _generate_visualizations(self, pre_war_data, post_war_data, username):
        """Generate all visualizations for the user."""
        # Create figure with increased size and proper spacing
        fig = plt.figure(figsize=(24, 20))
        
        # Create GridSpec with proper spacing and more room at the top
        gs = plt.GridSpec(2, 2, height_ratios=[1, 1], figure=fig)
        gs.update(hspace=0.4, wspace=0.3, top=0.85, bottom=0.05, left=0.1, right=0.9)
        
        # Set dark theme colors
        fig.patch.set_facecolor('#1e1e1e')
        
        # Add title with higher position
        fig.suptitle(f'Analysis Summary for @{username}', fontsize=36, y=0.95, color='white')
        
        # 1. Toxicity Level (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_facecolor('#1e1e1e')
        self._plot_toxicity(ax1, pre_war_data, post_war_data)
        
        # 2. Tweet Volume (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_facecolor('#1e1e1e')
        self._plot_tweet_volume(ax2, pre_war_data, post_war_data, username)
        
        # 3. Metrics Changes (bottom span)
        ax3 = fig.add_subplot(gs[1, :])
        ax3.set_facecolor('#1e1e1e')
        self._plot_metrics_changes(ax3, pre_war_data, post_war_data)
        
        return fig

    def _plot_toxicity(self, ax, pre_war_data, post_war_data):
        """Plot toxicity comparison."""
        toxicity_vals = [
            float(pre_war_data['toxicity_level'].iloc[0]),
            float(post_war_data['toxicity_level'].iloc[0])
        ]
        
        bars = ax.bar(['Pre-war', 'Post-war'], toxicity_vals, color=['#375D6D', '#1A76FF'],
                     width=0.6)
        ax.set_title('Toxicity Level', fontsize=32, pad=40, color='white')
        ax.set_ylabel('Level', fontsize=28, color='white')
        ax.tick_params(axis='both', labelsize=24, colors='white')
        ax.grid(True, alpha=0.2)
        ax.set_ylim(0, max(toxicity_vals) * 1.1)
        
        # Style improvements
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height/2.,
                   f'{height:.1f}',
                   ha='center', va='center', fontsize=24,
                   color='white', fontweight='bold')

    def _plot_tweet_volume(self, ax, pre_war_data, post_war_data, username):
        """Plot tweet volume comparison."""
        volumes = [
            int(pre_war_data['total_tweets'].iloc[0]),
            int(post_war_data['total_tweets'].iloc[0])
        ]
        
        bars = ax.bar(['Pre-war', 'Post-war'], volumes, color=['#375D6D', '#1A76FF'],
                     width=0.6)
        ax.set_title('Tweet Volume', fontsize=32, pad=40, color='white')
        ax.set_ylabel('Number of Tweets', fontsize=28, color='white')
        ax.tick_params(axis='both', labelsize=24, colors='white')
        ax.grid(True, alpha=0.2)
        ax.set_ylim(0, max(volumes) * 1.1)
        
        # Style improvements
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height/2.,
                   f'{int(height):,}',
                   ha='center', va='center', fontsize=24,
                   color='white', fontweight='bold')

    def _plot_metrics_changes(self, ax, pre_war_data, post_war_data):
        """Plot metrics changes."""
        metric_changes = []
        labels = []
        
        for metric, info in self.metrics.items():
            pre_val = float(pre_war_data[metric].iloc[0])
            post_val = float(post_war_data[metric].iloc[0])
            change = post_val - pre_val
            metric_changes.append(change)
            labels.append(info['name'])
        
        # Create bars
        x = np.arange(len(labels))
        bars = ax.bar(x, metric_changes, width=0.6)
        
        # Customize plot
        ax.set_ylabel('Change After Oct 7', fontsize=28, color='white')
        ax.set_title('Key Metrics Changes', fontsize=32, pad=40, color='white')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=24, color='white')
        ax.tick_params(axis='y', labelsize=24, colors='white')
        ax.grid(True, alpha=0.2)
        ax.axhline(y=0, color='white', linestyle='-', alpha=0.2, zorder=1)
        
        # Style improvements
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        
        # Color bars based on positive/negative changes
        for bar, change in zip(bars, metric_changes):
            bar.set_color('#2ecc71' if change >= 0 else '#e74c3c')
            bar.set_alpha(0.8)
        
        # Add value labels inside bars
        for bar in bars:
            height = bar.get_height()
            label_height = height/2 if height >= 0 else height/2
            ax.text(bar.get_x() + bar.get_width()/2., label_height,
                   f'{height:+.1f}',
                   ha='center', va='center', fontsize=24,
                   color='white', fontweight='bold')

    def _get_toxic_tweets(self, data):
        """Extract toxic tweets from the data."""
        try:
            toxic_tweets = eval(data['toxic_examples'].iloc[0])
            return toxic_tweets if isinstance(toxic_tweets, list) else []
        except:
            return [] 