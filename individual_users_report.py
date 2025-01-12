import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown
import numpy as np
import json
import boto3

class UserAnalysisReport:
    def __init__(self):
        # Set seaborn style
        sns.set_theme()
        # Additional matplotlib customization
        plt.rcParams['figure.figsize'] = [12, 10]
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        
        # Initialize AWS Bedrock client
        self.llm_client = boto3.client(
            service_name='bedrock-runtime',
            region_name='us-east-1'
        )
        
    def generate_report(self, period_results, test_users):
        """Generate the analysis report."""
        report = []
        
        # Get the period dates from the data
        pre_war_data = period_results['pre_war'].iloc[0]
        post_war_data = period_results['post_war'].iloc[0]
        
        # Add title with actual dates
        report.append("# Pre/Post War Analysis - Key Findings\n")
        report.append(f"*Comparing Pre-war (2023-07-09 to 2023-10-07) to Post-war (2024-10-01 to 2024-12-30)*\n\n")
        
        # Ensure output directory exists
        os.makedirs('data/user_analysis', exist_ok=True)
        
        # Generate focused visualizations for all users first
        viz_paths_by_user = {}
        for username in test_users:
            viz_path = self._generate_key_visualizations(period_results, [username])
            viz_paths_by_user[username] = viz_path[0]  # Store path for each user
        
        # Now generate the analysis for each user
        self._add_focused_user_analysis(report, period_results, test_users, viz_paths_by_user)
        
        # Save and display report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join('data', 'user_analysis', f'analysis_report_{timestamp}.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        display(Markdown('\n'.join(report)))
        return report_path, list(viz_paths_by_user.values())

    def _normalize_score(self, score, metric):
        """Normalize a score to 0-100 scale based on the metric type."""
        # Define expected ranges for each metric
        ranges = {
            'judicial_security_ratio_score': (-100, 100),  # -100 (full security) to +100 (full judicial)
            'rights_security_balance_score': (-100, 100),  # -100 (full security) to +100 (full rights)
            'emergency_powers_position_score': (-100, 100),  # -100 (against) to +100 (for)
            'domestic_international_ratio_score': (-100, 100)  # -100 (international) to +100 (domestic)
        }
        
        if metric not in ranges:
            return score
        
        min_val, max_val = ranges[metric]
        normalized = ((score - min_val) / (max_val - min_val)) * 100
        return max(0, min(100, normalized))  # Ensure result is between 0 and 100

    def _generate_key_visualizations(self, period_results, test_users):
        """Generate focused visualizations highlighting key changes."""
        metrics = {
            'judicial_security_ratio_score': {
                'name': 'Judicial-Security Balance'
            },
            'rights_security_balance_score': {
                'name': 'Rights-Security Balance'
            },
            'emergency_powers_position_score': {
                'name': 'Emergency Powers Position'
            },
            'domestic_international_ratio_score': {
                'name': 'Domestic-International Focus'
            }
        }
        
        # Define consistent colors
        PRE_WAR_COLOR = '#1f77b4'  # blue
        POST_WAR_COLOR = '#ff7f0e'  # orange
        
        viz_paths = []
        
        for username in test_users:
            # Create a figure with 4 subplots
            fig, axes = plt.subplots(2, 2, figsize=(20, 16))
            fig.suptitle(f'Analysis Summary for {username}', fontsize=16, y=1.02)
            
            # Get user data
            pre_data = period_results['pre_war'][period_results['pre_war']['username'] == username].iloc[0]
            post_data = period_results['post_war'][period_results['post_war']['username'] == username].iloc[0]
            
            # 1. All Metrics Comparison - Line Chart (top left)
            ax = axes[0, 0]
            
            # Prepare data for all metrics
            metric_values = {
                'Pre-war': [float(pre_data[m]) for m in metrics.keys()],
                'Post-war': [float(post_data[m]) for m in metrics.keys()]
            }
            
            # Create clearer labels for metrics
            metric_labels = [info['name'] for info in metrics.values()]
            x = range(len(metrics))
            
            # Plot with larger markers and thicker lines
            ax.plot(x, metric_values['Pre-war'], 'o-', label='Pre-war', 
                   markersize=10, linewidth=2, color=PRE_WAR_COLOR)
            ax.plot(x, metric_values['Post-war'], 's-', label='Post-war', 
                   markersize=10, linewidth=2, color=POST_WAR_COLOR)
            
            # Customize chart
            ax.set_xticks(x)
            ax.set_xticklabels(metric_labels, rotation=45, ha='right')
            ax.legend(loc='upper right')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add value labels on the points
            for i, (pre_val, post_val) in enumerate(zip(metric_values['Pre-war'], metric_values['Post-war'])):
                ax.annotate(f'{pre_val:.1f}', (i, pre_val), textcoords="offset points", 
                          xytext=(0,10), ha='center', fontsize=10)
                ax.annotate(f'{post_val:.1f}', (i, post_val), textcoords="offset points", 
                          xytext=(0,-15), ha='center', fontsize=10)
                # Add change label in the middle
                change = post_val - pre_val
                if abs(change) >= 6.0:  # Only show significant changes
                    ax.annotate(f'{change:+.1f}', (i, (pre_val + post_val)/2), textcoords="offset points",
                              xytext=(20,0), ha='left', fontsize=10, fontweight='bold')
            
            ax.set_title('All Metrics Comparison')
            ax.set_ylabel('Score')
            ax.set_ylim(-5, 105)
            
            # 2. Significant Changes - Bar Chart (top right)
            ax = axes[0, 1]
            
            # Calculate changes and filter significant metrics
            significant_metrics = []
            for metric_key, metric_info in metrics.items():
                pre_val = float(pre_data[metric_key])
                post_val = float(post_data[metric_key])
                change = abs(post_val - pre_val)
                if change >= 6.0:  # Show metrics with changes >= 6.0 points
                    significant_metrics.append({
                        'key': metric_key,
                        'name': metric_info['name'],
                        'pre': pre_val,
                        'post': post_val,
                        'change': post_val - pre_val
                    })
            
            # Sort by absolute change magnitude
            significant_metrics.sort(key=lambda x: abs(x['change']), reverse=True)
            
            if significant_metrics:
                # Set up positions for grouped bars
                x = np.arange(len(significant_metrics))
                width = 0.35
                
                # Create bars
                bars1 = ax.bar(x - width/2, [m['pre'] for m in significant_metrics], width, 
                             label='Pre-war', color=PRE_WAR_COLOR)
                bars2 = ax.bar(x + width/2, [m['post'] for m in significant_metrics], width,
                             label='Post-war', color=POST_WAR_COLOR)
                
                # Customize chart
                ax.set_xticks(x)
                ax.set_xticklabels([m['name'] for m in significant_metrics], 
                                 rotation=45, ha='right')
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # Add value labels on bars
                for i, metrics in enumerate(significant_metrics):
                    ax.text(i - width/2, metrics['pre'], f"{metrics['pre']:.1f}", 
                           ha='center', va='bottom')
                    ax.text(i + width/2, metrics['post'], f"{metrics['post']:.1f}", 
                           ha='center', va='bottom')
                    # Add change label in the middle
                    ax.text(i, max(metrics['pre'], metrics['post']) + 5,
                           f"{metrics['change']:+.1f}", ha='center', va='bottom',
                           fontweight='bold', color='black')
                
                ax.set_title('Significant Changes (≥6.0 points)')
                ax.set_ylabel('Score')
                ax.set_ylim(-5, 105)
            else:
                ax.text(0.5, 0.5, 'No significant changes\n(≥6.0 points)',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Significant Changes')
            
            # 3. Toxicity Level Changes - Bar Chart (bottom left)
            ax = axes[1, 0]
            toxicity_data = {
                'Pre-war': float(pre_data['toxicity_level']),
                'Post-war': float(post_data['toxicity_level'])
            }
            bars = ax.bar(['Pre-war', 'Post-war'], toxicity_data.values())
            bars[0].set_color(PRE_WAR_COLOR)
            bars[1].set_color(POST_WAR_COLOR)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom')
            
            # Add change label
            change = toxicity_data['Post-war'] - toxicity_data['Pre-war']
            ax.text(0.5, max(toxicity_data.values()) + 5,
                   f"{change:+.1f}", ha='center', va='bottom',
                   fontweight='bold', color='black')
            
            ax.set_title('Toxicity Level Changes')
            ax.set_ylim(0, 100)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # 4. Tweet Volume Comparison - Bar Chart (bottom right)
            ax = axes[1, 1]
            volume_pre = len(eval(pre_data['toxic_examples'])) if isinstance(pre_data['toxic_examples'], str) else len(pre_data['toxic_examples'])
            volume_post = len(eval(post_data['toxic_examples'])) if isinstance(post_data['toxic_examples'], str) else len(post_data['toxic_examples'])
            
            bars = ax.bar(['Pre-war', 'Post-war'], [volume_pre, volume_post])
            bars[0].set_color(PRE_WAR_COLOR)
            bars[1].set_color(POST_WAR_COLOR)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom')
            
            # Add change label
            change = volume_post - volume_pre
            ax.text(0.5, max(volume_pre, volume_post) + 1,
                   f"{change:+d}", ha='center', va='bottom',
                   fontweight='bold', color='black')
            
            ax.set_title('Tweet Volume')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Adjust layout and save
            plt.tight_layout()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            viz_path = os.path.join('data', 'user_analysis', f'analysis_{username}_{timestamp}.png')
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            viz_paths.append(viz_path)
        
        return viz_paths

    def _get_toxic_tweets(self, data):
        """Extract the 2 most toxic tweets from the data."""
        try:
            if isinstance(data['toxic_examples'], str):
                # Handle potential truncation in the string representation
                toxic_tweets = eval(data['toxic_examples'].replace('...', ''))
            else:
                toxic_tweets = data['toxic_examples']
            
            # Ensure we have full tweets
            if toxic_tweets and isinstance(toxic_tweets, list):
                # Filter out any obviously truncated tweets (ending with ...)
                full_tweets = [tweet for tweet in toxic_tweets if not tweet.endswith('...')]
                return full_tweets[:2] if full_tweets else []
            return []
        except Exception as e:
            print(f"Error processing toxic tweets: {e}")
            return []

    def _add_focused_user_analysis(self, report, period_results, test_users, viz_paths_by_user):
        """Add focused analysis highlighting significant changes for each user."""
        for username in test_users:
            report.append(f"\n## {username} - Key Changes\n")
            
            pre_data = period_results['pre_war'][period_results['pre_war']['username'] == username].iloc[0]
            post_data = period_results['post_war'][period_results['post_war']['username'] == username].iloc[0]
            
            # Add toxic tweets section
            report.append("\n### Most Toxic Tweets\n")
            
            # Pre-war toxic tweets
            report.append("\n**Pre-war Period:**\n")
            pre_toxic = self._get_toxic_tweets(pre_data)
            if pre_toxic:
                for tweet in pre_toxic:
                    report.append(f"```\n{tweet}\n```\n")
            else:
                report.append("*No toxic tweets found*\n")
            
            # Post-war toxic tweets
            report.append("\n**Post-war Period:**\n")
            post_toxic = self._get_toxic_tweets(post_data)
            if post_toxic:
                for tweet in post_toxic:
                    report.append(f"```\n{tweet}\n```\n")
            else:
                report.append("*No toxic tweets found*\n")
            
            # Add metrics analysis
            self._add_metrics_analysis(report, pre_data, post_data)
            
            # Add narrative analysis
            self._add_narrative_analysis(report, pre_data, post_data)
            
            # Add entity changes
            self._add_entity_changes(report, pre_data, post_data)
            
            # Add visualizations at the end of each user's section
            report.append("\n### Visualizations\n")
            report.append(f"![Analysis Summary for {username}]({viz_paths_by_user[username]})\n\n")
            report.append("---\n")  # Add separator between users

    def _add_metrics_analysis(self, report, pre_data, post_data):
        """Add metrics analysis to the report."""
        metrics = {
            'judicial_security_ratio_score': {
                'name': 'Judicial-Security Balance',
                'scale': '(0 represents exclusive focus on security measures, while 100 represents exclusive focus on judicial reforms)',
                'explanation': {
                    'positive': 'More focus on judicial reform relative to security',
                    'negative': 'More focus on security relative to judicial reform',
                    'zero': 'Equal focus on judicial reform and security'
                }
            },
            'rights_security_balance_score': {
                'name': 'Rights-Security Balance',
                'scale': '(0 represents exclusive focus on security measures, while 100 represents exclusive focus on citizen rights)',
                'explanation': {
                    'positive': 'More emphasis on rights relative to security',
                    'negative': 'More emphasis on security relative to rights',
                    'zero': 'Equal emphasis on rights and security'
                }
            },
            'emergency_powers_position_score': {
                'name': 'Emergency Powers Position',
                'scale': '(0 represents complete opposition to emergency powers, while 100 represents full support)',
                'explanation': {
                    'positive': 'More supportive of emergency powers',
                    'negative': 'More critical of emergency powers',
                    'zero': 'Neutral stance on emergency powers'
                }
            },
            'domestic_international_ratio_score': {
                'name': 'Domestic-International Focus',
                'scale': '(0 represents exclusive focus on international issues, while 100 represents exclusive focus on domestic issues)',
                'explanation': {
                    'positive': 'More focus on domestic issues',
                    'negative': 'More focus on international issues',
                    'zero': 'Equal focus on domestic and international issues'
                }
            }
        }
        
        report.append("\n### Changes in Focus\n")
        
        # Calculate all changes
        all_changes = []
        for metric, info in metrics.items():
            pre_val = float(pre_data[metric])
            post_val = float(post_data[metric])
            change = post_val - pre_val
            all_changes.append((metric, info, pre_val, post_val, change))
        
        # Sort by absolute change magnitude and take top 2
        all_changes.sort(key=lambda x: abs(x[4]), reverse=True)
        top_changes = all_changes[:2]  # Only take top 2 changes
        
        # Display top 2 changes
        for metric, info, pre_val, post_val, change in top_changes:
            report.append(f"**{info['name']}** {info['scale']}:\n\n")
            report.append(f"Pre-war score: {pre_val:.1f}\n")
            report.append(f"Post-war score: {post_val:.1f}\n")
            report.append(f"Change: {change:+.1f} points\n\n")

    def _add_narrative_analysis(self, report, pre_data, post_data):
        """Add narrative analysis to the report."""
        # Analyze narrative changes
        pre_narratives = eval(pre_data['narratives']) if isinstance(pre_data['narratives'], str) else pre_data['narratives']
        post_narratives = eval(post_data['narratives']) if isinstance(post_data['narratives'], str) else post_data['narratives']
        
        if pre_narratives or post_narratives:
            report.append("\n### Narrative Evolution\n")
            
            # Show top 3 narratives from each period
            report.append("**Pre-war Top 3 Narratives:**\n")
            for narrative in pre_narratives[:3]:
                report.append(f"- {narrative}\n")
            
            report.append("\n**Post-war Top 3 Narratives:**\n")
            for narrative in post_narratives[:3]:
                report.append(f"- {narrative}\n")
            
            # Create analysis prompt for the LLM
            prompt = f"""Analyze these two sets of narratives and create a concise summary (max 40 words) of how the user's narrative focus evolved:

            Pre-war narratives:
            {pre_narratives[:3]}

            Post-war narratives:
            {post_narratives[:3]}

            Consider:
            1. What themes remained consistent?
            2. What new themes emerged?
            3. What themes were dropped?
            
            Format the response as a single, clear sentence."""
            
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
                report.append("\n**Analysis of Changes:**\n")
                report.append(f"*{analysis}*\n\n")
            except Exception as e:
                # Fallback to metric-based analysis if LLM call fails
                judicial_security_change = post_data['judicial_security_ratio_score'] - pre_data['judicial_security_ratio_score']
                domestic_international_change = post_data['domestic_international_ratio_score'] - pre_data['domestic_international_ratio_score']
                
                if abs(judicial_security_change) > 20 or abs(domestic_international_change) > 20:
                    explanation = []
                    if abs(judicial_security_change) > 20:
                        if judicial_security_change > 0:
                            explanation.append("shifted focus more towards judicial reform")
                        else:
                            explanation.append("increased emphasis on security concerns")
                    
                    if abs(domestic_international_change) > 20:
                        if domestic_international_change > 0:
                            explanation.append("focused more on domestic issues")
                        else:
                            explanation.append("increased attention to international matters")
                    
                    report.append("\n**Analysis of Changes:**\n")
                    report.append(f"*The narrative {' and '.join(explanation)}.*\n\n")

    def _add_entity_changes(self, report, pre_data, post_data):
        """Add entity changes to the report."""
        # Entity changes (only if significant)
        pre_attacked = set(eval(pre_data['attacked_entities']) if isinstance(pre_data['attacked_entities'], str) else pre_data['attacked_entities'])
        post_attacked = set(eval(post_data['attacked_entities']) if isinstance(post_data['attacked_entities'], str) else post_data['attacked_entities'])
        
        new_targets = post_attacked - pre_attacked
        dropped_targets = pre_attacked - post_attacked
        
        if new_targets or dropped_targets:
            report.append("\n### Changes in Focus\n")
            if new_targets:
                report.append("\n**New Entities Criticized:**\n")
                for entity in new_targets:
                    report.append(f"- {entity}\n")
            if dropped_targets:
                report.append("\n**No Longer Criticized:**\n")
                for entity in dropped_targets:
                    report.append(f"- {entity}\n") 