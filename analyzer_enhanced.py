from analyzer import TweetAnalyzer
import pandas as pd
import os
from datetime import datetime
from typing import List, Dict
import re
import json
import boto3

class EnhancedTweetAnalyzer(TweetAnalyzer):
    def __init__(self, batch_size=25, max_retries=3):
        super().__init__(batch_size, max_retries)
        self.llm_client = boto3.client('bedrock-runtime', region_name='us-east-1')
        self.model_name = "anthropic.claude-3-haiku-20240307-v1:0"

    def _create_enhanced_prompt(self, tweets_data: str, username: str) -> str:
        """Create prompt for the additional ratios analysis"""
        return f"""Analyze these tweets from @{username} focusing on specific ratios.
        
        TWEETS DATA:
        {tweets_data}
        
        Provide scores for these specific aspects in JSON format:
        {{
            "judicial_security_ratio": {{
                "score": number between -100 and +100,
                "explanation": "Brief explanation of the score (max 25 words)",
                "confidence": number between 0 and 100
            }},
            "rights_security_balance": {{
                "score": number between 0 and 100,
                "explanation": "Brief explanation of the score (max 25 words)",
                "confidence": number between 0 and 100
            }},
            "emergency_powers_position": {{
                "score": number between 1 and 10,
                "explanation": "Brief explanation of the score (max 25 words)",
                "confidence": number between 0 and 100
            }},
            "domestic_international_ratio": {{
                "score": number between -100 and +100,
                "explanation": "Brief explanation of the score (max 25 words)",
                "confidence": number between 0 and 100
            }}
        }}
        
        SCORING GUIDELINES:
        1. Judicial-Security Ratio:
           - -100: Purely judicial reform focused
           - +100: Purely security focused
           
        2. Rights-Security Balance:
           - 0: Pure individual rights focus
           - 100: Pure collective security focus
           
        3. Emergency Powers:
           - 1: Minimal executive powers advocated
           - 10: Maximum executive powers advocated
           
        4. Domestic-International:
           - -100%: Purely domestic focus
           - +100%: Purely international focus
        
        Base your analysis on actual content and keep explanations under 25 words."""

    def analyze_enhanced_metrics(self, tweets: List[Dict], username: str) -> List[Dict]:
        """Perform the enhanced analysis for additional ratios in batches"""
        all_metrics = []
        
        # Calculate number of batches
        num_tweets = len(tweets)
        num_batches = (num_tweets + self.batch_size - 1) // self.batch_size
        print(f"\nAnalyzing enhanced metrics for @{username}")
        
        # Process each batch
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, num_tweets)
            batch_tweets = tweets[start_idx:end_idx]
            
            # Format batch data
            formatted_data = ""
            for i, tweet in enumerate(batch_tweets, 1):
                formatted_data += f"[Tweet {i}]\n"
                tweet_text = tweet.get('text', tweet.get('tweet_text', ''))
                formatted_data += f"Text: {tweet_text}\n\n"
            
            prompt = self._create_enhanced_prompt(formatted_data, username)
            
            try:
                # Make API call using AWS Bedrock
                response = self.llm_client.invoke_model(
                    body=json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": 4096,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1
                    }),
                    modelId=self.model_name,
                    accept='application/json',
                    contentType='application/json'
                )
                
                print(f"✓ Batch {batch_idx + 1}/{num_batches}")
                
                # Parse response
                response_body = json.loads(response.get('body').read())
                response_content = response_body.get('content')[0].get('text', '')
                
                # Extract metrics from response
                try:
                    # Clean up the response content
                    clean_content = response_content
                    if '{' in clean_content:
                        clean_content = '{' + clean_content.split('{', 1)[1]
                    if '}' in clean_content:
                        clean_content = clean_content.rsplit('}', 1)[0] + '}'
                    
                    metrics = json.loads(clean_content)
                    
                    # Limit explanations to 25 words
                    for metric in metrics:
                        if 'explanation' in metrics[metric]:
                            words = metrics[metric]['explanation'].split()
                            metrics[metric]['explanation'] = ' '.join(words[:25])
                    
                    all_metrics.append(metrics)
                except json.JSONDecodeError:
                    metrics = {}
                    patterns = {
                        'judicial_security_ratio': r'"judicial_security_ratio"\s*:\s*{\s*"score"\s*:\s*(-?\d+\.?\d*)',
                        'rights_security_balance': r'"rights_security_balance"\s*:\s*{\s*"score"\s*:\s*(\d+\.?\d*)',
                        'emergency_powers_position': r'"emergency_powers_position"\s*:\s*{\s*"score"\s*:\s*(\d+\.?\d*)',
                        'domestic_international_ratio': r'"domestic_international_ratio"\s*:\s*{\s*"score"\s*:\s*(-?\d+\.?\d*)'
                    }
                    
                    for metric, pattern in patterns.items():
                        match = re.search(pattern, response_content, re.IGNORECASE | re.MULTILINE)
                        if match:
                            score = float(match.group(1))
                            
                            conf_pattern = rf'"{metric}"\s*:\s*{{\s*.*?"confidence"\s*:\s*(\d+\.?\d*)'
                            conf_match = re.search(conf_pattern, response_content, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                            confidence = float(conf_match.group(1)) if conf_match else 70
                            
                            exp_pattern = rf'"{metric}"\s*:\s*{{\s*.*?"explanation"\s*:\s*"([^"]*)"'
                            exp_match = re.search(exp_pattern, response_content, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                            explanation = exp_match.group(1) if exp_match else "No explanation provided"
                            
                            # Limit explanation to 25 words
                            words = explanation.split()
                            explanation = ' '.join(words[:25])
                            
                            metrics[metric] = {
                                'score': score,
                                'confidence': confidence,
                                'explanation': explanation
                            }
                    
                    if metrics:
                        all_metrics.append(metrics)
                    
            except Exception as e:
                print(f"Error in batch {batch_idx + 1}: {str(e)}")
        
        print(f"✓ Completed: {len(all_metrics)} batches")
        return all_metrics

    def _average_numerical_scores(self, analyses: List[Dict]) -> Dict:
        """Average numerical scores across batches with weighted confidence"""
        averaged_scores = {}
        for metric in ['judicial_security_ratio', 'rights_security_balance', 
                      'emergency_powers_position', 'domestic_international_ratio']:
            scores = []
            confidences = []
            explanations = []
            
            for analysis in analyses:
                if metric in analysis and 'score' in analysis[metric]:
                    scores.append(float(analysis[metric]['score']))
                    confidences.append(float(analysis[metric]['confidence']))
                    explanations.append(analysis[metric]['explanation'])
            
            if scores and confidences:
                # Weighted average based on confidence
                weighted_score = sum(s * c for s, c in zip(scores, confidences)) / sum(confidences)
                averaged_scores[metric] = {
                    'score': round(weighted_score, 2),
                    'confidence': round(sum(confidences) / len(confidences), 2),
                    'explanation': max(explanations, key=len)  # Use most detailed explanation
                }
        
        return averaged_scores

    def merge_user_analyses_enhanced(self, df: pd.DataFrame, tweets_df: pd.DataFrame = None, period_label: str = None) -> pd.DataFrame:
        """Enhanced version that adds ratio metrics to already analyzed data"""
        print(f"\nAdding enhanced metrics for {period_label} period...")
        
        # Perform enhanced analysis for each user
        enhanced_results = []
        for username in df['username'].unique():
            print(f"\nAnalyzing enhanced metrics for @{username}")
            
            # Get all tweets for this user from the tweets DataFrame
            if tweets_df is not None:
                user_tweets = tweets_df[tweets_df['author_username'] == username].to_dict('records')
            else:
                print(f"Warning: No tweets DataFrame provided for {username}")
                continue
            
            if not user_tweets:
                print(f"Warning: No tweets found for {username}")
                continue
                
            print(f"Found {len(user_tweets)} tweets for enhanced analysis")
            
            # Get batch analyses for enhanced metrics only
            batch_metrics = self.analyze_enhanced_metrics(user_tweets, username)
            print(f"Total enhanced batches analyzed: {len(batch_metrics)}")
            
            if batch_metrics:
                # Average the numerical scores across batches
                averaged_metrics = self._average_numerical_scores(batch_metrics)
                
                # Flatten the metrics structure for DataFrame compatibility
                flattened_metrics = {
                    'username': username
                }
                
                for metric in ['judicial_security_ratio', 'rights_security_balance', 
                             'emergency_powers_position', 'domestic_international_ratio']:
                    if metric in averaged_metrics:
                        metric_data = averaged_metrics[metric]
                        flattened_metrics.update({
                            f"{metric}_score": metric_data.get('score'),
                            f"{metric}_confidence": metric_data.get('confidence'),
                            f"{metric}_explanation": metric_data.get('explanation')
                        })
                    else:
                        print(f"Warning: {metric} not found for {username}")
                        flattened_metrics.update({
                            f"{metric}_score": None,
                            f"{metric}_confidence": None,
                            f"{metric}_explanation": "Metric not available"
                        })
                
                enhanced_results.append(flattened_metrics)
        
        if enhanced_results:
            # Create enhanced DataFrame with proper column names
            enhanced_df = pd.DataFrame(enhanced_results)
            
            # Merge with original analysis
            final_df = pd.merge(df, enhanced_df, on='username', how='left')
            
            # Reorder columns to group related metrics together
            metric_columns = []
            for metric in ['judicial_security_ratio', 'rights_security_balance', 
                         'emergency_powers_position', 'domestic_international_ratio']:
                metric_columns.extend([f"{metric}_score", f"{metric}_confidence", f"{metric}_explanation"])
            
            # Get all columns except metric columns
            other_columns = [col for col in final_df.columns if col not in metric_columns and col != 'username']
            
            # Reorder columns: username, other columns, metric columns
            final_df = final_df[['username'] + other_columns + metric_columns]
        else:
            print("Warning: No enhanced metrics were generated")
            final_df = df
        
        # Save enhanced version
        if not final_df.empty and period_label:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_dir = os.path.join('data', 'analysis', period_label)
            os.makedirs(save_dir, exist_ok=True)
            
            filename = f'merged_analysis_{period_label}_enhanced_{timestamp}.csv'
            filepath = os.path.join(save_dir, filename)
            final_df.to_csv(filepath, index=False)
            print(f"\nSaved enhanced analysis to: {filepath}")
        
        return final_df 