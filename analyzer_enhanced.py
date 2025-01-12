from analyzer import TweetAnalyzer
import pandas as pd
import os
from datetime import datetime
from typing import List, Dict

class EnhancedTweetAnalyzer(TweetAnalyzer):
    def __init__(self, batch_size=25, max_retries=3):
        super().__init__(batch_size, max_retries)
    
    def _create_enhanced_prompt(self, tweets_data: str, username: str) -> str:
        """Create prompt for the additional ratios analysis"""
        return f"""Analyze these tweets from @{username} focusing on specific ratios.
        
        TWEETS DATA:
        {tweets_data}
        
        Provide scores for these specific aspects in JSON format:
        {{
            "judicial_security_ratio": {{
                "score": number between -100 and +100,
                "explanation": "Brief explanation of the score",
                "confidence": number between 0 and 100
            }},
            "rights_security_balance": {{
                "score": number between 0 and 100,
                "explanation": "Brief explanation of the score",
                "confidence": number between 0 and 100
            }},
            "emergency_powers_position": {{
                "score": number between 1 and 10,
                "explanation": "Brief explanation of the score",
                "confidence": number between 0 and 100
            }},
            "domestic_international_ratio": {{
                "score": number between -100 and +100,
                "explanation": "Brief explanation of the score",
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
        
        Base your analysis on actual content, provide clear explanations, and indicate confidence levels."""

    def analyze_enhanced_metrics(self, tweets: List[Dict], username: str) -> Dict:
        """Perform the enhanced analysis for additional ratios"""
        formatted_data = self._format_tweets_batch(tweets)
        prompt = self._create_enhanced_prompt(formatted_data, username)
        
        try:
            response = self._analyze_batch([{"text": prompt, "created_at": ""}], 
                                         username, 0)
            return response
        except Exception as e:
            print(f"Error in enhanced analysis for {username}: {str(e)}")
            return None

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

    def merge_user_analyses_enhanced(self, df: pd.DataFrame, period_label: str = None) -> pd.DataFrame:
        """Enhanced version of merge_user_analyses that includes numerical ratios"""
        # First get the regular merged analysis
        merged_df = super().merge_user_analyses(df, period_label)
        
        # Perform enhanced analysis for each user
        enhanced_results = []
        for username in merged_df['username'].unique():
            user_tweets = df[df['username'] == username].to_dict('records')
            enhanced_metrics = self.analyze_enhanced_metrics(user_tweets, username)
            if enhanced_metrics:
                enhanced_results.append({
                    'username': username,
                    **enhanced_metrics
                })
        
        # Create enhanced DataFrame
        enhanced_df = pd.DataFrame(enhanced_results)
        
        # Merge with original analysis
        final_df = pd.merge(merged_df, enhanced_df, on='username', how='left')
        
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