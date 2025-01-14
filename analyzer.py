import json
import re
import time
from datetime import datetime
import pandas as pd
from typing import List, Dict
import boto3
from botocore.config import Config
import os
from dotenv import load_dotenv

class TweetAnalyzer:
    def __init__(self, batch_size=25, max_retries=3):
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.llm_client = self._init_llm_client()
        # Use the specified model consistently
        self.model_name = "anthropic.claude-3-5-haiku-20241022-v1:0"
        
    def _init_llm_client(self):
        """Initialize AWS Bedrock client with timeouts and retries"""
        load_dotenv(override=True)
        config = Config(
            connect_timeout=10,
            read_timeout=30,
            retries={'max_attempts': 2}
        )
        return boto3.client(
            service_name='bedrock-runtime',
            region_name='us-west-2',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            aws_session_token=os.getenv('AWS_SESSION_TOKEN'),
            config=config
        )
    
    def _format_tweets_batch(self, tweets: List[Dict]) -> str:
        """Format tweets for LLM analysis"""
        formatted = "TWEET BATCH FOR ANALYSIS:\n\n"
        for i, tweet in enumerate(tweets, 1):
            formatted += f"[Tweet {i}]\n"
            formatted += f"Date: {tweet['created_at']}\n"
            formatted += f"Text: {tweet['text']}\n\n"
        return formatted
    
    def _create_prompt(self, formatted_tweets: str, username: str) -> str:
        """Create analysis prompt with emphasis on concise responses"""
        return f"""Context about Kohelet Forum:
        Kohelet Forum is an influential Israeli think tank established in 2012, self-defined as "non-partisan" but widely associated with Jewish nationalism and free-market principles. It gained significant attention for:
        - Leading role in designing and promoting the 2023 judicial reform, which sparked massive protests
        - Promoting free-market economics and limited government intervention
        - Facing controversy over transparency and foreign funding
        - Experiencing major changes in 2024 including loss of funding and staff reductions
        - Being criticized for its stance on public housing, workers' rights, and minimum wage
        
        You are a political expert in Israel politics analyzing tweets by user @{username}. 
        Note that most tweets are in Hebrew - analyze them in their original context.
        
        {formatted_tweets}
        
        Provide a concise analysis in the exact JSON format below. Keep all text fields brief and focused:
        {{
            "narratives": [3 brief phrases, max 10 words each],
            "attacked_entities": [
                List up to 5 SPECIFIC names of politicians or organizations being criticized.
                Use actual names, NOT generic terms like "radical groups" or "activists".
                If fewer than 5 specific entities are mentioned, include only those that are clearly identified.
            ],
            "protected_entities": [
                List up to 5 SPECIFIC names of politicians or organizations being defended.
                Use actual names, NOT generic terms like "supporters" or "the right wing".
                Do not include the trivial "Kohelet Forum" or the username who posted the tweets.
                If fewer than 5 specific entities are mentioned, include only those that are clearly identified.
            ],
            "toxicity_level": [0-100],
            "toxic_examples": [2 most toxic tweets from the batch, keep original Hebrew text],
            "emotional_tones": [2 lowercase words from: neutral, angry, cynical, fearful, frustrated, optimistic],
            "stated_goals": [2-3 goals, max 10 words each],
            "psychological_profile": [max 30 words]
        }}
        
        Keep all responses extremely concise to ensure complete JSON structure.
        For toxic_examples, preserve the original Hebrew text exactly as it appears in the tweets."""

    def _extract_json_with_regex(self, text: str) -> dict:
        """Extract JSON from text with improved error handling and robust regex fallback"""
        try:
            # Initialize extracted dictionary
            extracted = {}
            
            # First try to extract a complete JSON object
            json_pattern = r'\{[^{]*"narratives".*"confidence_explanation":[^}]*\}'
            match = re.search(json_pattern, text, re.DOTALL)
            
            if match:
                json_str = match.group(0)
                # Clean up common JSON issues
                json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                json_str = re.sub(r'\s+', ' ', json_str)
                json_str = re.sub(r'\[([^"\]\[]+)\]', 
                                lambda m: '[' + ','.join(f'"{x.strip()}"' for x in m.group(1).split(',')) + ']', 
                                json_str)
                
                try:
                    data = json.loads(json_str)
                    if self._validate_json_structure(data):
                        return data
                except json.JSONDecodeError:
                    pass

            # Always try regex fallback for individual fields
            patterns = {
                'narratives': r'"narratives"\s*:\s*\[(.*?)\](?=\s*,\s*")',
                'attacked_entities': r'"attacked_entities"\s*:\s*\[(.*?)\](?=\s*,\s*")',
                'protected_entities': r'"protected_entities"\s*:\s*\[(.*?)\](?=\s*,\s*")',
                'toxicity_level': r'"toxicity_level"\s*:\s*(\d+)',
                'toxic_examples': r'"toxic_examples"\s*:\s*\[(.*?)\](?=\s*,\s*")',
                'emotional_tones': r'"emotional_tones"\s*:\s*\[(.*?)\](?=\s*,\s*")',
                'stated_goals': r'"stated_goals"\s*:\s*\[(.*?)\](?=\s*,\s*")',
                'psychological_profile': r'"psychological_profile"\s*:\s*"([^"]*)"',
                'confidence_ratings': r'"confidence_ratings"\s*:\s*({[^}]+})',
                'confidence_explanation': r'"confidence_explanation"\s*:\s*"([^"]*)"'
            }
            
            for field, pattern in patterns.items():
                matches = re.findall(pattern, text, re.DOTALL)
                if matches:
                    value = matches[0]
                    if field == 'toxicity_level':
                        extracted[field] = int(value)
                    elif field == 'psychological_profile':
                        extracted[field] = value.strip()
                    elif field == 'confidence_ratings':
                        try:
                            # Clean and parse the confidence ratings JSON
                            ratings_str = value.replace("'", '"')  # Replace single quotes with double quotes
                            ratings = json.loads(ratings_str)
                            extracted[field] = ratings
                        except:
                            # Default confidence ratings if parsing fails
                            extracted[field] = {
                                'narratives_confidence': 70,
                                'attacked_entities_confidence': 70,
                                'protected_entities_confidence': 70,
                                'emotional_tones_confidence': 70,
                                'stated_goals_confidence': 70,
                                'overall_confidence': 70
                            }
                    else:
                        # Clean and parse array content
                        items = re.findall(r'"([^"]+)"', value)
                        if not items:
                            # Try without quotes
                            items = [x.strip() for x in value.split(',') if x.strip()]
                        
                        if field == 'toxic_examples':
                            # For toxic examples, preserve original text including quotes
                            items = re.findall(r'"([^"]+)"', value) or [x.strip() for x in value.split(',') if x.strip()]
                            if len(items) < 2:  # Ensure we always have 2 examples
                                items.extend(['N/A'] * (2 - len(items)))
                        
                        extracted[field] = items[:5] if field in ['narratives', 'attacked_entities', 'protected_entities'] else items
            
            # Ensure all required fields are present with defaults if needed
            defaults = {
                'narratives': ['No clear narrative identified'],
                'attacked_entities': ['No specific entities identified'],
                'protected_entities': ['No specific entities identified'],
                'toxicity_level': 0,
                'toxic_examples': ['No toxic content identified', 'No toxic content identified'],
                'emotional_tones': ['neutral', 'neutral'],
                'stated_goals': ['No clear goals identified'],
                'psychological_profile': 'Insufficient data for profile',
                'confidence_ratings': {
                    'narratives_confidence': 70,
                    'attacked_entities_confidence': 70,
                    'protected_entities_confidence': 70,
                    'emotional_tones_confidence': 70,
                    'stated_goals_confidence': 70,
                    'overall_confidence': 70
                },
                'confidence_explanation': 'Analysis completed with moderate confidence due to available data'
            }
            
            for field, default in defaults.items():
                if field not in extracted or not extracted[field]:
                    extracted[field] = default
            
            return extracted
            
        except Exception as e:
            print(f"\nJSON extraction error: {str(e)}")
            # Return a dictionary with default values instead of None
            return {
                'narratives': ['Error in analysis'],
                'attacked_entities': ['Error in analysis'],
                'protected_entities': ['Error in analysis'],
                'toxicity_level': 0,
                'toxic_examples': ['Error in analysis', 'Error in analysis'],
                'emotional_tones': ['neutral', 'neutral'],
                'stated_goals': ['Error in analysis'],
                'psychological_profile': 'Error in analysis',
                'confidence_ratings': {
                    'narratives_confidence': 70,
                    'attacked_entities_confidence': 70,
                    'protected_entities_confidence': 70,
                    'emotional_tones_confidence': 70,
                    'stated_goals_confidence': 70,
                    'overall_confidence': 70
                },
                'confidence_explanation': 'Analysis encountered an error, results may be incomplete'
            }

    def _validate_json_structure(self, data: dict) -> bool:
        """Validate JSON structure and content"""
        try:
            required_fields = {
                'narratives': (list, 3),
                'attacked_entities': (list, 3),
                'protected_entities': (list, 3),
                'toxicity_level': ((int, float), 1),
                'toxic_examples': (list, 2),
                'emotional_tones': (list, 2),
                'stated_goals': (list, 3),
                'psychological_profile': (str, 1),
                'confidence_ratings': (dict, 1),
                'confidence_explanation': (str, 1)
            }
            
            for field, (field_type, expected_length) in required_fields.items():
                if field not in data:
                    return False
                if not isinstance(data[field], field_type):
                    return False
                if isinstance(data[field], list) and len(data[field]) > expected_length:
                    data[field] = data[field][:expected_length]
            
            # Ensure emotional tones are lowercase
            data['emotional_tones'] = [tone.lower() for tone in data['emotional_tones']]
            
            return True
        except:
            return False

    def _fix_truncated_json(self, text: str) -> dict:
        """Attempt to fix truncated JSON by completing the structure"""
        try:
            # Extract available fields
            narratives = re.findall(r'"narratives":\s*\[(.*?)\]', text, re.DOTALL)
            attacked = re.findall(r'"attacked_entities":\s*\[(.*?)\]', text, re.DOTALL)
            protected = re.findall(r'"protected_entities":\s*\[(.*?)\]', text, re.DOTALL)
            toxicity = re.findall(r'"toxicity_level":\s*(\d+)', text)
            
            if narratives and attacked and protected and toxicity:
                # Construct minimal valid JSON
                json_str = {
                    "narratives": json.loads(f"[{narratives[0]}]"),
                    "attacked_entities": json.loads(f"[{attacked[0]}]"),
                    "protected_entities": json.loads(f"[{protected[0]}]"),
                    "toxicity_level": int(toxicity[0]),
                    "toxic_examples": ["Not available due to truncation", "Not available due to truncation"],
                    "emotional_tones": ["neutral", "neutral"],
                    "stated_goals": ["Goal not available due to truncation"],
                    "psychological_profile": "Profile not available due to truncation"
                }
                return json_str
        except:
            pass
        return None

    def _analyze_batch(self, tweets: List[Dict], username: str, batch_id: int) -> Dict:
        """Analyze a batch of tweets with improved error handling"""
        formatted_tweets = self._format_tweets_batch(tweets)
        prompt = self._create_prompt(formatted_tweets, username)
        
        for attempt in range(self.max_retries):
            try:
                response = self.llm_client.invoke_model(
                    body=json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": 4096,  # Set to 4096 consistently
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1
                    }),
                    modelId=self.model_name,
                    accept='application/json',
                    contentType='application/json'
                )
                
                result = json.loads(response['body'].read().decode('utf-8'))
                text_response = result['content'][0]['text']
                
                # Extract and validate the analysis
                analysis = self._extract_json_with_regex(text_response)
                if analysis:
                    return {
                        'username': username,
                        'batch_id': batch_id,
                        **analysis
                    }
                
                print(f"\nRetrying batch {batch_id} due to invalid response format")
                
            except Exception as e:
                print(f"\nError in batch {batch_id} for {username}, attempt {attempt + 1}: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
    
    def analyze_user_tweets(self, username: str, tweets: List[Dict]) -> pd.DataFrame:
        """Analyze all tweets for a user in batches with minimal logging"""
        results = []
        total_batches = (len(tweets) + self.batch_size - 1) // self.batch_size
        
        print(f"\nAnalyzing tweets for @{username}")
        print(f"Total tweets: {len(tweets)}")
        print(f"Number of batches: {total_batches}")
        
        for i in range(0, len(tweets), self.batch_size):
            batch_id = i // self.batch_size
            batch = tweets[i:i + self.batch_size]
            
            try:
                analysis = self._analyze_batch(batch, username, batch_id)
                results.append(analysis)
                print(f"✓ Batch {batch_id + 1}/{total_batches} completed")
                
            except Exception as e:
                print(f"✗ Batch {batch_id + 1}/{total_batches} failed: {str(e)}")
                continue
        
        if results:
            print(f"\nCompleted analysis for @{username}: {len(results)} batches processed")
            return pd.DataFrame(results)
        else:
            print(f"\nNo successful analyses for @{username}")
            return pd.DataFrame() 

    def create_user_summaries(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create user summaries using LLM analysis of all batches"""
        merged_results = []
        
        for username, user_df in df.groupby('username'):
            # Format all batches in a clear way for the LLM
            all_batches = []
            for _, batch in user_df.iterrows():
                all_batches.append({
                    'batch_id': batch['batch_id'],
                    'narratives': batch['narratives'],
                    'attacked_entities': batch['attacked_entities'],
                    'protected_entities': batch['protected_entities'],
                    'toxicity_level': batch['toxicity_level'],
                    'toxic_examples': batch['toxic_examples'],
                    'emotional_tones': batch['emotional_tones'],
                    'stated_goals': batch['stated_goals']
                })
            
            summary_prompt = f"""Analyze ALL batches of tweets from user @{username}.
            You will see multiple batches, each containing multiple items.
            Your task is to identify the most frequently occurring patterns across ALL batches.

            ALL BATCHES DATA:
            {json.dumps(all_batches, indent=2, ensure_ascii=False)}

            INSTRUCTIONS:
            1. Look at ALL narratives across ALL batches (if there are 11 batches with 3 narratives each, you're analyzing 33 narratives)
            2. Find the most frequently repeated or similar items
            3. For toxicity, find the highest score and the two most toxic examples from any batch
            4. For entities, count specific names/organizations that appear most often across all batches

            Create a summary showing:
            1. Top 3 most FREQUENTLY OCCURRING narratives across all batches
            2. Top 3 most FREQUENTLY MENTIONED specific attacked entities
            3. Top 3 most FREQUENTLY MENTIONED specific protected entities
            4. Highest toxicity score and two most toxic examples
            5. Two most common emotional tones
            6. Top 3 most repeated stated goals

            Respond in this exact JSON format:
            {{
                "narratives": [3 most frequent narratives],
                "attacked_entities": [3 most frequent specific names/organizations],
                "protected_entities": [3 most frequent specific names/organizations],
                "toxicity_level": highest_score_found,
                "toxic_examples": [2 most toxic tweets in original Hebrew],
                "emotional_tones": [2 most frequent tones],
                "stated_goals": [3 most frequent goals],
                "psychological_profile": "brief summary based on all batches"
            }}

            IMPORTANT:
            - Count occurrences across ALL batches to find truly dominant patterns
            - For entities, use only specific names (e.g., "Benjamin Netanyahu", not "activists")
            - Keep original Hebrew text in toxic examples
            - Use lowercase emotional tones (neutral, angry, cynical, fearful, frustrated, optimistic)"""
            
            try:
                response = self._analyze_batch([{"text": summary_prompt, "created_at": ""}], 
                                             username, 0)
                if response:
                    response['username'] = username
                    response['batch_id'] = 0
                    merged_results.append(response)
                
            except Exception as e:
                print(f"Error analyzing user {username}: {str(e)}")
        
        # Create final DataFrame and save
        if merged_results:
            merged_df = pd.DataFrame(merged_results)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            merged_filename = f'user_analysis_merged_{timestamp}.csv'
            merged_df.to_csv(merged_filename, index=False)
            return merged_df
        else:
            return pd.DataFrame()

    def create_final_user_analysis(self, df_users: pd.DataFrame) -> pd.DataFrame:
        """Create final user analysis with LLM summarization"""
        def create_final_summary_prompt(user_data):
            return f"""Analyze these Twitter statistics for @{user_data['username']} and create a final summary.
            
            Previous Analysis Results:
            - Top Narratives: {', '.join(str(x) for x in user_data['top_narratives'])}
            - Most Attacked: {', '.join(str(x) for x in user_data['top_attacked_entities'])}
            - Most Protected: {', '.join(str(x) for x in user_data['top_protected_entities'])}
            - Average Toxicity: {user_data['avg_toxicity_level']:.1f}
            - Emotional Tones: {', '.join(str(x) for x in user_data['dominant_emotional_tones'])}
            - Main Goals: {', '.join(str(x) for x in user_data['top_stated_goals'])}
            
            Create a final analysis in this EXACT JSON format:
            {{
                "narratives": [
                    "narrative 1 (specific, max 10 words)",
                    "narrative 2 (specific, max 10 words)",
                    "narrative 3 (specific, max 10 words)"
                ],
                "attacked_entities": [
                    "specific name 1",
                    "specific name 2",
                    "specific name 3"
                ],
                "protected_entities": [
                    "specific name 1",
                    "specific name 2",
                    "specific name 3"
                ],
                "toxicity_level": number between 0-100,
                "toxic_examples": [
                    "example 1 (keep original Hebrew)",
                    "example 2 (keep original Hebrew)"
                ],
                "emotional_tones": [
                    "tone1",
                    "tone2"
                ],
                "stated_goals": [
                    "goal 1 (specific, max 10 words)",
                    "goal 2 (specific, max 10 words)",
                    "goal 3 (specific, max 10 words)"
                ],
                "psychological_profile": "2-3 sentence summary"
            }}

            IMPORTANT:
            1. Use ONLY specific names for entities (e.g., "Benjamin Netanyahu", "Yariv Levin")
            2. If no specific names available, use "No specific entity identified"
            3. Keep original Hebrew text in toxic_examples
            4. Emotional tones must be lowercase: neutral, angry, cynical, fearful, frustrated, optimistic
            5. All text must be concise and specific"""

        # First create raw summaries with statistics
        raw_merged_results = self.create_user_summaries(df_users)
        
        # Process each user's data through the LLM
        final_results = []
        for _, user_data in raw_merged_results.iterrows():
            try:
                prompt = create_final_summary_prompt(user_data)
                response = self._analyze_batch([{"text": str(user_data), "created_at": ""}], 
                                             user_data['username'], 0)
                if response:
                    # Ensure response has required fields
                    required_fields = {
                        'narratives': (list, 3),
                        'attacked_entities': (list, 3),
                        'protected_entities': (list, 3),
                        'toxicity_level': ((int, float), 1),
                        'toxic_examples': (list, 2),
                        'emotional_tones': (list, 2),
                        'stated_goals': (list, 3),
                        'psychological_profile': (str, 1)
                    }
                    
                    # Validate and clean response
                    cleaned_response = {}
                    for field, (field_type, length) in required_fields.items():
                        value = response.get(field)
                        if isinstance(value, field_type):
                            if isinstance(value, list):
                                cleaned_response[field] = value[:length]
                            else:
                                cleaned_response[field] = value
                        else:
                            # Use defaults if invalid
                            if field_type == list:
                                cleaned_response[field] = ['No specific data'] * length
                            elif field_type == (int, float):
                                cleaned_response[field] = 0
                            else:
                                cleaned_response[field] = 'No data available'
                    
                    final_results.append({
                        'username': user_data['username'],
                        'batch_id': 0,  # Final summary is always batch 0
                        **cleaned_response
                    })
                else:
                    print(f"Failed to analyze user: {user_data['username']}")
            except Exception as e:
                print(f"Error analyzing user {user_data['username']}: {str(e)}")
        
        # Convert to DataFrame and save with timestamp
        if final_results:
            final_df = pd.DataFrame(final_results)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Ensure data/raw directory exists
            os.makedirs('data/raw', exist_ok=True)
            
            # Save to data/raw directory
            merged_filename = f'user_analysis_merged_{timestamp}.csv'
            filepath = os.path.join('data', 'raw', merged_filename)
            final_df.to_csv(filepath, index=False)
            print(f"\nSaved merged analysis to: {filepath}")
            return final_df
        else:
            print("No successful analyses to save")
            return pd.DataFrame() 

    def merge_user_analyses(self, df: pd.DataFrame, period_label: str = None) -> pd.DataFrame:
        """Merge multiple batch analyses into single user summaries using LLM with confidence ratings
        Args:
            df: DataFrame containing batch analyses
            period_label: Optional period label (e.g., 'pre_war', 'post_war')
        Returns:
            DataFrame with merged analyses
        """
        final_results = []
        
        # Process one user at a time
        for username, user_batches in df.groupby('username'):
            print(f"\nMerging analyses for @{username}")
            print(f"Total batches to analyze: {len(user_batches)}")
            
            merge_prompt = f"""Analyze ALL batches of tweets from user @{username}.
            You have {len(user_batches)} batches to analyze, with multiple items in each batch.
            
            ALL BATCHES FOR THIS USER:
            {user_batches.to_dict('records')}
            
            Your task is to analyze ALL batches together and find the most common patterns.
            For example, if you see 11 batches with 3 narratives each, analyze all 33 narratives to find the top 3 most frequent ones.
            
            Create ONE summary showing:
            1. Top 3 most FREQUENTLY OCCURRING narratives across all batches
            2. Top 3 most FREQUENTLY MENTIONED specific attacked entities
            3. Top 3 most FREQUENTLY MENTIONED specific protected entities
            4. Highest toxicity score seen in any batch
            5. Two most toxic tweet examples (in original Hebrew)
            6. Two most common emotional tones
            7. Top 3 most frequently stated goals
            
            Also provide confidence ratings (0-100) for your analysis:
            - Higher confidence (>80) means clear patterns, frequent repetition of themes/entities
            - Medium confidence (50-80) means some patterns but with variations
            - Lower confidence (<50) means unclear patterns or high variability
            
            Respond in this exact JSON format:
            {{
                "narratives": [3 most frequent narratives],
                "attacked_entities": [3 most frequent specific names/organizations],
                "protected_entities": [3 most frequent specific names/organizations],
                "toxicity_level": highest_toxicity_score,
                "toxic_examples": [2 most toxic Hebrew tweets],
                "emotional_tones": [2 most frequent tones],
                "stated_goals": [3 most frequent goals],
                "psychological_profile": "brief summary based on all batches",
                "confidence_ratings": {{
                    "narratives_confidence": score (0-100),
                    "attacked_entities_confidence": score (0-100),
                    "protected_entities_confidence": score (0-100),
                    "emotional_tones_confidence": score (0-100),
                    "stated_goals_confidence": score (0-100),
                    "overall_confidence": score (0-100)
                }},
                "confidence_explanation": "Brief explanation of confidence scores"
            }}
            
            IMPORTANT:
            - Count occurrences across ALL batches to find truly dominant patterns
            - For entities, use ONLY specific names (e.g., "Benjamin Netanyahu", not "activists")
            - Keep original Hebrew text in toxic examples
            - Use lowercase emotional tones (neutral, angry, cynical, fearful, frustrated, optimistic)
            - Base confidence scores on:
              * Frequency of pattern repetition
              * Consistency across batches
              * Clarity of specific entities/themes
              * Amount of available data"""
            
            try:
                response = self._analyze_batch([{"text": merge_prompt, "created_at": ""}], 
                                             username, 0)
                if response:
                    response['username'] = username
                    response['batch_id'] = 0  # Merged summary
                    if period_label:
                        response['period'] = period_label
                    final_results.append(response)
                else:
                    print(f"Failed to merge analyses for {username}")
            except Exception as e:
                print(f"Error merging analyses for {username}: {str(e)}")
        
        # Create final DataFrame and save with timestamp and period label
        if final_results:
            final_df = pd.DataFrame(final_results)
            # Remove batch_id as it's no longer needed
            if 'batch_id' in final_df.columns:
                final_df = final_df.drop('batch_id', axis=1)
            
            # Save with timestamp and period label
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Determine save directory based on period_label
            if period_label:
                save_dir = os.path.join('data', 'analysis', period_label)
            else:
                save_dir = os.path.join('data', 'analysis')
                
            # Create directory if it doesn't exist
            os.makedirs(save_dir, exist_ok=True)
            
            # Create filename with period label if provided
            if period_label:
                filename = f'merged_analysis_{period_label}_{timestamp}.csv'
            else:
                filename = f'merged_analysis_{timestamp}.csv'
                
            # Save the merged analysis
            filepath = os.path.join(save_dir, filename)
            final_df.to_csv(filepath, index=False)
            print(f"\nSaved merged analysis to: {filepath}")
            
            return final_df
        else:
            print("No successful merges to save")
            return pd.DataFrame() 

    def _format_group_data(self, users_df: pd.DataFrame) -> str:
        """
        Format group data for LLM analysis
        Args:
            users_df: DataFrame containing merged results for users
        Returns:
            Formatted string with user data
        """
        formatted_text = "GROUP ANALYSIS DATA:\n\n"
        
        for _, user_data in users_df.iterrows():
            formatted_text += f"USER: @{user_data['username']}\n"
            formatted_text += f"Narratives: {user_data['narratives']}\n"
            formatted_text += f"Attacked Entities: {user_data['attacked_entities']}\n"
            formatted_text += f"Protected Entities: {user_data['protected_entities']}\n"
            formatted_text += f"Toxicity Level: {user_data['toxicity_level']}\n"
            formatted_text += f"Emotional Tones: {user_data['emotional_tones']}\n"
            formatted_text += f"Stated Goals: {user_data['stated_goals']}\n"
            formatted_text += "-" * 50 + "\n\n"
        
        return formatted_text

    def _create_group_analysis_prompt(self, formatted_data: str) -> str:
        """
        Create prompt for analyzing patterns across multiple users
        Args:
            formatted_data: Formatted string containing all users' data
        Returns:
            Analysis prompt
        """
        return f"""You are analyzing a group of Twitter users who are members of or associated with the Kohelet Forum.
        Your task is to identify common patterns and group-level characteristics across all users.
        
        {formatted_data}
        
        Analyze ALL users' data together and create a group-level summary that shows:
        1. Most common narratives shared across users
        2. Most frequently attacked entities across the group
        3. Most frequently protected entities across the group
        4. Group-level toxicity patterns
        5. Dominant emotional tones in the group
        6. Common goals shared across users
        
        Also assess group cohesion and pattern strength:
        - How aligned are users in their narratives?
        - How consistent are attack/defense patterns?
        - How uniform are emotional tones?
        
        Provide your analysis in this EXACT JSON format:
        {{
            "narratives": [
                "3 most common narratives across ALL users, max 10 words each"
            ],
            "attacked_entities": [
                "3 most frequently attacked specific entities across ALL users"
            ],
            "protected_entities": [
                "3 most frequently protected specific entities across ALL users"
            ],
            "toxicity_level": "group average toxicity (0-100)",
            "toxic_examples": [
                "2 most representative toxic examples from any user (keep Hebrew)"
            ],
            "emotional_tones": [
                "2 most common emotional tones across users"
            ],
            "stated_goals": [
                "3 most common goals across users, max 10 words each"
            ],
            "group_cohesion": {{
                "narrative_alignment": score (0-100),
                "entity_alignment": score (0-100),
                "emotional_alignment": score (0-100),
                "goals_alignment": score (0-100),
                "overall_cohesion": score (0-100)
            }},
            "confidence_ratings": {{
                "narratives_confidence": score (0-100),
                "attacked_entities_confidence": score (0-100),
                "protected_entities_confidence": score (0-100),
                "emotional_tones_confidence": score (0-100),
                "stated_goals_confidence": score (0-100),
                "overall_confidence": score (0-100)
            }},
            "group_profile": "2-3 sentence summary of group characteristics"
        }}
        
        IMPORTANT:
        - Focus on patterns that appear across MULTIPLE users
        - Use specific names for entities (e.g., "Benjamin Netanyahu", not "politicians")
        - Keep original Hebrew text in toxic examples
        - Base confidence scores on pattern consistency across users
        - Base cohesion scores on how aligned users are in each aspect"""

    def analyze_user_group(self, users_df: pd.DataFrame, group_name: str = "default") -> pd.DataFrame:
        """
        Analyze patterns across a group of users
        Args:
            users_df: DataFrame containing merged results for users (one row per user)
            group_name: Name/identifier for this group
        Returns:
            DataFrame with group-level analysis
        """
        print(f"\nAnalyzing group: {group_name}")
        print(f"Number of users in group: {len(users_df)}")
        
        # Format data for analysis
        formatted_data = self._format_group_data(users_df)
        prompt = self._create_group_analysis_prompt(formatted_data)
        
        try:
            # Analyze group data
            response = self.llm_client.invoke_model(
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 4096,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1
                }),
                modelId="anthropic.claude-3-haiku-20240307-v1:0",
                accept='application/json',
                contentType='application/json'
            )
            
            result = json.loads(response['body'].read().decode('utf-8'))
            text_response = result['content'][0]['text']
            
            # Extract and validate the analysis
            analysis = self._extract_json_with_regex(text_response)
            if analysis:
                analysis['group_name'] = group_name
                analysis['user_count'] = len(users_df)
                analysis['period'] = users_df['period'].iloc[0] if 'period' in users_df.columns else 'unknown'
                
                # Convert to DataFrame
                df_result = pd.DataFrame([analysis])
                
                # Save results
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                # Ensure group_analysis directory exists
                os.makedirs(os.path.join('data', 'group_analysis'), exist_ok=True)
                
                # Save to group_analysis directory
                filename = f'group_analysis_{group_name}_{timestamp}.csv'
                filepath = os.path.join('data', 'group_analysis', filename)
                df_result.to_csv(filepath, index=False)
                print(f"\nSaved group analysis to: {filepath}")
                
                return df_result
            else:
                print("Failed to extract valid analysis from LLM response")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error analyzing group: {str(e)}")
            return pd.DataFrame()

    def compare_group_periods(self, 
                            pre_period_data: pd.DataFrame, 
                            post_period_data: pd.DataFrame,
                            group_name: str = "default") -> pd.DataFrame:
        """
        Compare group behavior between two periods
        Args:
            pre_period_data: DataFrame with pre-period user data
            post_period_data: DataFrame with post-period user data
            group_name: Name/identifier for this group
        Returns:
            DataFrame with period comparison
        """
        print(f"\nComparing periods for group: {group_name}")
        
        # Add period labels
        pre_period_data['period'] = 'pre_war'
        post_period_data['period'] = 'post_war'
        
        # Analyze each period
        pre_analysis = self.analyze_user_group(
            users_df=pre_period_data,
            group_name=f"{group_name}_pre_war"
        )
        
        post_analysis = self.analyze_user_group(
            users_df=post_period_data,
            group_name=f"{group_name}_post_war"
        )
        
        if not pre_analysis.empty and not post_analysis.empty:
            try:
                # Create comparison data
                comparison = {
                    'group_name': group_name,
                    'pre_war_narratives': pre_analysis['narratives'].iloc[0],
                    'post_war_narratives': post_analysis['narratives'].iloc[0],
                    'pre_war_attacked': pre_analysis['attacked_entities'].iloc[0],
                    'post_war_attacked': post_analysis['attacked_entities'].iloc[0],
                    'pre_war_protected': pre_analysis['protected_entities'].iloc[0],
                    'post_war_protected': post_analysis['protected_entities'].iloc[0],
                    'toxicity_change': float(post_analysis['toxicity_level'].iloc[0]) - float(pre_analysis['toxicity_level'].iloc[0]),
                    'pre_war_tones': pre_analysis['emotional_tones'].iloc[0],
                    'post_war_tones': post_analysis['emotional_tones'].iloc[0],
                    'pre_war_goals': pre_analysis['stated_goals'].iloc[0],
                    'post_war_goals': post_analysis['stated_goals'].iloc[0],
                    'pre_war_profile': pre_analysis['psychological_profile'].iloc[0],
                    'post_war_profile': post_analysis['psychological_profile'].iloc[0]
                }
                
                # Convert to DataFrame
                comparison_df = pd.DataFrame([comparison])
                
                # Save comparison
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                # Ensure group_analysis directory exists
                os.makedirs(os.path.join('data', 'group_analysis'), exist_ok=True)
                
                # Save to group_analysis directory
                filename = f'group_period_comparison_{group_name}_{timestamp}.csv'
                filepath = os.path.join('data', 'group_analysis', filename)
                comparison_df.to_csv(filepath, index=False)
                print(f"\nSaved period comparison to: {filepath}")
                
                return comparison_df
            except Exception as e:
                print(f"Error creating comparison: {str(e)}")
                return pd.DataFrame()
        else:
            print("Failed to generate period comparison due to missing analysis")
            return pd.DataFrame() 

    def _create_analysis_prompt(self, tweets_batch):
        """Create a prompt for basic tweet analysis."""
        prompt = f"""Analyze these tweets and provide scores and explanations for the following metrics.
All scores should be on a scale of 0-100:

1. Toxicity Level (0-100):
   - 0: Not toxic at all
   - 25: Mildly toxic
   - 50: Moderately toxic
   - 75: Very toxic
   - 100: Extremely toxic

2. Emotional Tones (select all that apply):
   - angry
   - frustrated
   - supportive
   - optimistic
   - concerned
   - critical
   - neutral
   - cynical

For each metric, provide:
1. A score (0-100) for toxicity
2. Examples of toxic tweets (if any)
3. List of emotional tones
4. A confidence rating (0-100)

Also identify:
1. Main narratives being promoted
2. Entities being criticized (attacked_entities)
3. Entities being defended (protected_entities)
4. Stated goals or objectives
5. Brief psychological profile (max 25 words)

Tweets to analyze:
{tweets_batch}

Respond in this JSON format:
{{
    "toxicity_level": <0-100>,
    "toxic_examples": ["example1", "example2"],
    "emotional_tones": ["tone1", "tone2"],
    "narratives": ["narrative1", "narrative2"],
    "attacked_entities": ["entity1", "entity2"],
    "protected_entities": ["entity1", "entity2"],
    "stated_goals": ["goal1", "goal2"],
    "psychological_profile": "brief profile",
    "confidence_ratings": {{
        "narratives_confidence": <0-100>,
        "entities_confidence": <0-100>,
        "toxicity_confidence": <0-100>
    }},
    "confidence_explanation": ["reason1", "reason2"]
}}"""
        return prompt 