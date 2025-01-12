import json
import re
import time
from typing import List, Dict
import pandas as pd
from analyzer import TweetAnalyzer
import numpy as np
import os
from datetime import datetime
import ast
from jsonschema import validate, ValidationError

# Add debug logging function
def debug_print(prefix: str, message: str, data: any = None):
    print(f"\n[DEBUG] {prefix}: {message}")
    if data is not None:
        print(f"Data: {data}")

class EnhancedTweetAnalyzer(TweetAnalyzer):
    def __init__(self, batch_size=25, max_retries=3):
        super().__init__(batch_size=batch_size, max_retries=max_retries)
        self.security_metrics = {
            'twitter_analysis': {'field': 'security_vs_judicial', 'score_range': (0, 100)},
            'policy_focus': {'field': 'security_vs_economic', 'score_range': (0, 100)},
            'oversight_tone': {'field': 'government_oversight', 'score_range': (1, 10)},
            'rights_vs_security': {'field': 'individual_vs_collective', 'score_range': (-10, 10)},
            'international_law': {'field': 'international_law_refs', 'score_range': (-100, 100)},
            'defense_spending': {'field': 'defense_budget_priority', 'score_range': (1, 10)},
            'emergency_powers': {'field': 'executive_powers', 'score_range': (-5, 5)},
            'reform_priority': {'field': 'judicial_vs_security', 'score_range': (0, 100)},
            'civil_liberties': {'field': 'emergency_restrictions', 'score_range': (1, 10)},
            'policy_focus_ratio': {'field': 'domestic_vs_international', 'score_range': (-100, 100)}
        }
        self.response_schema = {
            "type": "object",
            "required": [
                "narratives",
                "attacked_entities",
                "protected_entities",
                "toxicity_level",
                "emotional_tones"
            ],
            "properties": {
                "narratives": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 3
                },
                "attacked_entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 3
                },
                "protected_entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 3
                },
                "toxicity_level": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 100
                },
                "emotional_tones": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["neutral", "angry", "cynical", "fearful", "frustrated", "optimistic"]
                    },
                    "minItems": 1,
                    "maxItems": 2
                }
            },
            "additionalProperties": False
        }

    def _extract_json_with_regex(self, text: str) -> dict:
        """Enhanced version of JSON extraction with regex fallback and debugging."""
        debug_print("JSON_EXTRACT", "Starting JSON extraction")
        debug_print("JSON_EXTRACT", "Input text first 500 chars:", text[:500])
        
        try:
            # First try to find and parse JSON directly
            json_matches = list(re.finditer(r'\{(?:[^{}]|(?:\{[^{}]*\})*)*\}', text))
            if not json_matches:
                json_matches = list(re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text))
            
            debug_print("JSON_EXTRACT", f"Found {len(json_matches)} potential JSON matches")
            
            for i, match in enumerate(json_matches):
                try:
                    json_str = match.group(0)
                    debug_print("JSON_EXTRACT", f"Attempting to parse match {i}", json_str[:200])
                    
                    # Convert all theme words to lowercase before other cleanup
                    json_str = re.sub(r'"(sec|pol)_themes"\s*:\s*\[(.*?)\]', 
                                    lambda m: f'"{m.group(1)}_themes": [{m.group(2).lower()}]', 
                                    json_str)
                    
                    # Regular cleanup
                    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                    json_str = re.sub(r':\s*([^",\{\}\[\]\s][^,\{\}\[\]]*?)(\s*[,\}\]])', r': "\1"\2', json_str)
                    json_str = re.sub(r'(?<!")(\b\w+\b)(?!")(?=\s*:)', r'"\1"', json_str)
                    json_str = re.sub(r'\[\s*([^"\[\]]+?)\s*\]', lambda m: '[' + ','.join(f'"{x.strip().lower()}"' for x in m.group(1).split(',')) + ']', json_str)
                    json_str = re.sub(r'"\s*,\s*"', '", "', json_str)
                    json_str = re.sub(r'\s+', ' ', json_str)
                    
                    debug_print("JSON_EXTRACT", "Cleaned JSON string:", json_str[:200])
                    
                    result = json.loads(json_str)
                    if isinstance(result, dict):
                        debug_print("JSON_EXTRACT", "Successfully parsed JSON. Keys found:", list(result.keys()))
                        
                        # Map shortened field names to original names
                        field_mapping = {
                            'narr': 'narratives',
                            'atk': 'attacked_entities',
                            'prot': 'protected_entities',
                            'tox': 'toxicity_level',
                            'tone': 'emotional_tones',
                            'goals': 'stated_goals',
                            'profile': 'psychological_profile',
                            'sec_themes': 'security_themes',
                            'pol_themes': 'policy_themes',
                            'conf': 'confidence_ratings'
                        }
                        
                        # Convert shortened names to original names
                        mapped_result = {}
                        for key, value in result.items():
                            if key in field_mapping:
                                mapped_result[field_mapping[key]] = value
                            elif key == 'metrics':
                                # Handle nested metrics
                                metrics_mapping = {
                                    'sec_jud': 'security_vs_judicial',
                                    'gov_over': 'government_oversight',
                                    'ind_col': 'individual_vs_collective',
                                    'emerg': 'emergency_restrictions',
                                    'dom_intl': 'domestic_vs_international'
                                }
                                for metric_key, metric_value in value.items():
                                    if metric_key in metrics_mapping:
                                        mapped_result[metrics_mapping[metric_key]] = {
                                            'analysis': metric_value['desc'],
                                            'score': metric_value['score']
                                        }
                            else:
                                mapped_result[key] = value
                        
                        # Ensure themes are lowercase
                        if 'security_themes' in mapped_result:
                            mapped_result['security_themes'] = [theme.lower() for theme in mapped_result['security_themes']]
                        if 'policy_themes' in mapped_result:
                            mapped_result['policy_themes'] = [theme.lower() for theme in mapped_result['policy_themes']]
                        
                        # Check if this is a complete response with required fields
                        required_fields = {'narratives', 'attacked_entities', 'protected_entities', 'toxicity_level', 'emotional_tones'}
                        if required_fields.issubset(set(mapped_result.keys())):
                            return mapped_result
                        else:
                            debug_print("JSON_EXTRACT", f"Missing required fields in match {i}. Found keys:", list(mapped_result.keys()))
                            continue
                            
                except Exception as e:
                    debug_print("JSON_EXTRACT", f"Error parsing match {i}:", str(e))
                    continue
            
            debug_print("JSON_EXTRACT", "JSON parsing failed, attempting regex fallback")
            
            # Regex fallback patterns
            fallback_result = {}
            
            # Extract narratives
            narr_pattern = r'Narratives?:?\s*(?:\d\.\s*)?(.*?)(?=\n\n|\n[A-Z]|$)'
            narr_matches = re.findall(narr_pattern, text, re.DOTALL)
            if narr_matches:
                narratives = []
                for match in narr_matches[:3]:  # Take up to 3 narratives
                    items = re.split(r'\d\.\s*', match.strip())
                    narratives.extend([item.strip() for item in items if item.strip()])
                fallback_result['narratives'] = narratives[:3]
            
            # Extract attacked entities
            atk_pattern = r'Attacked Entities:?\s*(?:\d\.\s*)?(.*?)(?=\n\n|\n[A-Z]|$)'
            atk_matches = re.findall(atk_pattern, text, re.DOTALL)
            if atk_matches:
                entities = []
                for match in atk_matches[:3]:
                    items = re.split(r'\d\.\s*', match.strip())
                    entities.extend([item.strip() for item in items if item.strip()])
                fallback_result['attacked_entities'] = entities[:3]
            
            # Extract protected entities
            prot_pattern = r'Protected Entities:?\s*(?:\d\.\s*)?(.*?)(?=\n\n|\n[A-Z]|$)'
            prot_matches = re.findall(prot_pattern, text, re.DOTALL)
            if prot_matches:
                entities = []
                for match in prot_matches[:3]:
                    items = re.split(r'\d\.\s*', match.strip())
                    entities.extend([item.strip() for item in items if item.strip()])
                fallback_result['protected_entities'] = entities[:3]
            
            # Extract emotional tones
            tone_pattern = r'Emotional Tones?:?\s*(?:\d\.\s*)?(.*?)(?=\n\n|\n[A-Z]|$)'
            tone_matches = re.findall(tone_pattern, text, re.DOTALL)
            if tone_matches:
                tones = []
                for match in tone_matches[:2]:
                    items = re.split(r'\d\.\s*', match.strip())
                    tones.extend([item.strip().lower() for item in items if item.strip()])
                fallback_result['emotional_tones'] = tones[:2]
            
            # Add default values for required fields if missing
            if 'narratives' not in fallback_result:
                fallback_result['narratives'] = ["No clear narrative identified"]
            if 'attacked_entities' not in fallback_result:
                fallback_result['attacked_entities'] = ["Unspecified"]
            if 'protected_entities' not in fallback_result:
                fallback_result['protected_entities'] = ["Unspecified"]
            if 'emotional_tones' not in fallback_result:
                fallback_result['emotional_tones'] = ["neutral"]
            if 'toxicity_level' not in fallback_result:
                fallback_result['toxicity_level'] = 50
            
            debug_print("JSON_EXTRACT", "Regex fallback results:", fallback_result)
            return fallback_result
            
        except Exception as e:
            debug_print("JSON_EXTRACT", "Fatal error in JSON extraction:", str(e))
            import traceback
            traceback.print_exc()
            return {
                'narratives': ["Error in analysis"],
                'attacked_entities': ["Unspecified"],
                'protected_entities': ["Unspecified"],
                'emotional_tones': ["neutral"],
                'toxicity_level': 50
            }

    def _validate_json_structure(self, data: dict) -> bool:
        """Validate JSON structure using JSON Schema"""
        try:
            validate(instance=data, schema=self.response_schema)
            
            # Additional validation for emotional tones
            data['emotional_tones'] = [tone.lower() for tone in data['emotional_tones']]
            
            return True
        except ValidationError as e:
            debug_print("JSON_VALIDATION", f"Schema validation error: {str(e)}")
            return False
        except Exception as e:
            debug_print("JSON_VALIDATION", f"Unexpected validation error: {str(e)}")
            return False

    def _create_enhanced_prompt(self, formatted_tweets: str, username: str) -> str:
        """Create simplified prompt focusing on core metrics"""
        schema_example = {
            "narratives": ["narrative1", "narrative2", "narrative3"],
            "attacked_entities": ["entity1", "entity2", "entity3"],
            "protected_entities": ["entity1", "entity2", "entity3"],
            "toxicity_level": 75,
            "emotional_tones": ["angry", "frustrated"]
        }

        return f"""Analyze these tweets from user @{username} and provide a structured analysis in JSON format.
        
        IMPORTANT: Respond with ONLY a valid JSON object matching this exact structure:
        {json.dumps(schema_example, indent=2)}
        
        Focus on:
        1. Main narratives (up to 3)
        2. Entities that are criticized or attacked
        3. Entities that are defended or protected
        4. Overall toxicity level (0-100)
        5. Emotional tones (choose 1-2: neutral, angry, cynical, fearful, frustrated, optimistic)
        
        Tweets to analyze:
        {formatted_tweets}"""

    def _analyze_batch(self, tweets: List[Dict], username: str, batch_id: int) -> Dict:
        """Override analyze_batch to use enhanced prompt with debugging"""
        debug_print("ANALYZE_BATCH", f"Starting analysis for user {username}, batch {batch_id}")
        
        formatted_tweets = self._format_tweets_batch(tweets)
        prompt = self._create_enhanced_prompt(formatted_tweets, username)
        debug_print("ANALYZE_BATCH", "Created prompt", prompt[:200])
        
        for attempt in range(self.max_retries):
            try:
                debug_print("ANALYZE_BATCH", f"Attempt {attempt + 1}/{self.max_retries}")
                
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
                debug_print("ANALYZE_BATCH", "Got response from LLM", text_response[:500])
                
                # Extract and validate the enhanced analysis
                analysis = self._extract_json_with_regex(text_response)
                debug_print("ANALYZE_BATCH", "Extracted analysis", analysis)
                
                if analysis:
                    # Add security metrics with default values
                    security_metrics = {
                        'security_vs_judicial': {'score': 50, 'analysis': 'No specific analysis available'},
                        'government_oversight': {'score': 5, 'analysis': 'No specific analysis available'},
                        'individual_vs_collective': {'score': 0, 'analysis': 'No specific analysis available'},
                        'emergency_restrictions': {'score': 5, 'analysis': 'No specific analysis available'},
                        'domestic_vs_international': {'score': 0, 'analysis': 'No specific analysis available'}
                    }
                    
                    # Extract security metrics from the analysis if available
                    if 'security_analysis' in analysis:
                        for metric, data in analysis['security_analysis'].items():
                            if metric in security_metrics and isinstance(data, dict):
                                security_metrics[metric] = {
                                    'score': self._normalize_security_score(metric, data.get('score', security_metrics[metric]['score'])),
                                    'analysis': data.get('analysis', security_metrics[metric]['analysis'])
                                }
                    
                    # Add the security metrics to the analysis
                    analysis.update(security_metrics)
                    
                    # Ensure required fields are present with defaults
                    if 'security_themes' not in analysis or not analysis['security_themes']:
                        debug_print("ANALYZE_BATCH", "Using default security themes")
                        analysis['security_themes'] = ['border security', 'counter-terrorism']
                    if 'policy_themes' not in analysis or not analysis['policy_themes']:
                        debug_print("ANALYZE_BATCH", "Using default policy themes")
                        analysis['policy_themes'] = ['judicial reform', 'economic policy']
                    if 'confidence_ratings' not in analysis or not analysis['confidence_ratings']:
                        debug_print("ANALYZE_BATCH", "Using default confidence ratings")
                        analysis['confidence_ratings'] = {
                            'overall': 0.8,
                            'narratives': 0.85,
                            'entities': 0.75,
                            'toxicity': 0.75
                        }
                    if 'confidence_explanation' not in analysis or not analysis['confidence_explanation']:
                        debug_print("ANALYZE_BATCH", "Using default confidence explanation")
                        analysis['confidence_explanation'] = "Analysis based on consistent patterns in the tweet data."
                    
                    final_result = {
                        'username': username,
                        'batch_id': batch_id,
                        **analysis
                    }
                    debug_print("ANALYZE_BATCH", f"Final result keys for batch {batch_id}:", list(final_result.keys()))
                    return final_result
                
                debug_print("ANALYZE_BATCH", f"Invalid response format in attempt {attempt + 1}")
                
            except Exception as e:
                debug_print("ANALYZE_BATCH", f"Error in attempt {attempt + 1}:", str(e))
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)

    def _normalize_security_score(self, metric: str, score: float) -> float:
        """Normalize security metric scores to their expected ranges"""
        ranges = {
            'security_vs_judicial': (0, 100),
            'government_oversight': (1, 10),
            'individual_vs_collective': (-10, 10),
            'emergency_restrictions': (1, 10),
            'domestic_vs_international': (-100, 100)
        }
        
        if metric in ranges:
            min_val, max_val = ranges[metric]
            return max(min_val, min(max_val, float(score)))

    def _calculate_average_security_scores(self, user_batches: pd.DataFrame) -> dict:
        """
        Calculate average scores for numeric security metrics across batches.
        Preserves both scores and analysis text, and adds score-only fields for compatibility.
        """
        security_metrics = [
            'security_vs_judicial',
            'government_oversight',
            'individual_vs_collective',
            'emergency_restrictions',
            'domestic_vs_international'
        ]
        
        avg_scores = {}
        for metric in security_metrics:
            scores = []
            analyses = []
            for _, batch in user_batches.iterrows():
                if metric in batch:
                    metric_data = batch[metric]
                    if isinstance(metric_data, str):
                        try:
                            metric_data = ast.literal_eval(metric_data)
                        except:
                            continue
                    
                    if isinstance(metric_data, dict):
                        if 'score' in metric_data:
                            scores.append(float(metric_data['score']))
                        if 'analysis' in metric_data:
                            analyses.append(metric_data['analysis'])
            
            if scores:
                # Store both full metric data and score-only field
                avg_scores[metric] = {
                    'score': sum(scores) / len(scores),
                    'analysis': analyses[-1] if analyses else "No analysis available"  # Use the most recent analysis
                }
                # Add score-only field for backward compatibility
                avg_scores[f"{metric}_score"] = sum(scores) / len(scores)
            else:
                avg_scores[metric] = {
                    'score': 0,
                    'analysis': "No analysis available"
                }
                avg_scores[f"{metric}_score"] = 0
        
        return avg_scores

    def _calculate_group_averages(self, period_data: pd.DataFrame) -> Dict:
        """
        Calculate averages for all numerical fields in the data
        
        Args:
            period_data: DataFrame containing user analyses
        Returns:
            Dictionary of averaged numerical values
        """
        numerical_fields = {
            'toxicity_level': 'mean',
            'confidence_ratings': 'mean',
        }
        
        averages = {}
        for field, agg_func in numerical_fields.items():
            if field in period_data.columns:
                try:
                    if field == 'confidence_ratings':
                        # Handle confidence ratings which might be stored as JSON
                        values = []
                        for rating in period_data[field]:
                            try:
                                if isinstance(rating, str):
                                    rating_dict = json.loads(rating)
                                    values.append(rating_dict.get('overall_confidence', 0))
                            except:
                                continue
                        if values:
                            averages[field] = sum(values) / len(values)
                    else:
                        values = pd.to_numeric(period_data[field], errors='coerce')
                        averages[field] = getattr(values, agg_func)()
                except Exception as e:
                    print(f"Error calculating average for {field}: {str(e)}")
                    averages[field] = 0
        
        return averages

    def _merge_group_texts(self, period_data: pd.DataFrame) -> Dict:
        """
        Merge and rank textual fields across the group
        """
        text_fields = {
            'narratives': 3,
            'attacked_entities': 3,
            'protected_entities': 3,
            'emotional_tones': 2,
            'stated_goals': 3,
            'toxic_examples': 2,
            'psychological_profile': 1,
            'security_themes': 3,
            'policy_themes': 3
        }
        
        # Default themes if extraction fails - all lowercase
        default_themes = {
            'security_themes': ['border security', 'counter-terrorism', 'national security'],
            'policy_themes': ['judicial reform', 'economic policy', 'civil rights']
        }
        
        # Theme fields that should always be lowercase
        theme_fields = {'security_themes', 'policy_themes'}
        
        merged_texts = {}
        for field, top_n in text_fields.items():
            if field in period_data.columns:
                try:
                    all_items = []
                    for items in period_data[field]:
                        try:
                            # Convert to lowercase immediately for theme fields
                            if field in theme_fields and isinstance(items, str):
                                items = items.lower()
                            
                            # Handle various input formats
                            if isinstance(items, str):
                                # Try parsing as JSON first
                                try:
                                    # Convert to lowercase before parsing if it's a theme field
                                    if field in theme_fields:
                                        items = items.lower()
                                    parsed_items = json.loads(items.replace("'", '"'))
                                    if isinstance(parsed_items, list):
                                        items = parsed_items
                                    else:
                                        items = ast.literal_eval(items)
                                except:
                                    if items.startswith('[') and items.endswith(']'):
                                        try:
                                            # Convert to lowercase before parsing if it's a theme field
                                            if field in theme_fields:
                                                items = items.lower()
                                            items = items.replace("'", '"')
                                            items = re.sub(r'(?<!")(\b\w+\b)(?!")', r'"\1"', items)
                                            items = json.loads(items)
                                        except:
                                            items = [items]
                                    else:
                                        items = [items]
                            
                            if isinstance(items, list):
                                cleaned_items = []
                                for item in items:
                                    if isinstance(item, str):
                                        # Always lowercase for themes
                                        if field in theme_fields:
                                            item = item.lower()
                                        item = item.strip()
                                        if item and not item.startswith(',') and not item.isspace():
                                            # For themes, store lowercase version for both comparison and display
                                            if field in theme_fields:
                                                cleaned_items.append((item.lower(), item.lower()))
                                            else:
                                                cleaned_items.append((item.lower(), item))
                                all_items.extend(cleaned_items)
                            elif items:
                                if isinstance(items, str):
                                    if field in theme_fields:
                                        items = items.lower()
                                    items = items.strip()
                                    if items and not items.startswith(',') and not items.isspace():
                                        if field in theme_fields:
                                            all_items.append((items.lower(), items.lower()))
                                        else:
                                            all_items.append((items.lower(), items))
                                else:
                                    item_str = str(items)
                                    if field in theme_fields:
                                        item_str = item_str.lower()
                                    if field in theme_fields:
                                        all_items.append((item_str.lower(), item_str.lower()))
                                    else:
                                        all_items.append((item_str.lower(), item_str))
                        except Exception as e:
                            print(f"Error processing item in {field}: {str(e)}")
                            continue
                    
                    # Filter out empty strings and empty lists
                    all_items = [(lower, original) for lower, original in all_items 
                               if lower and lower != '[]' and not lower.isspace()]
                    
                    if all_items:
                        from collections import Counter
                        counts = Counter(lower for lower, _ in all_items)
                        
                        # For themes, always use lowercase version
                        if field in theme_fields:
                            merged_texts[field] = [item.lower() for item, _ in counts.most_common(top_n)]
                        else:
                            # Create mapping of lowercase to original case for non-theme fields
                            case_mapping = {}
                            for lower, original in all_items:
                                if lower not in case_mapping:
                                    case_mapping[lower] = original
                            merged_texts[field] = [case_mapping[item] for item, _ in counts.most_common(top_n)]
                        
                        print(f"\nDebug - {field} items found:")
                        for item, count in counts.most_common(top_n):
                            print(f"  {item}: {count}")
                    else:
                        print(f"\nNo {field} found")
                        if field in default_themes:
                            merged_texts[field] = default_themes[field]
                            print(f"Using default {field}")
                        else:
                            merged_texts[field] = []
                except Exception as e:
                    print(f"Error merging text field {field}: {str(e)}")
                    if field in default_themes:
                        merged_texts[field] = default_themes[field]
                        print(f"Using default {field} after error")
                    else:
                        merged_texts[field] = []
        
        return merged_texts

    def merge_enhanced_user_analyses(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge multiple batch analyses into single user summaries with enhanced security metrics.
        Uses LLM for textual analysis (on 3 representative batches) and averaging for numeric scores (on all batches).
        """
        final_results = []
        
        # Process one user at a time
        for username, user_batches in df.groupby('username'):
            print(f"\nMerging analyses for @{username} ({len(user_batches)} batches)")
            
            try:
                # Calculate averaged numeric scores across ALL batches
                avg_scores = self._calculate_average_security_scores(user_batches)
                
                # Calculate average toxicity
                toxicity_scores = []
                for _, batch in user_batches.iterrows():
                    if isinstance(batch['toxicity_level'], (int, float)):
                        toxicity_scores.append(batch['toxicity_level'])
                avg_toxicity = sum(toxicity_scores) / len(toxicity_scores) if toxicity_scores else 0
                
                # Extract and merge textual fields from ALL batches
                merged_texts = self._merge_group_texts(user_batches)
                
                # Use LLM to analyze the representative batches
                merge_prompt = self._create_enhanced_merge_prompt(username, user_batches)
                print(f"Analyzing patterns across batches...")
                llm_analysis = self._analyze_batch([{"text": merge_prompt, "created_at": ""}], username, 0)
                
                if llm_analysis:
                    # Create the final result
                    final_result = {
                        'username': username,
                        'toxicity_level': avg_toxicity,
                        # Text fields from merged_texts
                        'narratives': merged_texts.get('narratives', []),
                        'emotional_tones': merged_texts.get('emotional_tones', []),
                        'attacked_entities': merged_texts.get('attacked_entities', []),
                        'protected_entities': merged_texts.get('protected_entities', []),
                        'stated_goals': merged_texts.get('stated_goals', []),
                        'toxic_examples': merged_texts.get('toxic_examples', []),
                        'psychological_profile': merged_texts.get('psychological_profile', []),
                        # Fields from LLM analysis
                        'security_themes': llm_analysis.get('security_themes', []),
                        'policy_themes': llm_analysis.get('policy_themes', []),
                        'confidence_ratings': llm_analysis.get('confidence_ratings', {}),
                        'confidence_explanation': llm_analysis.get('confidence_explanation', '')
                    }
                    
                    # Add both numeric scores and LLM-merged analysis
                    security_metrics = ['security_vs_judicial', 'government_oversight', 'individual_vs_collective', 
                                     'emergency_restrictions', 'domestic_vs_international']
                    
                    for metric in security_metrics:
                        if metric in avg_scores:
                            final_result[metric] = avg_scores[metric]
                            final_result[f"{metric}_score"] = avg_scores[f"{metric}_score"]
                    
                    final_results.append(final_result)
                    print(f"Successfully processed @{username}")
                    
                    # Debug output for LLM analysis fields
                    print("\nDebug - LLM Analysis fields:")
                    print(f"Security Themes: {final_result['security_themes']}")
                    print(f"Policy Themes: {final_result['policy_themes']}")
                    print(f"Confidence Ratings: {final_result['confidence_ratings']}")
                    print(f"Confidence Explanation: {final_result['confidence_explanation']}")
                else:
                    print(f"Failed to get LLM analysis for {username}")
                    continue
                
            except Exception as e:
                print(f"Error merging analyses for {username}: {str(e)}")
                continue
        
        return pd.DataFrame(final_results)

    def _create_enhanced_merge_prompt(self, username: str, user_batches: pd.DataFrame) -> str:
        """Create prompt for merging enhanced security metrics across representative batches."""
        # Extract just the key metrics we need to analyze
        metrics_to_analyze = ['security_vs_judicial', 'government_oversight', 
                            'individual_vs_collective', 'emergency_restrictions', 
                            'domestic_vs_international']
        
        # Create a summary of the metrics from representative batches
        metric_summaries = []
        for metric in metrics_to_analyze:
            analyses = []
            for _, batch in user_batches.iterrows():
                if metric in batch and isinstance(batch[metric], dict):
                    analyses.append(batch[metric]['analysis'])
            
            if analyses:
                summary = f"\n{metric}:\n" + "\n".join(f"- {analysis}" for analysis in analyses)
                metric_summaries.append(summary)
        
        prompt = f"""Analyze the patterns in tweets from user @{username}.
        You have {len(user_batches)} batches to analyze. These batches were selected to show the range of behavior.
        Here are the analyses from these batches:
        
        {''.join(metric_summaries)}
        
        Your task is to analyze these patterns and provide a comprehensive summary.
        Focus on identifying the most dominant themes and patterns in:

        1. Security Themes (e.g., military operations, counter-terrorism, border security)
        2. Policy Themes (e.g., judicial reform, economic policy, civil rights)
        3. Security vs Judicial Balance
        4. Government Oversight
        5. Individual vs Collective Rights
        6. Emergency Powers
        7. Domestic vs International Focus
        
        Provide your analysis in this EXACT JSON format:
        {{
            "security_themes": ["theme1", "theme2", "theme3"],
            "policy_themes": ["theme1", "theme2", "theme3"],
            "security_analysis": {{
                "security_vs_judicial": {{
                    "analysis": "comprehensive analysis of patterns",
                    "dominant_themes": ["theme1", "theme2", "theme3"]
                }},
                "government_oversight": {{
                    "analysis": "comprehensive analysis of patterns",
                    "dominant_themes": ["theme1", "theme2", "theme3"]
                }},
                "individual_vs_collective": {{
                    "analysis": "comprehensive analysis of patterns",
                    "dominant_themes": ["theme1", "theme2", "theme3"]
                }},
                "emergency_powers": {{
                    "analysis": "comprehensive analysis of patterns",
                    "dominant_themes": ["theme1", "theme2", "theme3"]
                }},
                "domestic_vs_international": {{
                    "analysis": "comprehensive analysis of patterns",
                    "dominant_themes": ["theme1", "theme2", "theme3"]
                }}
            }},
            "confidence_ratings": {{
                "security_metrics_confidence": score (0-100),
                "pattern_consistency": score (0-100),
                "overall_confidence": score (0-100)
            }},
            "confidence_explanation": "Brief explanation of confidence scores"
        }}
        
        IMPORTANT:
        - Focus on identifying the most significant and recurring themes
        - Consider how these themes interact with each other
        - Base confidence scores on the consistency of patterns across examples"""
        
        return prompt

    def calculate_group_metrics(self, period_data: pd.DataFrame) -> Dict:
        """
        Calculate aggregate security metrics for the group
        
        Args:
            period_data: DataFrame containing user analyses for a period
        Returns:
            Dictionary containing group-level metrics
        """
        # First deduplicate by username
        period_data = period_data.drop_duplicates(subset=['username'])
        print(f"\nProcessing {len(period_data)} unique users")
        
        # Initialize metrics dictionary
        group_stats = {}
        
        # Process security metrics
        security_metrics = [
            'security_vs_judicial',
            'government_oversight',
            'individual_vs_collective',
            'emergency_restrictions',
            'domestic_vs_international'
        ]
        
        # Process each security metric
        for metric in security_metrics:
            try:
                values = []
                for _, row in period_data.iterrows():
                    try:
                        # Try score column first
                        score_col = f'{metric}_score'
                        if score_col in row and pd.notnull(row[score_col]):
                            values.append(float(row[score_col]))
                            continue
                        
                        # If no score column, try metric dictionary
                        if metric in row:
                            metric_data = row[metric]
                            if isinstance(metric_data, str):
                                try:
                                    metric_data = ast.literal_eval(metric_data)
                                except:
                                    continue
                            
                            if isinstance(metric_data, dict) and 'score' in metric_data:
                                values.append(float(metric_data['score']))
                                
                    except Exception as e:
                        print(f"Error processing row for {metric}: {str(e)}")
                        continue

                if values:
                    values = np.array(values)
                    print(f"Found {len(values)} unique values for {metric}: {values}")
                    group_stats[metric] = {
                        'mean': float(np.mean(values)),
                        'median': float(np.median(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'consensus_level': float(self._calculate_consensus(values))
                    }
                else:
                    print(f"No valid values found for {metric}")
                    group_stats[metric] = {
                        'mean': 0.0,
                        'median': 0.0,
                        'std': 0.0,
                        'min': 0.0,
                        'max': 0.0,
                        'consensus_level': 0.0
                    }
            except Exception as e:
                print(f"Error processing metric {metric}: {str(e)}")
                group_stats[metric] = {
                    'mean': 0.0,
                    'median': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'consensus_level': 0.0
                }
        
        # Calculate group toxicity stats
        try:
            toxicity_values = pd.to_numeric(period_data['toxicity_level'], errors='coerce')
            toxicity_values = toxicity_values[~np.isnan(toxicity_values)]
            if len(toxicity_values) > 0:
                group_stats['toxicity'] = {
                    'mean': float(np.mean(toxicity_values)),
                    'median': float(np.median(toxicity_values)),
                    'std': float(np.std(toxicity_values))
                }
            else:
                group_stats['toxicity'] = {
                    'mean': 0.0,
                    'median': 0.0,
                    'std': 0.0
                }
        except Exception as e:
            print(f"Error processing toxicity: {str(e)}")
            group_stats['toxicity'] = {
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0
            }
        
        return group_stats

    def compare_group_periods(self, pre_period_data: pd.DataFrame, post_period_data: pd.DataFrame, group_name: str) -> pd.DataFrame:
        """Compare metrics between two periods for a group"""
        print(f"\nComparing periods for group: {group_name}")
        
        # Deduplicate data
        pre_period_data = pre_period_data.drop_duplicates(subset=['username'])
        post_period_data = post_period_data.drop_duplicates(subset=['username'])
        
        print(f"\nAnalyzing {len(pre_period_data)} pre-war users and {len(post_period_data)} post-war users")
        
        # Process each period
        pre_period_metrics = self.calculate_group_metrics(pre_period_data)
        post_period_metrics = self.calculate_group_metrics(post_period_data)
        
        # Calculate changes
        changes = {}
        for metric in pre_period_metrics:
            if metric in post_period_metrics:
                pre_val = pre_period_metrics[metric].get('mean', 0)
                post_val = post_period_metrics[metric].get('mean', 0)
                pre_consensus = pre_period_metrics[metric].get('consensus_level', 0)
                post_consensus = post_period_metrics[metric].get('consensus_level', 0)
                
                print(f"\n{metric.upper()}:")
                print(f"  Pre-war: {pre_val:.2f} (consensus: {pre_consensus:.2f})")
                print(f"  Post-war: {post_val:.2f} (consensus: {post_consensus:.2f})")
                print(f"  Change: {post_val - pre_val:+.2f}")
                print(f"  Consensus Change: {post_consensus - pre_consensus:+.2f}")
                
                changes[metric] = {
                    'pre_war_value': pre_val,
                    'post_war_value': post_val,
                    'change': post_val - pre_val,
                    'pre_war_consensus': pre_consensus,
                    'post_war_consensus': post_consensus,
                    'consensus_change': post_consensus - pre_consensus
                }
        
        # Create prompt for LLM analysis
        analysis_prompt = f"""Analyze the changes in group behavior between pre-war and post-war periods for the {group_name} group.

Changes in metrics:
{json.dumps(changes, indent=2)}

Additional context:
- Pre-war users: {len(pre_period_data)}
- Post-war users: {len(post_period_data)}
- Pre-war average toxicity: {pre_period_metrics.get('toxicity', {}).get('mean', 0):.2f}
- Post-war average toxicity: {post_period_metrics.get('toxicity', {}).get('mean', 0):.2f}

Provide a comprehensive analysis in this EXACT JSON format:
{{
    "period_comparison": {{
        "key_changes": {{
            "security_shifts": [
                "description of major security-related changes",
                "description of changes in security vs judicial balance",
                "description of changes in oversight attitudes"
            ],
            "behavioral_changes": [
                "description of changes in toxicity and tone",
                "description of changes in narrative focus",
                "description of changes in group consensus"
            ],
            "consensus_changes": [
                "analysis of changes in group alignment",
                "analysis of areas of increasing/decreasing consensus",
                "implications of consensus changes"
            ]
        }},
        "group_evolution": {{
            "analysis": "comprehensive analysis of how the group's positions evolved",
            "main_trends": [
                "key trend 1",
                "key trend 2",
                "key trend 3"
            ]
        }},
        "impact_analysis": {{
            "analysis": "analysis of the impact of these changes",
            "significant_impacts": [
                "impact 1",
                "impact 2",
                "impact 3"
            ]
        }}
    }}
}}

Focus on:
1. Major shifts in security vs judicial balance
2. Changes in group consensus and alignment
3. Evolution of positions on key issues
4. Impact of these changes on discourse"""

        # Get LLM analysis
        llm_analysis = self._analyze_batch([{"text": analysis_prompt, "created_at": ""}], group_name, 0)
        
        # Save period comparison
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        comparison_file = os.path.join('data', 'group_analysis', f'group_period_comparison_{group_name}_{timestamp}.csv')
        
        comparison_data = {
            'group_name': group_name,
            'pre_war_metrics': pre_period_metrics,
            'post_war_metrics': post_period_metrics,
            'changes': changes,
            'analysis': llm_analysis
        }
        
        pd.DataFrame([comparison_data]).to_csv(comparison_file, index=False)
        print(f"\nSaved period comparison to: {comparison_file}")
        
        return pd.DataFrame([comparison_data]) 

    def _calculate_consensus(self, values: np.ndarray) -> float:
        """
        Calculate consensus level based on value distribution
        Returns score between 0 (no consensus) and 1 (full consensus)
        Handles small sample sizes more appropriately
        """
        if len(values) < 2:
            return 0.5  # Return moderate consensus for single values
        
        # Use coefficient of variation as consensus measure
        mean = np.mean(values)
        std = np.std(values)
        
        # For small samples (n < 5), adjust consensus calculation
        if len(values) < 5:
            # Use range relative to mean as additional factor
            value_range = np.max(values) - np.min(values)
            range_factor = value_range / (abs(mean) if mean != 0 else 1)
            
            # Combine CV and range factor
            cv = std / (abs(mean) if mean != 0 else 1)
            combined_factor = (cv + range_factor) / 2
            
            # Scale consensus more conservatively for small samples
            consensus = 1 / (1 + combined_factor)
            # Apply small sample penalty
            consensus *= (len(values) / 5)
        else:
            # Use standard CV for larger samples
            cv = std / (abs(mean) if mean != 0 else 1)
            consensus = 1 / (1 + cv)
        
        return float(consensus) 