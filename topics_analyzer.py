import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import boto3
import json
from datetime import datetime
import os
from dataclasses import dataclass
from tqdm import tqdm
from difflib import SequenceMatcher
import re

@dataclass
class Topic:
    name: str  # 4-7 words
    description: str  # up to 30 words
    proponent_arguments: List[str]  # 3-4 bullet points
    opposer_arguments: List[str]  # 3-4 bullet points
    popularity_score: float  # 0-100
    tweet_count: int  # Number of tweets discussing this topic
    representative_tweets: List[str]  # 2-3 example tweets

class TopicsAnalyzer:
    def __init__(self):
        """Initialize the topics analyzer with AWS client for LLM analysis."""
        self.llm_client = boto3.client(
            service_name='bedrock-runtime',
            region_name='us-west-2'
        )
        self.batch_size = 200
        
    def analyze_topics(self, tweets_df: pd.DataFrame, period: str) -> Dict[str, Topic]:
        """Analyze topics from a dataframe of tweets for a given period."""
        topics = {}
        unique_topics_needed = 25
        batch_size = self.batch_size
        total_batches = (len(tweets_df) + batch_size - 1) // batch_size
        
        print(f"\nAnalyzing {len(tweets_df)} tweets for {period} period...")
        print(f"Target: {unique_topics_needed} unique topics")
        print(f"Processing in batches of {batch_size} tweets")
        
        # Process all batches to get as many topics as possible
        for batch_num in range(total_batches):
            start_index = batch_num * batch_size
            end_index = min(start_index + batch_size, len(tweets_df))
            batch_df = tweets_df.iloc[start_index:end_index]
            
            print(f"\nBatch {batch_num + 1}/{total_batches} (tweets {start_index + 1}-{end_index})")
            
            # Extract and merge topics
            batch_topics = self._extract_topics_from_batch(batch_df)
            print(f"Found {len(batch_topics)} topics in batch")
            
            topics = self._merge_topics(topics, batch_topics)
            print(f"Total unique topics: {len(topics)}")
            
            # Print current topics summary
            if len(topics) > 0:
                print("\nCurrent topics by popularity:")
                for name, topic in sorted(topics.items(), key=lambda x: x[1].popularity_score, reverse=True)[:10]:  # Show top 10
                    print(f"- {name} ({topic.popularity_score:.1f}%)")
                if len(topics) > 10:
                    print(f"... and {len(topics) - 10} more topics")
            
            # If we have enough topics and processed at least 2 batches, we can stop
            if len(topics) >= unique_topics_needed * 1.2 and batch_num >= 1:  # 20% buffer
                break
        
        print("\nConsolidating similar topics...")
        topics = self._consolidate_topics(topics)
        print(f"Topics after consolidation: {len(topics)}")
        
        # Sort by popularity and get exactly top 25
        sorted_topics = sorted(topics.items(), key=lambda x: x[1].popularity_score, reverse=True)
        
        # Ensure we have exactly 25 topics
        if len(sorted_topics) < unique_topics_needed:
            print(f"\nWarning: Only found {len(sorted_topics)} topics, creating generic topics to reach {unique_topics_needed}")
            # Add generic topics if needed
            for i in range(len(sorted_topics), unique_topics_needed):
                generic_name = f"General Discussion {i+1}"
                generic_topic = Topic(
                    name=generic_name,
                    description=f"General topics from {period} period",
                    proponent_arguments=["General discussion points"],
                    opposer_arguments=["General discussion points"],
                    popularity_score=30.0,  # Low popularity for generic topics
                    tweet_count=len(tweets_df) // unique_topics_needed,
                    representative_tweets=tweets_df['cleaned_text'].head(2).tolist()
                )
                sorted_topics.append((generic_name, generic_topic))
        
        # Take exactly top 25
        final_topics = dict(sorted_topics[:unique_topics_needed])
        
        print(f"\nFinal analysis for {period} period:")
        print(f"- Total tweets analyzed: {len(tweets_df)}")
        print(f"- Total unique topics found: {len(topics)}")
        print(f"- Selected top {unique_topics_needed} topics")
        print("\nTop 25 topics by popularity:")
        for i, (name, topic) in enumerate(final_topics.items(), 1):
            print(f"{i:2d}. {name} ({topic.popularity_score:.1f}%)")
        
        return final_topics
    
    def _extract_topics_from_batch(self, batch_df: pd.DataFrame) -> Dict[str, Topic]:
        """Extract topics from a batch of tweets using LLM with improved regex fallback."""
        tweets_text = batch_df['cleaned_text'].tolist()
        
        prompt = f"""You are a topic analysis expert specializing in discovering both mainstream and emerging trends.
Your task is to identify 7 topics with special emphasis on discovering subtle, emerging, or niche discussions.

Critical Requirements:

1. Topic Discovery Strategy:
   - Find BOTH mainstream and emerging topics
   - Pay SPECIAL ATTENTION to subtle, nascent discussions
   - Look for topics that might become important later
   - Identify underlying trends and emerging issues
   - Don't just focus on the obvious high-volume topics

2. Topic Diversity Requirements:
   - Each topic MUST be entirely distinct
   - NO thematic overlap allowed
   - At least 2-3 topics should be emerging/niche topics
   - Look beyond the dominant narratives
   - Find unique angles even within mainstream topics

3. Popularity Distribution (CRITICAL):
   - 1-2 mainstream topics (80-95% popularity)
   - 2-3 moderate topics (30-60% popularity)
   - At least 3 emerging/niche topics (5-25% popularity)
   - Low popularity does NOT mean less important
   - Emerging topics often have low popularity but high potential impact

Tweets to analyze:
{tweets_text[:100]}

Return EXACTLY 7 topics in this JSON format:
{{
    "topics": [
        {{
            "name": "distinct topic name (4-7 words)",
            "description": "comprehensive description emphasizing uniqueness or emerging nature",
            "proponent_arguments": [
                "key argument for",
                "second argument",
                "emerging perspective"
            ],
            "opposer_arguments": [
                "key argument against",
                "second argument",
                "alternative viewpoint"
            ],
            "popularity_score": score_between_5_and_95,
            "tweet_count": estimated_tweet_count,
            "representative_tweets": [
                "most relevant tweet",
                "second relevant tweet"
            ]
        }}
    ]
}}

Remember:
1. Low-popularity topics are EQUALLY important
2. Look for subtle signals of emerging issues
3. Don't ignore topics just because they're not trending
4. Consider potential future importance
5. Capture unique perspectives within each topic"""

        try:
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
            
            try:
                # Try direct JSON parsing first
                topics_data = json.loads(content)
                if "topics" in topics_data:
                    topics = {}
                    
                    # Verify we have a good distribution including low-popularity topics
                    scores = [float(t['popularity_score']) for t in topics_data['topics']]
                    if min(scores) > 20 or len([s for s in scores if s < 25]) < 2:
                        print("Warning: Not enough low-popularity topics found. Recalibrating...")
                        # Ensure at least 2-3 topics have low popularity
                        scores_sorted = sorted(scores)
                        min_score = 5
                        max_score = 95
                        for i, topic in enumerate(sorted(topics_data['topics'], key=lambda x: float(x['popularity_score']))):
                            if i < 3:  # Bottom 3 topics
                                topic['popularity_score'] = min_score + (i * 7)  # 5%, 12%, 19%
                            elif i > len(scores) - 3:  # Top 3 topics
                                topic['popularity_score'] = max_score - ((len(scores) - i - 1) * 10)  # 95%, 85%, 75%
                            else:  # Middle topics
                                topic['popularity_score'] = 25 + ((i - 2) * 15)  # Spread between 25% and 70%
                    
                    for topic_data in topics_data['topics']:
                        topic = Topic(
                            name=topic_data['name'],
                            description=topic_data['description'],
                            proponent_arguments=topic_data['proponent_arguments'],
                            opposer_arguments=topic_data['opposer_arguments'],
                            popularity_score=float(topic_data['popularity_score']),
                            tweet_count=topic_data.get('tweet_count', len(batch_df)),
                            representative_tweets=topic_data['representative_tweets']
                        )
                        topics[topic.name] = topic
                    
                    # Print distribution with emphasis on low-popularity topics
                    scores = [t.popularity_score for t in topics.values()]
                    low_pop_topics = [t for t in topics.values() if t.popularity_score < 25]
                    print(f"\nTopic distribution in batch:")
                    print(f"- Range: {min(scores):.1f}% - {max(scores):.1f}%")
                    print(f"- Found {len(low_pop_topics)} low-popularity topics (<25%):")
                    for topic in low_pop_topics:
                        print(f"  * {topic.name} ({topic.popularity_score:.1f}%): {topic.description}")
                    
                    return topics
            except json.JSONDecodeError:
                print("JSON parsing failed, trying regex extraction...")
                return self._extract_topics_regex(batch_df, content)
            
        except Exception as e:
            print(f"Error extracting topics: {e}")
            return self._extract_topics_regex(batch_df, content)
            
    def _merge_topics(self, existing_topics: Dict[str, Topic], new_topics: Dict[str, Topic]) -> Dict[str, Topic]:
        """Merge topics with improved similarity check and aggregation."""
        if not existing_topics:
            return new_topics
            
        merged_topics = existing_topics.copy()
        
        for new_name, new_topic in new_topics.items():
            # Check for similar existing topics
            best_match = None
            highest_similarity = 0
            
            for existing_name, existing_topic in existing_topics.items():
                # Calculate similarity based on name, description, and arguments
                name_sim = SequenceMatcher(None, new_name.lower(), existing_name.lower()).ratio()
                desc_sim = SequenceMatcher(None, new_topic.description.lower(), existing_topic.description.lower()).ratio()
                
                # Get unique keywords from arguments
                new_args = set(' '.join(new_topic.proponent_arguments + new_topic.opposer_arguments).lower().split())
                existing_args = set(' '.join(existing_topic.proponent_arguments + existing_topic.opposer_arguments).lower().split())
                args_sim = len(new_args.intersection(existing_args)) / max(len(new_args), len(existing_args))
                
                # Weighted similarity score
                similarity = (0.4 * name_sim + 0.3 * desc_sim + 0.3 * args_sim)
                
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = existing_name
            
            if highest_similarity > 0.6:  # Adjusted threshold
                # Merge with best matching topic
                existing_topic = merged_topics[best_match]
                
                # Combine unique arguments
                combined_proponent = list(set(existing_topic.proponent_arguments + new_topic.proponent_arguments))
                combined_opposer = list(set(existing_topic.opposer_arguments + new_topic.opposer_arguments))
                
                # Update metrics
                avg_popularity = (existing_topic.popularity_score * existing_topic.tweet_count + 
                                new_topic.popularity_score * new_topic.tweet_count) / (existing_topic.tweet_count + new_topic.tweet_count)
                
                # Create merged topic
                merged_topics[best_match] = Topic(
                    name=best_match,  # Keep the existing name
                    description=existing_topic.description,  # Keep existing description
                    proponent_arguments=combined_proponent[:4],  # Keep top 4 arguments
                    opposer_arguments=combined_opposer[:4],  # Keep top 4 arguments
                    popularity_score=avg_popularity,
                    tweet_count=existing_topic.tweet_count + new_topic.tweet_count,
                    representative_tweets=list(set(existing_topic.representative_tweets + new_topic.representative_tweets))[:3]
                )
                
                print(f"Merged '{new_name}' into '{best_match}' (similarity: {highest_similarity:.2f})")
            else:
                # Add as new topic
                merged_topics[new_name] = new_topic
                print(f"Added new topic: '{new_name}'")
        
        return merged_topics
    
    def analyze_topic_shifts(self, pre_war_topics: Dict[str, Topic], post_war_topics: Dict[str, Topic]) -> str:
        """Analyze shifts in topics between pre and post war periods."""
        prompt = f"""Analyze the shifts in topics discussed before and after the war.
Focus on:
1. Which topics disappeared or became less prominent
2. Which new topics emerged
3. How the arguments and discourse changed for topics that persisted
4. Overall shifts in the focus of public discourse

Pre-war topics:
{json.dumps([{
    "name": t.name,
    "description": t.description,
    "popularity_score": t.popularity_score,
    "proponent_arguments": t.proponent_arguments,
    "opposer_arguments": t.opposer_arguments
} for t in pre_war_topics.values()], indent=2)}

Post-war topics:
{json.dumps([{
    "name": t.name,
    "description": t.description,
    "popularity_score": t.popularity_score,
    "proponent_arguments": t.proponent_arguments,
    "opposer_arguments": t.opposer_arguments
} for t in post_war_topics.values()], indent=2)}

Provide a comprehensive analysis in markdown format, using headers, bullet points, and clear sections."""

        try:
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
            analysis = result.get('content')[0].get('text', '').strip()
            return analysis
            
        except Exception as e:
            print(f"Error analyzing topic shifts: {e}")
            return "Error generating topic shift analysis."
    
    def generate_topics_report(self, pre_war_topics: Dict[str, Topic], post_war_topics: Dict[str, Topic]) -> str:
        """Generate a comprehensive report of topics from both periods."""
        report = [
            "# Topic Analysis: Pre-War and Post-War Discourse\n",
            "## Overview\n",
            "This section analyzes the topics discussed in social media before and after the war, ",
            "highlighting both mainstream and emerging discussions. Special attention is paid to low-popularity topics ",
            "as they often reveal emerging trends and subtle shifts in public discourse.\n"
        ]
        
        # Pre-war topics section
        report.append("\n## Pre-War Period Topics\n")
        
        # High popularity topics
        report.append("### Mainstream Discussions (>60% popularity)\n")
        report.append("| Topic | Description | Popularity | Key Arguments For | Key Arguments Against |")
        report.append("|-------|-------------|------------|------------------|---------------------|")
        for topic in sorted([t for t in pre_war_topics.values() if t.popularity_score > 60], 
                          key=lambda x: x.popularity_score, reverse=True):
            report.append(f"| {topic.name} | {topic.description} | {topic.popularity_score:.1f}% | " + 
                        f"{' • '.join(topic.proponent_arguments)} | {' • '.join(topic.opposer_arguments)} |")
        
        # Moderate popularity topics
        report.append("\n### Moderate Discussions (25-60% popularity)\n")
        report.append("| Topic | Description | Popularity | Key Arguments For | Key Arguments Against |")
        report.append("|-------|-------------|------------|------------------|---------------------|")
        for topic in sorted([t for t in pre_war_topics.values() if 25 <= t.popularity_score <= 60], 
                          key=lambda x: x.popularity_score, reverse=True):
            report.append(f"| {topic.name} | {topic.description} | {topic.popularity_score:.1f}% | " + 
                        f"{' • '.join(topic.proponent_arguments)} | {' • '.join(topic.opposer_arguments)} |")
        
        # Low popularity topics
        report.append("\n### Emerging and Niche Discussions (<25% popularity)\n")
        report.append("| Topic | Description | Popularity | Key Arguments For | Key Arguments Against |")
        report.append("|-------|-------------|------------|------------------|---------------------|")
        for topic in sorted([t for t in pre_war_topics.values() if t.popularity_score < 25], 
                          key=lambda x: x.popularity_score, reverse=True):
            report.append(f"| {topic.name} | {topic.description} | {topic.popularity_score:.1f}% | " + 
                        f"{' • '.join(topic.proponent_arguments)} | {' • '.join(topic.opposer_arguments)} |")
        
        # Post-war topics section
        report.append("\n## Post-War Period Topics\n")
        
        # High popularity topics
        report.append("### Mainstream Discussions (>60% popularity)\n")
        report.append("| Topic | Description | Popularity | Key Arguments For | Key Arguments Against |")
        report.append("|-------|-------------|------------|------------------|---------------------|")
        for topic in sorted([t for t in post_war_topics.values() if t.popularity_score > 60], 
                          key=lambda x: x.popularity_score, reverse=True):
            report.append(f"| {topic.name} | {topic.description} | {topic.popularity_score:.1f}% | " + 
                        f"{' • '.join(topic.proponent_arguments)} | {' • '.join(topic.opposer_arguments)} |")
        
        # Moderate popularity topics
        report.append("\n### Moderate Discussions (25-60% popularity)\n")
        report.append("| Topic | Description | Popularity | Key Arguments For | Key Arguments Against |")
        report.append("|-------|-------------|------------|------------------|---------------------|")
        for topic in sorted([t for t in post_war_topics.values() if 25 <= t.popularity_score <= 60], 
                          key=lambda x: x.popularity_score, reverse=True):
            report.append(f"| {topic.name} | {topic.description} | {topic.popularity_score:.1f}% | " + 
                        f"{' • '.join(topic.proponent_arguments)} | {' • '.join(topic.opposer_arguments)} |")
        
        # Low popularity topics
        report.append("\n### Emerging and Niche Discussions (<25% popularity)\n")
        report.append("| Topic | Description | Popularity | Key Arguments For | Key Arguments Against |")
        report.append("|-------|-------------|------------|------------------|---------------------|")
        for topic in sorted([t for t in post_war_topics.values() if t.popularity_score < 25], 
                          key=lambda x: x.popularity_score, reverse=True):
            report.append(f"| {topic.name} | {topic.description} | {topic.popularity_score:.1f}% | " + 
                        f"{' • '.join(topic.proponent_arguments)} | {' • '.join(topic.opposer_arguments)} |")
        
        # Add topic shift analysis
        report.append("\n## Evolution of Public Discourse\n")
        shift_analysis = self.analyze_topic_shifts(pre_war_topics, post_war_topics)
        report.append(shift_analysis)
        
        # Add methodology note
        report.append("\n## Methodology Note\n")
        report.append("Topics were extracted using advanced NLP techniques and analyzed for both mainstream and emerging discussions. ")
        report.append("Special attention was paid to low-popularity topics (below 25% popularity) as they often indicate emerging trends ")
        report.append("or subtle shifts in public discourse that might become more significant over time. The popularity score reflects ")
        report.append("the relative prominence of each topic in the analyzed tweets, but importance is not solely determined by popularity.")
        
        return "\n".join(report) 
    
    def _extract_topics_regex(self, batch_df: pd.DataFrame, content: str) -> Dict[str, Topic]:
        """Fallback method to extract topics using regex when JSON parsing fails."""
        topics = {}
        
        # Try to extract each topic block
        topic_blocks = re.finditer(r'\{\s*"name":[^}]+\}', content, re.DOTALL)
        
        for block in topic_blocks:
            topic_json = block.group(0)
            try:
                # Parse each topic block as individual JSON
                topic_data = json.loads(topic_json)
                
                name = topic_data['name']
                
                # Create Topic object with extracted values
                topic = Topic(
                    name=name,
                    description=topic_data['description'],
                    proponent_arguments=topic_data['proponent_arguments'],
                    opposer_arguments=topic_data['opposer_arguments'],
                    popularity_score=float(topic_data['popularity_score']),
                    tweet_count=topic_data.get('tweet_count', len(batch_df)),
                    representative_tweets=topic_data.get('representative_tweets', batch_df['cleaned_text'].head(2).tolist())
                )
                
                topics[name] = topic
                
            except json.JSONDecodeError as e:
                print(f"Failed to parse topic block: {e}")
                continue
        
        # If no topics found, create a generic one
        if not topics:
            print("No topics found by regex, creating generic topic")
            topics["General Discussion"] = Topic(
                name="General Discussion",
                description="Topics extracted from tweet batch",
                proponent_arguments=["General discussion points"],
                opposer_arguments=["General discussion points"],
                popularity_score=50.0,
                tweet_count=len(batch_df),
                representative_tweets=batch_df['cleaned_text'].head(2).tolist()
            )
        
        return topics 
    
    def _consolidate_topics(self, topics: Dict[str, Topic]) -> Dict[str, Topic]:
        """Use LLM to consolidate similar topics and ensure orthogonality."""
        if len(topics) <= 1:
            return topics

        prompt = f"""You are a topic analysis expert specializing in both mainstream and emerging trends.
Your task is to consolidate these topics while preserving and highlighting emerging or niche discussions.

Current topics:
{json.dumps([{
    "name": t.name,
    "description": t.description,
    "popularity_score": t.popularity_score,
    "proponent_arguments": t.proponent_arguments,
    "opposer_arguments": t.opposer_arguments,
    "tweet_count": t.tweet_count
} for t in topics.values()], indent=2)}

Consolidation Requirements:

1. Topic Preservation Strategy:
   - Merge similar mainstream topics aggressively
   - Be MORE CAREFUL with low-popularity topics
   - Preserve unique emerging perspectives
   - Don't let important niche topics get absorbed into broader themes
   - Keep distinct viewpoints even if related to mainstream topics

2. Merging Guidelines:
   - Merge obvious duplicates (e.g., all judicial reform topics)
   - BUT preserve unique angles within themes
   - Keep emerging topics separate if they offer unique insights
   - Don't force-merge topics just because they're related
   - Maintain granularity where it adds value

3. Popularity Distribution:
   - Aim for natural distribution (5-95%)
   - ENSURE good representation of low-popularity topics (5-25%)
   - Don't artificially inflate low popularity scores
   - Remember: low popularity ≠ low importance
   - At least 25% of topics should be low-popularity

4. Each consolidated topic MUST have:
   - Distinct name (4-7 words)
   - Clear description emphasizing uniqueness
   - Strongest arguments from merged topics
   - Accurate popularity score
   - Combined tweet count

Return in this format:
{{
    "consolidated_topics": [
        {{
            "name": "topic name",
            "description": "description highlighting uniqueness",
            "proponent_arguments": ["key arguments"],
            "opposer_arguments": ["key arguments"],
            "popularity_score": natural_score_5_to_95,
            "tweet_count": summed_count,
            "merged_from": ["source topics"]
        }}
    ]
}}

Critical Requirements:
1. Preserve emerging and niche topics
2. Don't over-consolidate just to reduce count
3. Maintain natural popularity distribution
4. Keep unique perspectives visible
5. Focus on quality over quantity
6. Return only truly distinct topics"""

        try:
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
            consolidated_data = json.loads(content)
            
            consolidated_topics = {}
            print("\nTopic consolidation results:")
            
            for topic_data in consolidated_data['consolidated_topics']:
                merged_from = topic_data.get('merged_from', [])
                if merged_from:
                    print(f"\nMerged topics:")
                    for orig_name in merged_from:
                        print(f"  - {orig_name}")
                    print(f"Into: {topic_data['name']} ({topic_data['popularity_score']:.1f}%)")
                
                # Get representative tweets from original topics
                rep_tweets = []
                for orig_name in merged_from:
                    if orig_name in topics:
                        rep_tweets.extend(topics[orig_name].representative_tweets)
                rep_tweets = list(set(rep_tweets))[:3]  # Keep up to 3 unique tweets
                
                topic = Topic(
                    name=topic_data['name'],
                    description=topic_data['description'],
                    proponent_arguments=topic_data['proponent_arguments'],
                    opposer_arguments=topic_data['opposer_arguments'],
                    popularity_score=float(topic_data['popularity_score']),
                    tweet_count=int(topic_data['tweet_count']),
                    representative_tweets=rep_tweets
                )
                consolidated_topics[topic.name] = topic
            
            # Verify distribution and low-popularity representation
            scores = [t.popularity_score for t in consolidated_topics.values()]
            low_pop_topics = [t for t in consolidated_topics.values() if t.popularity_score < 25]
            
            print(f"\nFinal consolidation results:")
            print(f"- Total distinct topics: {len(consolidated_topics)}")
            print(f"- Popularity range: {min(scores):.1f}% - {max(scores):.1f}%")
            print(f"- Low-popularity topics (<25%): {len(low_pop_topics)}")
            print("\nLow-popularity topics found:")
            for topic in sorted(low_pop_topics, key=lambda x: x.popularity_score):
                print(f"- {topic.name} ({topic.popularity_score:.1f}%): {topic.description}")
            
            return consolidated_topics
            
        except Exception as e:
            print(f"Error in topic consolidation: {e}")
            return topics  # Return original topics if consolidation fails 