import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import redis
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union
import logging
import pickle
from functools import lru_cache

class TravelRecommender:
    def __init__(self, redis_host='localhost', redis_port=6379):
        # Initialize Redis for caching
        self.redis_client = redis.Redis(host=redis_host, port=redis_port)
        self.cache_ttl = 3600  # Cache TTL in seconds (1 hour)
        self.long_ttl = 86400  # Long-term cache TTL (24 hours)
        
        # Initialize storage for our models
        self.post_vectors = None
        self.user_item_matrix = None
        self.post_features = None
        self.similarity_matrix = None
        self.user_connections = None
        self.user_index_mapping = {}
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _get_cache_key(self, prefix: str, identifier: Union[int, str]) -> str:
        """Generate a standardized cache key"""
        return f"{prefix}:{identifier}"

    def _cache_get(self, key: str) -> Optional[bytes]:
        """Get data from cache with error handling"""
        try:
            return self.redis_client.get(key)
        except redis.RedisError as e:
            self.logger.error(f"Redis get error for key {key}: {str(e)}")
            return None

    def _cache_set(self, key: str, value: bytes, ttl: int = None) -> bool:
        """Set data in cache with error handling"""
        try:
            if ttl is None:
                ttl = self.cache_ttl
            return self.redis_client.setex(key, ttl, value)
        except redis.RedisError as e:
            self.logger.error(f"Redis set error for key {key}: {str(e)}")
            return False

    def _serialize_numpy(self, array: np.ndarray) -> bytes:
        """
        Serialize numpy array for Redis storage with type conversion
        """
        try:
            if isinstance(array, np.ndarray):
                array = array.tolist()  # Convert numpy array to list
            return pickle.dumps(array)
        except Exception as e:
            self.logger.error(f"Error serializing numpy array: {str(e)}")
            raise

    def _deserialize_numpy(self, data: bytes) -> np.ndarray:
        """
        Deserialize numpy array from Redis storage with type conversion
        """
        try:
            array = pickle.loads(data)
            if isinstance(array, list):
                array = np.array(array)  # Convert back to numpy array
            return array
        except Exception as e:
            self.logger.error(f"Error deserializing numpy array: {str(e)}")
            raise

    def set_user_connections(self, connections_df: pd.DataFrame):
        """Set up user connections with caching"""
        try:
            def parse_connections(conn_str):
                if isinstance(conn_str, str):
                    cleaned = conn_str.strip('[]').replace(' ', '')
                    return [int(x) for x in cleaned.split(',') if x]
                elif isinstance(conn_str, list):
                    return conn_str
                return []

            self.user_connections = {}
            for _, row in connections_df.iterrows():
                user_id = row['user_id']
                connections = parse_connections(row['connections'])
                self.user_connections[user_id] = connections
                
                # Cache individual user connections
                cache_key = self._get_cache_key('user_connections', user_id)
                self._cache_set(cache_key, json.dumps(connections).encode(), self.long_ttl)
            
            self.logger.info(f"Successfully processed and cached connections for {len(self.user_connections)} users")
            
        except Exception as e:
            self.logger.error(f"Error setting user connections: {str(e)}")
            self.user_connections = {}
    
    def _serialize_dict_with_numpy(self, d: dict) -> dict:
        """
        Convert dictionary with numpy values to JSON-serializable format
        """
        converted = {}
        for k, v in d.items():
            # Convert keys
            if isinstance(k, np.integer):
                k = int(k)
            elif isinstance(k, np.floating):
                k = float(k)
                
            # Convert values
            if isinstance(v, np.integer):
                v = int(v)
            elif isinstance(v, np.floating):
                v = float(v)
            elif isinstance(v, np.ndarray):
                v = v.tolist()
                
            converted[k] = v
        return converted

    def get_user_connections(self, user_id: int) -> List[int]:
        """Get user connections with caching"""
        cache_key = self._get_cache_key('user_connections', user_id)
        cached_data = self._cache_get(cache_key)
        
        if cached_data:
            try:
                return json.loads(cached_data)
            except json.JSONDecodeError:
                self.logger.error(f"Error decoding cached connections for user {user_id}")
        
        connections = self.user_connections.get(user_id, [])
        self._cache_set(cache_key, json.dumps(connections).encode(), self.long_ttl)
        return connections

    def build_user_item_matrix(self, interactions_df: pd.DataFrame) -> np.ndarray:
        """
        Build user-item interaction matrix with proper index mapping and type conversion
        """
        try:
            # Create user index mapping with explicit type conversion to int
            unique_users = sorted(interactions_df['user_id'].unique())
            self.user_index_mapping = {int(user_id): idx for idx, user_id in enumerate(unique_users)}
            
            # Define interaction weights
            weights = {
                'view': 1,
                'like': 3,
                'comment': 4,
                'share': 5,
                'save': 3
            }
            
            # Map user_ids to indices
            interactions_df['user_idx'] = interactions_df['user_id'].map(self.user_index_mapping)
            
            # Create pivot table with weighted interactions
            matrix = pd.pivot_table(
                interactions_df,
                values='activity_type',
                index='user_idx',
                columns='content_id',
                aggfunc=lambda x: sum(weights.get(i, 1) for i in x),
                fill_value=0
            )
            
            return matrix.values
            
        except Exception as e:
            self.logger.error(f"Error building user-item matrix: {str(e)}")
            return np.array([])

    def get_user_matrix_index(self, user_id: int) -> int:
        """
        Get the correct matrix index for a user_id with type conversion
        """
        try:
            # Convert numpy.int64 to Python int if necessary
            user_id = int(user_id)

            # Check if the user exists in the  user item matrix or not

            # user_idx=self.user_index_mapping.get(user_id)
            # print("User ID index is from the user_item_matrix is ",user_idx)
            # print(f"Type of user idx {type(user_idx)} ")
            return self.user_index_mapping[user_id]
        except KeyError:
            raise ValueError(f"User ID {user_id} not found in interaction history")

    def update_models(self, posts_df: pd.DataFrame, interactions_df: pd.DataFrame):
        """
        Update models with proper type conversion for caching
        """
        try:
            self.logger.info("Starting model update...")            
            # Update content-based features
            self.post_vectors = self.process_post_content(posts_df)
            self.post_features = posts_df
            
            # Cache post vectors
            self._cache_set('post_vectors', self._serialize_numpy(self.post_vectors), self.long_ttl)
            
            # Update collaborative filtering matrix
            self.user_item_matrix = self.build_user_item_matrix(interactions_df)
            self._cache_set('user_item_matrix', self._serialize_numpy(self.user_item_matrix), self.long_ttl)
            
            # Calculate and cache similarity matrix
            self.similarity_matrix = cosine_similarity(self.post_vectors)
            self._cache_set('similarity_matrix', self._serialize_numpy(self.similarity_matrix), self.long_ttl)
            
            # Cache post features
            self._cache_set('post_features', posts_df.to_json().encode(), self.long_ttl)
            
            # Cache user index mapping with type conversion
            converted_mapping = self._serialize_dict_with_numpy(self.user_index_mapping)
            self._cache_set('user_index_mapping', json.dumps(converted_mapping).encode(), self.long_ttl)
            
            # Update timestamp
            self._cache_set('last_update_timestamp', str(datetime.now()).encode(), self.long_ttl)
            
            self.logger.info("Model update completed and cached successfully")
            
        except Exception as e:
            self.logger.error(f"Error updating models: {str(e)}")
            raise

    def calculate_recency_score(self, post_date: str, max_age_days: int = 60) -> float:
        """
        Calculate recency score for a post with proper timezone handling
        """
        try:
            # Convert string to datetime if needed
            if isinstance(post_date, str):
                post_date = pd.to_datetime(post_date)
                
            # Ensure we're working with timezone-naive datetimes
            if post_date.tzinfo is not None:
                post_date = post_date.tz_localize(None)
                
            now = pd.Timestamp.now()
            
            # Calculate age in days
            age_days = (now - post_date).days
            
            # Debug logging
            self.logger.debug(f"Post date: {post_date}, Now: {now}, Age days: {age_days}")
            
            # Handle edge cases
            if age_days < 0:
                self.logger.warning(f"Negative age detected for post date {post_date}")
                return 0.0
                
            if age_days > max_age_days:
                return 0.0
                
            # Calculate recency score
            recency_score = 1 - (age_days / max_age_days)
            
            self.logger.debug(f"Calculated recency score: {recency_score}")
            
            return max(0.0, min(1.0, recency_score))  # Ensure score is between 0 and 1
            
        except Exception as e:
            self.logger.error(f"Error calculating recency score: {str(e)}")
            return 0.0
        
    def process_post_content(self, posts_df: pd.DataFrame) -> np.ndarray:

        """
        Process post content using TF-IDF vectorization
        """
        try:
            # Print columns for debugging
            self.logger.info(f"Available columns: {posts_df.columns}")
            
            # Combine relevant text fields with proper null handling
            posts_df['combined_text'] = (
                posts_df['caption'].fillna('') + ' ' +
                posts_df['processed_categories'].fillna('') + ' ' +
                posts_df['processed_subcategories'].fillna('')
            ).str.strip()
            
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            # Convert any non-string values to strings
            posts_df['combined_text'] = posts_df['combined_text'].astype(str)
            
            vector = vectorizer.fit_transform(posts_df['combined_text'])
            return vector
        
        except Exception as e:
            self.logger.error(f"Error in process_post_content: {str(e)}")
            raise

    def get_similar_posts(self, post_id: int, n_recommendations: int = 5) -> List[Dict]:
        """
        Get similar posts based on content-based filtering for a given post_id.
        Recommendations are filtered to match the asset_type (image or video) of the input post.
        
        Parameters:
        - post_id (int): The ID of the post to find similar posts for.
        - n_recommendations (int): Number of similar posts to recommend.
        
        Returns:
        - List[Dict]: A list of dictionaries containing similar posts with their scores.
        """
        try:
            # Find the index of the given post_id in the post_features DataFrame
            post_idx = self.post_features[self.post_features['post_id'] == post_id].index[0]
            
            # Get the asset_type of the input post
            input_asset_type = self.post_features.iloc[post_idx]['asset_type']
            print(input_asset_type)
            
            # Get the similarity scores for the given post from the similarity matrix
            sim_scores = self.similarity_matrix[post_idx]
            
            # Filter posts to match the asset_type of the input post
            valid_indices = [
                idx for idx, score in enumerate(sim_scores) 
                if self.post_features.iloc[idx]['asset_type'] == input_asset_type
            ]
            
            # Sort the similarity scores in descending order and get the top n_recommendations
            similar_indices = sorted(valid_indices, key=lambda idx: sim_scores[idx], reverse=True)[1:n_recommendations + 1]  # Exclude the post itself
            
            # Prepare the recommendations
            recommendations = []
            for idx in similar_indices:
                recommendations.append({
                    'post_id': int(self.post_features.iloc[idx]['post_id']),
                    'score': float(sim_scores[idx]),
                    'type': 'content',
                    'asset_type': self.post_features.iloc[idx]['asset_type']  # Include asset_type in the response
                })
            
            return recommendations
        
        except IndexError:
            self.logger.error(f"Post ID {post_id} not found in post_features.")
            return []
        except Exception as e:
            self.logger.error(f"Error in get_similar_posts: {str(e)}")
            return []

    def get_content_based_recommendations(self, user_id: int, n_recommendations: int = 5) -> List[Dict]:
        """Get content-based recommendations with caching"""
        cache_key = self._get_cache_key('content_recs', f"{user_id}:{n_recommendations}")
        cached_data = self._cache_get(cache_key)
        
        if cached_data:
            try:
                return json.loads(cached_data)
            except json.JSONDecodeError:
                self.logger.error("Error decoding cached content recommendations")
        
        try:
            user_idx = self.get_user_matrix_index(user_id)
            user_interactions = self.user_item_matrix[user_idx]
            interacted_posts = np.where(user_interactions > 0)[0]
            
            if len(interacted_posts) == 0:
                recommendations = self.get_popular_recommendations(n_recommendations)
            else:
                sim_scores = np.mean([self.similarity_matrix[i] for i in interacted_posts], axis=0)
                similar_posts = np.argsort(sim_scores)[::-1]
                valid_indices = [i for i in similar_posts if i < len(self.post_features)]
                
                recommendations = [
                    {
                        'post_id': int(self.post_features.iloc[i]['post_id']),
                        'score': float(sim_scores[i]),
                        'type': 'content'
                    }
                    for i in valid_indices
                    if i not in interacted_posts
                ][:n_recommendations]
            
            # Cache the recommendations
            self._cache_set(cache_key, json.dumps(recommendations).encode())
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error in content-based recommendations: {str(e)}")
            return self.get_popular_recommendations(n_recommendations)

    @lru_cache(maxsize=128)
    def get_collaborative_recommendations(self, user_id: int, n_recommendations: int = 5, detailed_response: bool = True) -> list:
        """
        Get collaborative recommendations with caching.

        Ensures a 4:1 ratio of images to videos if possible but does not force it. 
        If either category is insufficient, it continues with whatever is available.
        Falls back to popular recommendations only if total posts are insufficient.

        Parameters:
        - user_id (int): The ID of the user to get recommendations for.
        - n_recommendations (int): Number of recommendations to return.
        - detailed_response (bool): If True, return detailed dictionaries; else, return post IDs.

        Returns:
        - List of recommended posts (either detailed or IDs).
        """
        try:
            # if user_id not in self.user_item_matrix:
            #     print(f"User {user_id} not found in user-item matrix.")

            user_idx = self.get_user_matrix_index(user_id)
            # print(f"User index: {user_idx}")
            user_vector = self.user_item_matrix[user_idx]  # This line might be causing the issue
            # print(f"User vector: {user_vector}")
            user_similarity = cosine_similarity([self.user_item_matrix[user_idx]], self.user_item_matrix)[0]
            # print(f"User similarity scores for {user_id}: {user_similarity}")

            similar_users = [u for u in np.argsort(user_similarity)[::-1] if u != user_idx][:5]
            # print(f"Similar users for {user_id}: {similar_users}")
            # for sim_user_idx in similar_users:
                # print(f"User {sim_user_idx} interacted posts: {np.where(self.user_item_matrix[sim_user_idx] > 0)}")


            if not similar_users:
                return self.get_popular_recommendations(n_recommendations, detailed_response)

            similar_user_posts = defaultdict(float)
            

            for sim_user_idx in similar_users:
                sim_score = user_similarity[sim_user_idx]
                user_ratings = self.user_item_matrix[sim_user_idx]

                for post_idx, rating in enumerate(user_ratings):
                    if rating > 0 and post_idx < len(self.post_features):
                        post_data = self.post_features.iloc[post_idx]
                        recency_score = self.calculate_recency_score(pd.to_datetime(post_data['created_at']))

                        combined_score = (sim_score * 0.4) + (rating * 0.3) + (recency_score * 0.3)
                        similar_user_posts[post_idx] += combined_score
                # print("Similar user posts are ",similar_user_posts)

            # Sort posts by score
            sorted_posts = sorted(similar_user_posts.items(), key=lambda x: x[1], reverse=True)
            # print(f"\n\n Sorted posts are {sorted_posts}")

            recommendations, detailed_recommendations = [], []
            image_count, video_count = 0, 0
            
            for post_idx, score in sorted_posts:
                post_data = self.post_features.iloc[post_idx]
                post_id = int(post_data['post_id'])
                # print("THE POST ID TO BE RECOMMENDED",post_id)
            
            # Separate posts into image and video categories
            image_posts, video_posts = [], []
            for post_idx, score in sorted_posts:
                post_data = self.post_features.iloc[post_idx]
                post_id = int(post_data['post_id'])

                if self.user_item_matrix[user_idx][post_idx] > 0:
                    continue  # Skip already interacted posts

                if post_data['asset_type'] == 'image':
                    image_posts.append((post_id, score))
                else:
                    video_posts.append((post_id, score))

            # recommendations, detailed_recommendations = [], []
            # image_count, video_count = 0, 0

            while len(recommendations) < n_recommendations and (image_posts or video_posts):
                if image_count < 4 and image_posts:
                    post_id, score = image_posts.pop(0)
                    image_count += 1
                elif video_count < 1 and video_posts:
                    post_id, score = video_posts.pop(0)
                    video_count += 1
                else:
                    # If no more videos or images are available, continue recommending whatever is left
                    if image_posts:
                        post_id, score = image_posts.pop(0)
                    elif video_posts:
                        post_id, score = video_posts.pop(0)
                    else:
                        break  # No more posts left to recommend

                recommendations.append(post_id)
                detailed_recommendations.append({'post_id': post_id, 'score': float(score), 'type': 'collaborative'})

                # Reset counters once we complete a cycle
                if image_count == 4 and video_count == 1:
                    image_count, video_count = 0, 0

            # If total recommendations are still insufficient, fallback to popular recommendations
            if len(recommendations) < n_recommendations:
                additional_recs = self.get_popular_recommendations(n_recommendations - len(recommendations),detailed_response=True)
                print(f"Additional recommendations are {additional_recs}")
                recommendations.extend([rec['post_id'] for rec in additional_recs])
                detailed_recommendations.extend(additional_recs)

            return detailed_recommendations if detailed_response else recommendations

        except Exception as e:
            self.logger.error(f"Error in collaborative recommendations: {str(e)}")
            return self.get_popular_recommendations(n_recommendations, detailed_response)


    @lru_cache(maxsize=128)
    def get_popular_recommendations(self, n_recommendations: int = 5, detailed_response: bool = False) -> Union[List[int], List[Dict]]:
        """
        Get trending posts based on engagement metrics while maintaining the original output structure.
        
        Parameters:
        - n_recommendations: Number of recommendations to return
        - detailed_response: If True, return detailed dictionaries; if False, return post IDs
        
        Returns:
        - Union[List[int], List[Dict]]: Either a list of post IDs or detailed recommendation dictionaries
        """
        try:
            # Calculate weighted engagement score
            engagement_weights = {
                'view_count': 0.2,
                'like_count': 0.3,
                'comment_count': 0.2,
                'share_count': 0.2
            }
            
            # Calculate engagement score
            self.post_features['engagement_score'] = (
                self.post_features['view_count'] * engagement_weights['view_count'] +
                self.post_features['like_count'] * engagement_weights['like_count'] +
                self.post_features['comment_count'] * engagement_weights['comment_count'] +
                self.post_features['share_count'] * engagement_weights['share_count'] 
            )
            
            # Add recency factor
            self.post_features['recency_score'] = self.post_features['created_at'].apply(self.calculate_recency_score)
            
            # Final trending score combines engagement and recency
            self.post_features['trending_score'] = (
                self.post_features['engagement_score'] * 0.7 + 
                self.post_features['recency_score'] * 0.3
            )
            
            # Sort and select top posts
            trending_posts = self.post_features.sort_values('trending_score', ascending=False).head(n_recommendations)
            
            if detailed_response:
                # Prepare detailed recommendations with the original structure
                recommendations = [
                    {
                        'post_id': int(row['post_id']),
                        'score': float(row['trending_score']),  # Match the original 'score' key
                        'type': 'popular',  # Match the original 'type' key
                        'recency_score': float(row['recency_score']),  # Match the original 'recency_score' key
                        'popularity_score': float(row['engagement_score']),  # Match the original 'popularity_score' key
                        'rank': rank + 1  # Add rank to match the original structure
                    }
                    for rank, (_, row) in enumerate(trending_posts.iterrows())
                ]
            else:
                # Return just the post IDs in ranked order
                recommendations = list(trending_posts['post_id'].astype(int))
            
            return recommendations
        
        except Exception as e:
            self.logger.error(f"Error getting trending posts: {str(e)}")
            return [] if detailed_response else []

    # Note: The rest of the methods (calculate_recency_score, get_popular_recommendations, etc.)
    # remain the same as they are either helper methods or less frequently called


    @lru_cache(maxsize=128)
    def get_hybrid_recommendations(self, user_id: int, n_recommendations: int = 5,
                                 connection_ratio: Optional[float] = None,
                                 max_post_age_days: Optional[int] = None,
                                 detailed_response: bool = False) -> Union[List[int], List[Dict]]:
        """
        Get hybrid recommendations with caching
        
        Parameters remain the same as original method
        """
        # Generate cache key including all parameters that affect the output
        cache_key = self._get_cache_key(
            'hybrid_recs',
            f"{user_id}:{n_recommendations}:{connection_ratio}:{max_post_age_days}:{detailed_response}"
        )
        
        # Try to get cached recommendations
        cached_data = self._cache_get(cache_key)
        if cached_data:
            try:
                return json.loads(cached_data)
            except json.JSONDecodeError:
                self.logger.error("Error decoding cached hybrid recommendations")
        
        try:
            # Get user's connections with caching
            user_connections = set(self.get_user_connections(user_id))
            self.logger.info(f"Finding posts from connections for user {user_id}: {user_connections}")
            
            current_time = pd.Timestamp.now()
            connection_recommendations = []
            non_connection_recommendations = []
            recommended_post_ids = set()
            
            # Process connection posts
            if user_connections:
                for idx, post in self.post_features.iterrows():
                    try:
                        post_author = int(post['user_id'])
                        post_date = pd.to_datetime(post['created_at'])
                        
                        if post_author in user_connections:
                            # Normalize date handling
                            if isinstance(post_date, str):
                                post_date = pd.to_datetime(post_date).replace(tzinfo=None)
                            elif hasattr(post_date, 'tzinfo') and post_date.tzinfo is not None:
                                post_date = post_date.replace(tzinfo=None)
                                
                            post_age_days = (current_time - post_date).days
                            
                            # Check age criterion
                            if max_post_age_days is not None and post_age_days > max_post_age_days:
                                continue
                                
                            # Calculate recency score with caching
                            recency_score_key = self._get_cache_key('recency_score', f"{post['post_id']}:{post_date}")
                            cached_recency = self._cache_get(recency_score_key)
                            
                            if cached_recency:
                                recency_score = float(cached_recency)
                            else:
                                recency_score = self.calculate_recency_score(post['created_at'])
                                self._cache_set(recency_score_key, str(recency_score).encode())
                            
                            if detailed_response:
                                rec = {
                                    'post_id': int(post['post_id']),
                                    'score': recency_score,
                                    'types': ['connection'],
                                    'from_connection': True,
                                    'author_id': post_author,
                                    'age_days': post_age_days
                                }
                            else:
                                rec = int(post['post_id'])
                                
                            connection_recommendations.append(rec)
                            recommended_post_ids.add(int(post['post_id']))
                            
                    except (ValueError, KeyError) as e:
                        continue
                
                if detailed_response:
                    connection_recommendations.sort(key=lambda x: x['score'], reverse=True)
            
            # Calculate recommendation distribution
            if connection_ratio is not None:
                target_connection_count = int(n_recommendations * connection_ratio)
                target_non_connection_count = n_recommendations - target_connection_count
            else:
                target_connection_count = len(connection_recommendations)
                target_non_connection_count = n_recommendations - target_connection_count
            
            # Get non-connection recommendations if needed
            if target_non_connection_count > 0:
                # Try collaborative filtering with caching
                try:
                    collab_recs = self.get_collaborative_recommendations(user_id, target_non_connection_count * 2)
                    for rec in collab_recs:
                        post_id = rec['post_id']
                        if post_id not in recommended_post_ids:
                            if detailed_response:
                                rec['types'] = ['collaborative']
                                rec['from_connection'] = False
                                non_connection_recommendations.append(rec)
                            else:
                                non_connection_recommendations.append(post_id)
                            recommended_post_ids.add(post_id)
                except Exception as e:
                    self.logger.error(f"Error getting collaborative recommendations: {str(e)}")
                
                # Try content-based if needed
                if len(non_connection_recommendations) < target_non_connection_count:
                    remaining = target_non_connection_count - len(non_connection_recommendations)
                    try:
                        content_recs = self.get_content_based_recommendations(user_id, remaining * 2)
                        for rec in content_recs:
                            post_id = rec['post_id']
                            if post_id not in recommended_post_ids:
                                if detailed_response:
                                    rec['types'] = ['content']
                                    rec['from_connection'] = False
                                    non_connection_recommendations.append(rec)
                                else:
                                    non_connection_recommendations.append(post_id)
                                recommended_post_ids.add(post_id)
                    except Exception as e:
                        self.logger.error(f"Error getting content-based recommendations: {str(e)}")
                
                if detailed_response:
                    non_connection_recommendations.sort(key=lambda x: x['score'], reverse=True)
            
            # Combine recommendations
            final_recommendations = (
                connection_recommendations[:target_connection_count] +
                non_connection_recommendations[:target_non_connection_count]
            )
            
            # Add popular recommendations if needed
            if len(final_recommendations) < n_recommendations:
                remaining = n_recommendations - len(final_recommendations)
                try:
                    popular_recs = self.get_popular_recommendations(remaining)
                    for rec in popular_recs:
                        post_id = rec['post_id'] if detailed_response else rec
                        if post_id not in recommended_post_ids:
                            if detailed_response:
                                rec['types'] = ['popular']
                                rec['from_connection'] = False
                                final_recommendations.append(rec)
                            else:
                                final_recommendations.append(post_id)
                            recommended_post_ids.add(post_id)
                except Exception as e:
                    self.logger.error(f"Error getting popular recommendations: {str(e)}")
            
            # Final sorting if detailed
            if detailed_response:
                final_recommendations.sort(key=lambda x: (x['from_connection'], x['score']), reverse=True)
            
            final_recommendations = final_recommendations[:n_recommendations]
            
            # Cache the final recommendations
            self._cache_set(cache_key, json.dumps(final_recommendations).encode())
            
            return final_recommendations
                
        except Exception as e:
            self.logger.error(f"Error in hybrid recommendations: {str(e)}")
            fallback_recs = self.get_popular_recommendations(n_recommendations)
            if not detailed_response:
                return [rec['post_id'] for rec in fallback_recs]
            return fallback_recs