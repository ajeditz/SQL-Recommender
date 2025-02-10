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
                index='user_id',
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


