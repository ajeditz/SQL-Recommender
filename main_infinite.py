from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Dict 
import json
import pandas as pd
# from recommender import TravelRecommender
from recommender_inmemory import  TravelRecommender
import pymysql
import asyncio
from sqlalchemy import create_engine
import logging
import os
from dotenv import load_dotenv
load_dotenv()

# Database Connection Details
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv("DB_PORT")
DB_DATABASE = os.getenv("DB_DATABASE")
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")

class RecommendationRequest(BaseModel):
    user_id: int = Field(..., description="User ID to get recommendations for")
    num_recommendations: int = Field(default=10, ge=1, le=50, description="Number of recommendations to return")
    connection_ratio: Optional[float] = Field(
        default=None, 
        ge=0.0, 
        le=1.0, 
        description="Ratio of connection posts to include (0.0 to 1.0). If None, prioritizes all connection posts first"
    )
    max_post_age_days: Optional[int] = Field(
        default=30, 
        ge=1, 
        description="Maximum age of connection posts in days"
    )
    detailed_response: bool = False
    page: int = Field(default=1, ge=1, description="Page number for paginated results")
    page_size: int = Field(default=10, ge=1, le=50, description="Number of results per page")

class ExploreRecommendationRequest(BaseModel):
    user_id: int = Field(..., description="User ID to get recommendations for")
    num_recommendations: int = Field(default=10, ge=1, le=100, description="Number of recommendations to return")
    detailed_response: bool = False
    page: int = Field(default=1, ge=1, description="Page number for paginated results")
    page_size: int = Field(default=10, ge=1, le=50, description="Number of results per page")


class TrendingRecommendationsRequest(BaseModel):
    num_recommendations: int = Field(
        default=10, 
        ge=1, 
        le=1000, 
        description="Number of trending recommendations to return"
    )
    detailed_response:bool =False

class MoreLikeThisRequest(BaseModel):
    num_recommendations: int = Field(
        default=10, 
        ge=1, 
        le=1000, 
        description="Number of similar posts to return"
    )
    post_id:int

class AddInteraction(BaseModel):
    user_id: int = Field(..., description="User ID performing the interaction")
    post_id: int = Field(..., description="Post ID being interacted with")
    interaction_type: Literal['like', 'save', 'comment', 'view', 'share'] = Field(
        ..., 
        description="Type of interaction"
    )

# Initialize FastAPI app
app = FastAPI(
    title="Travel Post Recommendation API",
    description="API for getting personalized travel post recommendations",
    version="1.0.0"
)

# Initialize data and recommender
# try:
#     DATABASE_URL = f"mysql+pymysql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_DATABASE}"
#     engine = create_engine(DATABASE_URL)
    
#     # Fetch data from RDS MySQL tables
#     posts_df = pd.read_sql("SELECT * FROM post_interest_view", con=engine)
#     interactions_df = pd.read_sql("SELECT * FROM post_activity", con=engine)
#     connections_df = pd.read_sql("SELECT * FROM user_followers_following", con=engine)

#     # Initialize recommender system
#     recommender = TravelRecommender()
#     recommender.update_models(posts_df, interactions_df)
#     recommender.set_user_connections(connections_df)
    
# except Exception as e:
#     print(f"Error initializing recommender: {str(e)}")
#     raise
DATABASE_URL = f"mysql+pymysql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_DATABASE}"
engine = create_engine(DATABASE_URL)

recommender = TravelRecommender()

async def fetch_data():
    """Periodically fetch fresh data and update the recommender system."""
    while True:
        try:
            logging.info("Fetching fresh data from database...")
            
            # Fetch updated tables
            posts_df = pd.read_sql("SELECT * FROM post_interest_view", con=engine)
            interactions_df = pd.read_sql("SELECT * FROM post_activity", con=engine)
            connections_df = pd.read_sql("SELECT * FROM user_followers_following", con=engine)

            # Update recommender models    
            recommender.update_models(posts_df, interactions_df)
            recommender.set_user_connections(connections_df)

            logging.info("Recommender system updated successfully.")

        except Exception as e:
            logging.error(f"Error fetching data: {str(e)}")

        await asyncio.sleep(3600)  # Wait 5 minutes before the next update


from datetime import datetime, timedelta
import json

CACHE_EXPIRY_TIME = timedelta(hours=24)  # Expiry time for recommendations

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(fetch_data())

@app.post("/recommend", response_model_exclude_none=True, summary="Get personalized recommendations")
async def recommend_posts(request: RecommendationRequest, force_refresh: bool = False):
    """Get personalized hybrid recommendations with circular infinite scrolling."""
    try:
        cache_key = f"recommendations:{request.user_id}:{request.detailed_response}"
        cached_data = recommender._cache_get(cache_key)
        
        if force_refresh or not cached_data:
            recommendations = recommender.get_hybrid_recommendations(
                user_id=request.user_id,
                n_recommendations=100,
                connection_ratio=request.connection_ratio,
                max_post_age_days=request.max_post_age_days,
                detailed_response=request.detailed_response
            )
            recommender._cache_set(cache_key, json.dumps(recommendations).encode())
        else:
            recommendations = json.loads(cached_data)
        
        if request.last_item_index is not None:
            next_index = (request.last_item_index + request.num_recommendations) % len(recommendations)
        else:
            next_index = request.num_recommendations
        
        paginated_recommendations = recommendations[next_index - request.num_recommendations:next_index]
        
        return {
            "user_id": request.user_id,
            "num_recommendations": len(paginated_recommendations),
            "next_index": next_index,
            "recommendations": paginated_recommendations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

from datetime import datetime, timedelta

CACHE_EXPIRY_TIME = timedelta(hours=24)  # Set cache expiry time (e.g., refresh every 24 hours)

@app.post("/explore",
          response_model_exclude_none=True,
          summary="Get collaborative recommendations")
async def get_collaborative_recommendations(request: ExploreRecommendationRequest, force_refresh: bool = False):
    """Get recommendations based on collaborative filtering only"""

    try:
        cache_key = f"collab_recs:{request.user_id}:{request.detailed_response}"
        cache_timestamp_key = f"collab_recs_timestamp:{request.user_id}"  # Store cache timestamp
        
        # Check if cache exists
        cached_data = recommender._cache_get(cache_key)
        cache_timestamp = recommender._cache_get(cache_timestamp_key)

        # Convert timestamp from cache
        cache_time = datetime.fromisoformat(cache_timestamp.decode()) if cache_timestamp else None
        cache_expired = cache_time and (datetime.utcnow() - cache_time) > CACHE_EXPIRY_TIME

        if force_refresh or not cached_data or cache_expired:
            # If forced refresh or cache expired, generate new recommendations
            recommendations = recommender.get_collaborative_recommendations(
                user_id=request.user_id,
                n_recommendations=request.num_recommendations,  # Get max recommendations
                detailed_response=request.detailed_response
            )

            # Update cache with new recommendations
            recommender._cache_set(cache_key, json.dumps(recommendations).encode())
            recommender._cache_set(cache_timestamp_key, datetime.utcnow().isoformat().encode())
        else:
            # Load from cache
            try:
                recommendations = json.loads(cached_data)
            except json.JSONDecodeError:
                recommender.logger.error("Error decoding cached recommendations")
                recommendations = []

        # Handle case where recommendations are exhausted
        total_available = len(recommendations)
        start_idx = (request.page - 1) * request.page_size
        end_idx = start_idx + request.page_size

        if start_idx >= total_available:
            # If no more cached recommendations left, trigger a refresh
            recommendations = recommender.get_collaborative_recommendations(
                user_id=request.user_id,
                n_recommendations=request.num_recommendations,  # Get new recommendations
                detailed_response=request.detailed_response
            )

            # Update cache with new recommendations
            recommender._cache_set(cache_key, json.dumps(recommendations).encode())
            recommender._cache_set(cache_timestamp_key, datetime.utcnow().isoformat().encode())

            # Reset pagination
            start_idx, end_idx = 0, request.page_size

        paginated_recommendations = recommendations[start_idx:end_idx]

        return {
            "user_id": request.user_id,
            "num_recommendations": len(paginated_recommendations),
            "total_recommendations": len(recommendations),
            "page": request.page,
            "page_size": request.page_size,
            "recommendations": paginated_recommendations
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/more_like_this")
async def get_similar_posts(request:MoreLikeThisRequest):
    try:
        recommendations= recommender.get_similar_posts(
            post_id=request.post_id,
            n_recommendations=request.num_recommendations
        )

        if not recommendations:
            raise HTTPException(status_code=404, detail="No similar posts found")

        return {
            "num_recommendations": len(recommendations),
            "recommendations": recommendations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    


    
@app.post("/trending",  # Changed from "trending_posts" for consistency
          response_model_exclude_none=True,
          summary="Get trending posts")
async def get_trending_posts(request: TrendingRecommendationsRequest):
    """Get current trending posts based on popularity and recency"""
    try:

        # popularity_df = pd.read_csv(r"D:\app_square\recommedaiton_engine\recommdder_program\Recommender-Program\pilot_posts2.csv")
        recommendations = recommender.get_popular_recommendations(
            n_recommendations=request.num_recommendations,
            detailed_response=request.detailed_response,
            # popularity_df=popularity_df
        )

        if not recommendations:
            raise HTTPException(status_code=404, detail="No trending posts found")

        return {
            "num_recommendations": len(recommendations),
            "recommendations": recommendations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    


@app.post("/interactions",  # Changed from "add_interaction" for consistency
          response_model_exclude_none=True,
          summary="Record a user interaction with a post")
async def add_interaction(request: AddInteraction):
    """Record a new user interaction (like, save, comment, etc.) with a post"""
    try:
        success = recommender.add_new_interaction(
            request.user_id,
            request.post_id,
            request.interaction_type
        )

        if not success:
            raise HTTPException(
                status_code=400,
                detail="Failed to record interaction"
            )

        return {"status": "success", "message": "Interaction recorded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")  # Changed from root() for clarity
async def health_check():
    """Check if the API is running"""
    return {
        "status": "healthy",
        "message": "Recommendation API is running"
    }


if __name__=="__main__":
    import uvicorn
    uvicorn.run("main:app",host="127.0.0.1",port=8000,reload=True)