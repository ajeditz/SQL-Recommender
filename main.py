from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal, Optional
import pandas as pd
from recommender import TravelRecommender

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


class TrendingRecommendationsRequest(BaseModel):
    num_recommendations: int = Field(
        default=10, 
        ge=1, 
        le=50, 
        description="Number of trending recommendations to return"
    )
    detailed_response:bool =False

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
try:
    posts_df = pd.read_csv("post_interest_view_cached.csv")
    interactions_df = pd.read_csv("post_activity_cached.csv")
    connections_df = pd.read_csv("user_followers_following_cached.csv")
    
    recommender = TravelRecommender()
    recommender.update_models(posts_df, interactions_df)
    recommender.set_user_connections(connections_df)
    
except Exception as e:
    print(f"Error initializing recommender: {str(e)}")
    raise

@app.post("/recommend", response_model_exclude_none=True, summary="Get personalized recommendations")
async def recommend_posts(request: RecommendationRequest):
    try:
        recommendations = recommender.get_hybrid_recommendations(
            user_id=request.user_id,
            n_recommendations=request.num_recommendations,
            connection_ratio=request.connection_ratio,
            max_post_age_days=request.max_post_age_days,
            detailed_response=request.detailed_response
        )
        
        if not recommendations:
            raise HTTPException(status_code=404, detail=f"No recommendations found for user {request.user_id}")

        # Paginate the recommendations
        start_idx = (request.page - 1) * request.page_size
        end_idx = start_idx + request.page_size
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


@app.post("/explore",
          response_model_exclude_none=True,
          summary="Get collaborative recommendations")
async def get_collaborative_recommendations(request: RecommendationRequest):
    """Get recommendations based on collaborative filtering only"""
    try:
        recommendations = recommender.get_collaborative_recommendations(
            user_id=request.user_id,
            n_recommendations=request.num_recommendations,
            detailed_response=request.detailed_response  # Fixed typo from 'detailed_respons'
        )

        if not recommendations:
            raise HTTPException(
                status_code=404,
                detail=f"No collaborative recommendations found for user {request.user_id}"
            )

        return {
            "user_id": request.user_id,
            "num_recommendations": len(recommendations),
            "recommendations": recommendations
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
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
    uvicorn.run("main:app",host="127.0.0.1",port=9000,reload=True)