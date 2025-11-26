from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any
import logging

from app.pydantic_models.shorts.generate_shorts_request_model import GenerateShortsRequestModel
from app.dependencies.reel_dependencies import get_reel_service
from app.services.reel_service import ReelService
from app.dependencies.security_deps import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/reels", tags=["reels"])


@router.post("/extract")
async def extract_shorts(
    reel_request: GenerateShortsRequestModel,
    service: ReelService = Depends(get_reel_service)
    # TODO: Enable auth for production deployment
) -> Dict[str, Any]:
    """
    Extract highlight shorts from a YouTube video.
    
    This endpoint processes a YouTube video to create short-form content:
    - Downloads video and audio from YouTube
    - Transcribes the audio content

    - Uses AI to identify highlight moments
    - Creates face-centered 9:16 format video clips
    - Applies automatic captions
    
    Args:
        reel_request: Video processing parameters including YouTube URL and preferences
        service: Reel processing service (automatically injected)
    
    Returns:
        Dictionary containing processing results and output file paths
        
    Raises:
        HTTPException: 
            - 422: Invalid input parameters
            - 500: Processing error
    """
    try:
        logger.info(f"Processing reel extraction request - URL: {reel_request.youtube_url}")

        # Process the reel without user context
        result = await service.process_reel(reel_data_input=reel_request)

        logger.info("Reel extraction completed successfully")
        
        return result
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Error processing reel extraction: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your request. Please try again later."
        )


@router.get("/status/{processing_id}")
async def get_processing_status(
    processing_id: str
) -> Dict[str, Any]:
    """
    Get the status of a video processing job.
    
    **Authentication Required**: Valid JWT token must be provided.
    Users can only access their own processing jobs.
    
    Args:
        processing_id: Unique identifier for the processing job
    
    Returns:
        Dictionary containing processing status and progress information
        
    Raises:
        HTTPException:
            - 401: Authentication required or invalid token
            - 403: Access denied (not your processing job)
            - 404: Processing job not found
    """
    try:
        logger.info(f"Status check requested for processing ID: {processing_id}")
        
        # TODO: Implement processing status tracking
        # This would typically involve checking a database or cache
        # for the processing job status
        
        # Placeholder response
        return {
            "processing_id": processing_id,
            "status": "not_implemented",
            "message": "Processing status tracking not yet implemented"
        }
        
    except Exception as e:
        logger.error(
            f"Error checking processing status for processing_id {processing_id}: {str(e)}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error checking processing status"
        )


@router.get("/history")
async def get_user_history(
    limit: int = 10,
    offset: int = 0
) -> Dict[str, Any]:
    """
    Get user's processing history.
    
    **Authentication Required**: Valid JWT token must be provided.
    Returns only the authenticated user's processing history.
    
    Args:
        limit: Maximum number of records to return (default: 10, max: 100)
        offset: Number of records to skip for pagination (default: 0)
    
    Returns:
        Dictionary containing user's processing history
        
    Raises:
        HTTPException:
            - 401: Authentication required or invalid token
            - 422: Invalid pagination parameters
    """
    try:
        # Validate pagination parameters
        if limit < 1 or limit > 100:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Limit must be between 1 and 100"
            )
        
        if offset < 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Offset must be non-negative"
            )
        
        logger.info(f"History requested (limit: {limit}, offset: {offset})")
        
        # TODO: Implement user history tracking
        # This would typically involve querying a database
        # for the user's processing history
        
        # Placeholder response
        return {
            "user_id": "anonymous",
            "history": [],
            "total_count": 0,
            "limit": limit,
            "offset": offset,
            "message": "User history tracking not yet implemented"
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Error fetching history: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching user history"
        )

