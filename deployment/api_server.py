"""
FastAPI server for Thai text classification model deployment.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import json
import os
import logging
from typing import Dict, List, Optional
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Thai Text Classification API",
    description="API for Thai sentiment analysis and text classification",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and tokenizer
model = None
tokenizer = None
classifier = None
model_metadata = None

# Request/Response models
class TextInput(BaseModel):
    text: str
    return_probabilities: bool = False

class PredictionResponse(BaseModel):
    text: str
    prediction: str
    confidence: float
    probabilities: Optional[Dict[str, float]] = None
    processing_time: float

class BatchTextInput(BaseModel):
    texts: List[str]
    return_probabilities: bool = False

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_processing_time: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model_info: Optional[Dict] = None

@app.on_event("startup")
async def load_model():
    """Load the model and tokenizer on startup."""
    global model, tokenizer, classifier, model_metadata
    
    try:
        model_path = os.environ.get("MODEL_PATH", "./final_model")
        
        logger.info(f"Loading model from: {model_path}")
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # Create pipeline
        classifier = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Load metadata if available
        metadata_path = os.path.join(model_path, "model_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                model_metadata = json.load(f)
        
        logger.info("Model loaded successfully!")
        logger.info(f"Device: {device}")
        logger.info(f"Model: {model_metadata.get('model_name', 'Unknown') if model_metadata else 'Unknown'}")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def preprocess_thai_text(text: str) -> str:
    """Preprocess Thai text."""
    # Remove excessive whitespace
    text = ' '.join(text.split())
    return text

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic information."""
    return {
        "message": "Thai Text Classification API",
        "version": "1.0.0",
        "endpoints": "/docs for interactive documentation"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        device=str(device),
        model_info=model_metadata
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(input_data: TextInput):
    """Predict sentiment for a single text."""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    import time
    start_time = time.time()
    
    try:
        # Preprocess text
        processed_text = preprocess_thai_text(input_data.text)
        
        # Make prediction
        result = classifier(processed_text)
        
        # Extract prediction and confidence
        prediction = result[0]['label']
        confidence = result[0]['score']
        
        # Get probabilities if requested
        probabilities = None
        if input_data.return_probabilities:
            # Get all class probabilities
            all_results = classifier(processed_text, return_all_scores=True)
            probabilities = {item['label']: item['score'] for item in all_results[0]}
        
        processing_time = time.time() - start_time
        
        return PredictionResponse(
            text=input_data.text,
            prediction=prediction,
            confidence=confidence,
            probabilities=probabilities,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(input_data: BatchTextInput):
    """Predict sentiment for multiple texts."""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    import time
    start_time = time.time()
    
    try:
        predictions = []
        
        for text in input_data.texts:
            text_start_time = time.time()
            
            # Preprocess text
            processed_text = preprocess_thai_text(text)
            
            # Make prediction
            result = classifier(processed_text)
            
            # Extract prediction and confidence
            prediction = result[0]['label']
            confidence = result[0]['score']
            
            # Get probabilities if requested
            probabilities = None
            if input_data.return_probabilities:
                all_results = classifier(processed_text, return_all_scores=True)
                probabilities = {item['label']: item['score'] for item in all_results[0]}
            
            text_processing_time = time.time() - text_start_time
            
            predictions.append(PredictionResponse(
                text=text,
                prediction=prediction,
                confidence=confidence,
                probabilities=probabilities,
                processing_time=text_processing_time
            ))
        
        total_processing_time = time.time() - start_time
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_processing_time=total_processing_time
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Get model information."""
    if model_metadata is None:
        raise HTTPException(status_code=404, detail="Model metadata not available")
    
    return model_metadata

@app.get("/model/stats")
async def model_stats():
    """Get model statistics."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    device = next(model.parameters()).device
    param_count = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "device": str(device),
        "total_parameters": param_count,
        "trainable_parameters": trainable_params,
        "model_size_mb": param_count * 4 / (1024 * 1024),  # Assuming float32
        "vocab_size": tokenizer.vocab_size if tokenizer else None
    }


# Example usage endpoints for testing
@app.get("/examples")
async def get_examples():
    """Get example texts for testing."""
    return {
        "positive_examples": [
            "สินค้าดีมาก คุณภาพเยี่ยม ประทับใจ",
            "บริการดีเยี่ยม พนักงานเป็นกันเอง",
            "อาหารอร่อยมาก บรรยากาศดี"
        ],
        "negative_examples": [
            "ไม่พอใจ บริการแย่มาก",
            "สินค้าไม่ดี คุณภาพต่ำ",
            "ผิดหวังมาก ไม่คุ้มค่า"
        ]
    }

# --- Hugging Face Model Upload Endpoint ---
from fastapi import UploadFile, File, Form
from huggingface_hub import HfApi, HfFolder, RepositoryNotFoundError

@app.post("/huggingface/upload")
async def upload_to_huggingface(
    repo_id: str = Form(..., description="Hugging Face repo id, e.g. username/model-name"),
    token: str = Form(..., description="Hugging Face User Access Token (write permission)"),
    model_dir: str = Form("./final_model", description="Path to model directory to upload")
):
    """
    Upload the current model directory to Hugging Face Hub.
    - repo_id: e.g. username/model-name
    - token: Hugging Face User Access Token (with write access)
    - model_dir: Path to model directory (default: ./final_model)
    """
    try:
        api = HfApi()
        # Set token for this session
        HfFolder.save_token(token)
        # Create repo if it doesn't exist
        try:
            api.repo_info(repo_id, token=token)
        except RepositoryNotFoundError:
            api.create_repo(repo_id, token=token, private=False, exist_ok=True)
        # Upload folder
        api.upload_folder(
            repo_id=repo_id,
            folder_path=model_dir,
            path_in_repo=".",
            token=token,
            commit_message="Upload model from FastAPI server"
        )
        return {"status": "success", "message": f"Model uploaded to https://huggingface.co/{repo_id}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    # Configuration
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))
    workers = int(os.environ.get("WORKERS", 1))
    
    # Run server
    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        workers=workers,
        reload=False
    )
