import numpy as np
from PIL import Image, ImageEnhance
from py_real_esrgan.model import RealESRGAN
import torch
import io
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import jwt
from jwt import PyJWTError
import logging
from datetime import datetime, timedelta
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define constants
REAL_ESRGAN_WEIGHTS = 'weights/RealESRGAN_x4.pth'
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "your-secret-key")  # Use environment variable
ALGORITHM = "HS256"
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB in bytes
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'gif', 'webp', 'png'}
EXPECTED_AUDIENCE = 'thekraftors.com'
TOKEN_EXPIRATION = timedelta(hours=1)  # Token expires after 1 hour

def enhance_image(input_image: Image.Image) -> Optional[Image.Image]:
    try:
        logger.info("Enhancing image...")
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        model = RealESRGAN(device, scale=4)
        model.load_weights(REAL_ESRGAN_WEIGHTS, download=True)
        with torch.no_grad():
            sr_image = model.predict(input_image)
        contrast_enhancer = ImageEnhance.Contrast(sr_image)
        sr_image = contrast_enhancer.enhance(1.2)
        brightness_enhancer = ImageEnhance.Brightness(sr_image)
        sr_image = brightness_enhancer.enhance(1.1)
        return sr_image
    except Exception as ex:
        logger.error(f"Error in image enhancement: {str(ex)}")
        return None

def validate_image(input_image: Image.Image) -> Optional[str]:
    try:
        logger.info("Validating image size...")
        width, height = input_image.size
        if width < 15 or height < 8:
            return "Image dimensions are too small for enhancement. Minimum dimensions are 15x8."
        return None
    except Exception as ex:
        logger.error(f"Error in image validation: {str(ex)}")
        return "Invalid image format or size."

def decode_jwt(token: str) -> Optional[dict]:
    try:
        logger.info(f"Attempting to decode token: {token[:10]}...")  # Log first 10 chars of token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM], audience=EXPECTED_AUDIENCE)
        logger.info("Token decoded successfully")
        return payload
    except jwt.InvalidSignatureError:
        logger.error(f"Invalid token signature. Used SECRET_KEY: {SECRET_KEY[:5]}...")  # Log first 5 chars of secret key
        raise HTTPException(status_code=401, detail="Invalid token signature")
    except Exception as e:
        logger.error(f"Error decoding token: {str(e)}")
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware with more restrictive settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://thekraftors.com"],  # Replace with your frontend domain
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

class StandardResponse(BaseModel):
    code: int
    message: str
    data: Optional[dict] = None

def verify_jwt_token(authorization: str = Header(...)) -> dict:
    try:
        scheme, token = authorization.split()
        if scheme.lower() != 'bearer':
            raise HTTPException(status_code=401, detail="Invalid authentication scheme")
        payload = decode_jwt(token)
        return payload
    except ValueError:
        logger.error("Invalid or missing Bearer token")
        raise HTTPException(status_code=401, detail="Invalid or missing Bearer token")

@app.post("/enhancement/", response_model=StandardResponse)
async def process_image(
    file: UploadFile = File(...),
    authorization: str = Depends(verify_jwt_token)
):
    try:
        logger.info("Processing image...")
        if not file.content_type.startswith('image/'):
            return StandardResponse(code=400, message="Invalid file type")
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension not in ALLOWED_EXTENSIONS:
            return StandardResponse(code=400, message="Please enter a valid image")
        
        file_content = await file.read()
        if len(file_content) > MAX_FILE_SIZE:
            return StandardResponse(code=400, message="Image size should not be greater than 5 MB")
        
        input_image = Image.open(io.BytesIO(file_content)).convert('RGB')
        
        validation_error = validate_image(input_image)
        if validation_error:
            return StandardResponse(code=400, message=validation_error)
        
        processed_image = enhance_image(input_image)
        if not processed_image:
            return StandardResponse(code=500, message="Image enhancement failed")
        
        img_byte_arr = io.BytesIO()
        processed_image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        return StandardResponse(code=200, message="Success", data={"image_base64": img_base64})
    except Exception as e:
        logger.error(f"Error in process_image: {str(e)}")
        return StandardResponse(code=500, message="An unexpected error occurred")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5003)
