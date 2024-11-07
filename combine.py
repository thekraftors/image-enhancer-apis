import io
import warnings
import base64
import os
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from py_real_esrgan.model import RealESRGAN

from typing import Optional
import jwt
from jwt import PyJWTError
from PIL import Image, ImageEnhance
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define constants
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "your-secret-key")  # Use environment variable
ALGORITHM = "HS256"
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB in bytes
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'gif', 'webp', 'png'}
EXPECTED_AUDIENCE = 'thekraftors.com'

# Suppress NNPACK warnings
warnings.filterwarnings("ignore", message=".*Could not initialize NNPACK!.*")

# Set the precision for matrix multiplication
torch.set_float32_matmul_precision("high")

class BackgroundRemover:
    def __init__(self):
        self.model = self.load_model()
        self.transform_image = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    def load_model(self):
        model = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True)
        model.to("cpu")
        return model
    
    def remove_background(self, input_image: Image.Image) -> Optional[Image.Image]:
        try:
            image_size = input_image.size
            if input_image.mode == 'RGBA':
                input_image = input_image.convert('RGB')
            input_tensor = self.transform_image(input_image).unsqueeze(0).to("cpu")
            with torch.no_grad():
                preds = self.model(input_tensor)[-1].sigmoid().cpu()
            pred = preds[0].squeeze()
            pred_pil = transforms.ToPILImage()(pred)
            mask = pred_pil.resize(image_size)
            result = Image.new('RGBA', image_size, (0, 0, 0, 0))
            result.paste(input_image, (0, 0), mask)
            return result
        except Exception as ex:
            logger.error(f"Error in background removal: {str(ex)}")
            return None

def enhance_image(input_image: Image.Image) -> Optional[Image.Image]:
    try:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        model = RealESRGAN(device, scale=4)
        model.load_weights('weights/RealESRGAN_x4.pth', download=True)
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
        width, height = input_image.size
        if width < 15 or height < 8:
            return "Image dimensions are too small. Minimum dimensions are 15x8."
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
    except PyJWTError as e:
        logger.error(f"Error decoding token: {str(e)}")
        raise HTTPException(status_code=401, detail="Invalid or expired token")

def verify_jwt_token(authorization: str = Header(...)) -> dict:
    try:
        scheme, token = authorization.split()
        if scheme.lower() != 'bearer':
            raise HTTPException(status_code=401, detail="Invalid authentication scheme")
        return decode_jwt(token)
    except ValueError:
        logger.error("Invalid or missing Bearer token")
        raise HTTPException(status_code=401, detail="Invalid or missing Bearer token")

# Initialize FastAPI app and BackgroundRemover
app = FastAPI()
bg_remover = BackgroundRemover()

# Add CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StandardResponse(BaseModel):
    code: int
    message: str
    data: Optional[dict] = None

@app.post("/process_image/", response_model=StandardResponse)
async def process_image(
    file: UploadFile = File(...),
    authorization: str = Depends(verify_jwt_token)  # Use JWT verification here
):
    try:
        if not file.content_type.startswith('image/'):
            return StandardResponse(code=400, message="Invalid file type")
        
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension not in ALLOWED_EXTENSIONS:
            return StandardResponse(code=400, message="Please enter a valid image")
        
        # Read file content to check the size
        file_content = await file.read()
        if len(file_content) > MAX_FILE_SIZE:
            return StandardResponse(code=400, message="Image size should not be greater than 5 MB")
        
        # Reset the file pointer to the beginning
        file.file.seek(0)
        input_image = Image.open(io.BytesIO(file_content)).convert('RGBA')

        # Validate the image before processing
        validation_error = validate_image(input_image)
        if validation_error:
            return StandardResponse(code=400, message=validation_error)

        # Enhance the image first
        enhanced_image = enhance_image(input_image.convert('RGB'))
        if not enhanced_image:
            return StandardResponse(code=500, message="Image enhancement failed")
        enhanced_image = enhanced_image.convert('RGBA')

        # Remove background from the enhanced image
        processed_image = bg_remover.remove_background(enhanced_image)
        if not processed_image:
            return StandardResponse(code=500, message="Background removal failed")

        output_format = 'PNG'
        img_byte_arr = io.BytesIO()
        processed_image.save(img_byte_arr, format=output_format)
        img_byte_arr.seek(0)
        
        # Convert image bytes to base64 string
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        return StandardResponse(code=200, message="Success", data={"image_base64": img_base64})
    
    except ValueError as ve:
        return StandardResponse(code=500, message=str(ve))
    except Exception as e:
        return StandardResponse(code=500, message=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5005) 
