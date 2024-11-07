import io
import base64
import os
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import jwt
from jwt import PyJWTError
from PIL import Image
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
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
EXPECTED_AUDIENCE = 'thekraftors.com'
class StandardResponse(BaseModel):
    code: int
    message: str
    data: Optional[dict] = None

class BackgroundRemover:
    def __init__(self):
        self.device = torch.device("cpu")  # Force CPU usage for now
        self.model = self.load_model()
        self.transform_image = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def load_model(self):
        model = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True)
        model.to(self.device)
        return model

    def remove_background(self, input_image: Image.Image) -> Optional[Image.Image]:
        try:
            image_size = input_image.size
            if input_image.mode == 'RGBA':
                input_image = input_image.convert('RGB')
            input_tensor = self.transform_image(input_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Call the model without any device_type argument
                outputs = self.model(input_tensor)
                
                # Check if the output is a tuple or list and get the last element
                if isinstance(outputs, (tuple, list)):
                    preds = outputs[-1]
                else:
                    preds = outputs
                
                preds = preds.sigmoid().cpu()

            pred = preds[0].squeeze()
            pred_pil = transforms.ToPILImage()(pred)
            mask = pred_pil.resize(image_size)
            result = Image.new('RGBA', image_size, (0, 0, 0, 0))
            result.paste(input_image, (0, 0), mask)
            return result
        except Exception as ex:
            logger.error(f"Error in background removal: {str(ex)}")
            return None

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
@app.post("/remove_background/", response_model=StandardResponse)
async def remove_background(
    file: UploadFile = File(...),
    authorization: str = Depends(verify_jwt_token)
):
    try:
        logger.info("Processing image for background removal...")
        if not file.content_type.startswith('image/'):
            return StandardResponse(code=400, message="Invalid file type")
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension not in ALLOWED_EXTENSIONS:
            return StandardResponse(code=400, message="Please enter a valid image")
        file_content = await file.read()
        if len(file_content) > MAX_FILE_SIZE:
            return StandardResponse(code=400, message="Image size should not be greater than 5 MB")
        input_image = Image.open(io.BytesIO(file_content))
        background_remover = BackgroundRemover()
        processed_image = background_remover.remove_background(input_image)
        if processed_image is None:
            return StandardResponse(code=500, message="Background removal failed")
        img_byte_arr = io.BytesIO()
        processed_image.save(img_byte_arr, format='PNG')  # Save as PNG for transparency
        img_byte_arr.seek(0)
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        return StandardResponse(code=200, message="Background removed successfully", data={"image_base64": img_base64})
    except Exception as e:
        logger.error(f"Error in remove_background: {str(e)}")
        return StandardResponse(code=500, message="An unexpected error occurred")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5004)
