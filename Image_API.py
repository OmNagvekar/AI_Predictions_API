from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import imghdr
import uvicorn
import os
from garbage import Predictions

app = FastAPI()
UPLOAD_DIR = "upload"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post('/')
async def root(file: UploadFile= File()):
    # raising exception if received file type is not image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
    # read file
    file_content = await file.read()
    # Validate file as an image using imghdr
    file_type = imghdr.what(None, h=file_content)
    if  file_type not in ["jpeg", "png", "bmp", "tiff","jpg"]:
        raise HTTPException(status_code=400, detail="Unsupported image type.")
    
    # Save the file to disk
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(file_content)
    return JSONResponse(content={"filename": file.filename, "message": "Image uploaded successfully!"})

async def get_pred():
    pass

def garbage_intensity():
    model = Predictions(file_path)


if __name__=='__main__':
    uvicorn.run('Image_API:app',host='localhost',port=8000,reload=True)