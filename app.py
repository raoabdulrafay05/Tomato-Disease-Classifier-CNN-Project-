from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse,JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import numpy as np
from model.leaf_prediction import model,MODEL_VERSION
from config.tomato_classes import CLASS_NAMES
from helpers.utlis import read_file_as_image

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="template")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/ping")
async def ping():
    return {"message": "Model is live and loaded correctly!"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img_batch = read_file_as_image(img_bytes)

    try : 
        preds = model.predict(img_batch)[0]
        class_idx = int(np.argmax(preds))
        class_name = CLASS_NAMES[class_idx]
        confidence = float(preds[class_idx])

        return {
            "class": class_name,
            "confidence": confidence
        }
    except Exception as e:
        JSONResponse(status_code=500, content=str(e))

@app.get("/health")
def health_check():
    return {
        'status' : "OK",
        'version': MODEL_VERSION,
        'model_loaded': "Loaded successfully" if model is not None else "Error in loading model" 
    }

# ✅ Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)