from fastapi import FastAPI
from fastapi import UploadFile,File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import matplotlib.image as mpimg
from io import BytesIO

from prediction import predict, preprocess, read_image,load_my_model

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/index')
def hello_world(name: str):
    return f"Hello {name}!"

@app.get('/api/model/load')
def hello_world(model_name: str):
    return model_name

@app.post('/api/preprocess')
async def predict_image(model_name: str, file: bytes = File(...)):
    img = mpimg.imread(BytesIO(file))
    lum_img = img[:, :, 0]
    
    image = preprocess(image)
    predictions = predict(image, model_name)
    
    return "The image classified is Fire" if predictions[0]<0 else "The image classified is Smoke" 


@app.post('/api/predict')
async def predict_image(model_name: str, file: bytes = File(...)):
    image = read_image(file)
    image = preprocess(image)
    predictions = predict(image, model_name)
    
    return "The image classified is Fire" if predictions[0]<0 else "The image classified is Smoke"

@app.post('/api/predict2')
async def predict_image(model_name: str):
    # image = read_image(file)
    # image = preprocess(image)
    # predictions = predict(image, model_name)
    
    return "The image classified is Fire"

if __name__ == "__main__":
    uvicorn.run(app, port=8080, host='0.0.0.0')
