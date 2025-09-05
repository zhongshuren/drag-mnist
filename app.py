from fastapi import FastAPI
import uvicorn
from omegaconf import OmegaConf

from app.api_wrapper import wrapper
from app.content import DragMNISTManager

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

configs = OmegaConf.load('config/config_app.yaml')

app = FastAPI()
manager = DragMNISTManager(configs)
app = wrapper(app, content_manager=manager)

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)