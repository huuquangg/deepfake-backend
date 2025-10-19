from fastapi import FastAPI
from app.core.middleware import setup_middleware
from app.routes import include_routers

def create_app() -> FastAPI:
    app = FastAPI(title="Deepfake Backend API")
    setup_middleware(app)
    include_routers(app)
    return app

app = create_app()
