from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        print(f"Incoming request: {request.method} {request.url}")
        response = await call_next(request)
        return response

def setup_middleware(app: FastAPI):
    app.add_middleware(LoggingMiddleware)
