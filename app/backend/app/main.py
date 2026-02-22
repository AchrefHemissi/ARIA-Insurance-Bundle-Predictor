"""FastAPI application entrypoint."""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import health, metrics, model, predict
from app.core.model_loader import ModelStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clear on shutdown."""
    ModelStore.load()
    yield
    ModelStore.clear()


app = FastAPI(
    title="Insurance Bundle Prediction API",
    description="REST API for predicting insurance coverage bundles using a LightGBM model.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(metrics.router)
app.include_router(model.router)
app.include_router(predict.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
