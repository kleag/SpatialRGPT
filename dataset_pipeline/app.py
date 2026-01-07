import json
import os
import tempfile
import time

from contextlib import asynccontextmanager
from fastapi import APIRouter, FastAPI, File, UploadFile
from typing import Dict, Any

from run_scene_graph import SceneGraphBuilder, parse_args

args = parse_args([])
timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
args.timestamp = timestamp


# Default to the Docker volume path
WEIGHTS_DIR = os.getenv("WEIGHTS_DIR", "/app/weights")

# Use these paths in your model loading functions
SAM_CHECKPOINT = os.path.join(WEIGHTS_DIR, "grounded_sam/sam_vit_h_4b8939.pth")
PERSP_CHECKPOINT = os.path.join(
    WEIGHTS_DIR, "perspective_fields/paramnet_360cities_edina_rpf.pth"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load heavy models here
    print("Loading Grounded-SAM and PerspectiveFields to GPU...")
    app.state.sgb = SceneGraphBuilder(args)

    yield
    # Clean up and release GPU memory
    del app.state.sgb


app = FastAPI(
    title="3D Scene Graph FastAPI Service",
    description="A 3D Scene Graph extraction service",
    version="1.0.0",
    lifespan=lifespan,
)


router = APIRouter()


@router.post("/process-image")
async def process_image(file: UploadFile = File(...)) -> Dict[str, Any]:
    # 'delete=True' is the default; the file vanishes as soon as it is closed
    with tempfile.NamedTemporaryFile(
        delete=True, suffix=f"_{file.filename}"
    ) as temp_file:
        # 1. Read the uploaded content and write to the temp file
        content = await file.read()
        temp_file.write(content)

        # 2. Flush the buffer to ensure the file is written to disk
        temp_file.flush()

        # 3. Call your function using temp_file.name (the absolute path)
        scene_graph = json.loads(app.state.sgb.annotate(temp_file.name).to_json())

    return scene_graph


@router.get("/")
def read_root():
    """
    Landing endpoint.
    """
    return {"message": "Welcome to the 3D Scene Graph extraction API!"}


@router.get("/health")
def health_check():
    """
    Used by Docker/Kubernetes to monitor container health.
    """
    return {"status": "healthy"}


app.include_router(router)
