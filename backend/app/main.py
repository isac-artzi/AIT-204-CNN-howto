from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.models import TutorialStep, TutorialOverview
from app.tutorial_content import STEPS

app = FastAPI(title="CNN Tutorial API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/api/steps", response_model=list[TutorialOverview])
def list_steps():
    """Return a summary of all tutorial steps."""
    return [
        TutorialOverview(id=s.id, title=s.title, subtitle=s.subtitle)
        for s in STEPS
    ]


@app.get("/api/steps/{step_id}", response_model=TutorialStep)
def get_step(step_id: int):
    """Return full content for a single tutorial step."""
    for step in STEPS:
        if step.id == step_id:
            return step
    raise HTTPException(status_code=404, detail="Step not found")
