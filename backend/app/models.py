from pydantic import BaseModel


class CodeBlock(BaseModel):
    filename: str
    language: str
    code: str
    description: str


class Task(BaseModel):
    title: str
    instructions: str


class TutorialStep(BaseModel):
    id: int
    title: str
    subtitle: str
    theory: str  # Markdown content
    tasks: list[Task]
    code_blocks: list[CodeBlock]


class TutorialOverview(BaseModel):
    id: int
    title: str
    subtitle: str
