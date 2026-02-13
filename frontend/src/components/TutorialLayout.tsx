import TheoryPanel from "./TheoryPanel";
import TaskPanel from "./TaskPanel";
import CodePanel from "./CodePanel";
import type { TutorialStep } from "../types/tutorial";

interface Props {
  step: TutorialStep;
}

export default function TutorialLayout({ step }: Props) {
  return (
    <div className="tutorial-layout">
      <TheoryPanel content={step.theory} />
      <TaskPanel tasks={step.tasks} />
      <CodePanel codeBlocks={step.code_blocks} />
    </div>
  );
}
