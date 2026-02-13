import type { TutorialOverview } from "../types/tutorial";

interface Props {
  steps: TutorialOverview[];
  currentStep: number;
  onStepChange: (stepId: number) => void;
}

export default function StepNavigation({
  steps,
  currentStep,
  onStepChange,
}: Props) {
  const currentIdx = steps.findIndex((s) => s.id === currentStep);
  const hasPrev = currentIdx > 0;
  const hasNext = currentIdx < steps.length - 1;

  return (
    <div className="step-navigation">
      <button
        className="nav-btn"
        disabled={!hasPrev}
        onClick={() => hasPrev && onStepChange(steps[currentIdx - 1].id)}
      >
        &larr; Previous
      </button>

      <select
        className="step-select"
        value={currentStep}
        onChange={(e) => onStepChange(Number(e.target.value))}
      >
        {steps.map((s) => (
          <option key={s.id} value={s.id}>
            {s.id}. {s.title}
          </option>
        ))}
      </select>

      <button
        className="nav-btn nav-btn-primary"
        disabled={!hasNext}
        onClick={() => hasNext && onStepChange(steps[currentIdx + 1].id)}
      >
        Next &rarr;
      </button>
    </div>
  );
}
