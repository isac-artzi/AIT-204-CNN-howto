import { useState } from "react";
import { useStepList, useStep } from "./hooks/useTutorial";
import StepNavigation from "./components/StepNavigation";
import ProgressBar from "./components/ProgressBar";
import TutorialLayout from "./components/TutorialLayout";
import "./App.css";

export default function App() {
  const { steps, loading: stepsLoading } = useStepList();
  const [currentStepId, setCurrentStepId] = useState(1);
  const { step, loading: stepLoading } = useStep(currentStepId);

  if (stepsLoading) {
    return <div className="loading">Loading tutorial...</div>;
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-content">
          <h1 className="app-title">
            Build a CNN with PyTorch
          </h1>
          <p className="app-subtitle">
            CIFAR-10 Image Classification — Interactive Tutorial
          </p>
        </div>
      </header>

      <div className="nav-bar">
        <ProgressBar current={currentStepId} total={steps.length} />
        <StepNavigation
          steps={steps}
          currentStep={currentStepId}
          onStepChange={setCurrentStepId}
        />
      </div>

      {step && !stepLoading ? (
        <>
          <div className="step-header">
            <h2 className="step-title">
              Step {step.id}: {step.title}
            </h2>
            <p className="step-subtitle">{step.subtitle}</p>
          </div>
          <TutorialLayout step={step} />
        </>
      ) : (
        <div className="loading">Loading step...</div>
      )}
    </div>
  );
}
