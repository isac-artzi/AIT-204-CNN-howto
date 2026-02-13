import { useState, useEffect } from "react";
import type { TutorialStep, TutorialOverview } from "../types/tutorial";

const API_BASE = "/api";

export function useStepList() {
  const [steps, setSteps] = useState<TutorialOverview[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API_BASE}/steps`)
      .then((res) => res.json())
      .then((data) => {
        setSteps(data);
        setLoading(false);
      });
  }, []);

  return { steps, loading };
}

export function useStep(stepId: number) {
  const [step, setStep] = useState<TutorialStep | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    fetch(`${API_BASE}/steps/${stepId}`)
      .then((res) => res.json())
      .then((data) => {
        setStep(data);
        setLoading(false);
      });
  }, [stepId]);

  return { step, loading };
}
