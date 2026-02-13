interface Props {
  current: number;
  total: number;
}

export default function ProgressBar({ current, total }: Props) {
  const pct = (current / total) * 100;

  return (
    <div className="progress-container">
      <div className="progress-bar">
        <div className="progress-fill" style={{ width: `${pct}%` }} />
      </div>
      <span className="progress-label">
        Step {current} of {total}
      </span>
    </div>
  );
}
