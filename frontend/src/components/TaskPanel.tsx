import ReactMarkdown from "react-markdown";
import type { Task } from "../types/tutorial";

interface Props {
  tasks: Task[];
}

export default function TaskPanel({ tasks }: Props) {
  return (
    <div className="panel task-panel">
      <div className="panel-header">
        <span className="panel-icon">&#128736;</span> Development Tasks
      </div>
      <div className="panel-body">
        {tasks.map((task, i) => (
          <div key={i} className="task-card">
            <div className="task-number">{i + 1}</div>
            <div className="task-content">
              <h3 className="task-title">{task.title}</h3>
              <div className="markdown-body">
                <ReactMarkdown>{task.instructions}</ReactMarkdown>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
