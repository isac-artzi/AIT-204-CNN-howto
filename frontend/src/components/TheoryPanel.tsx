import ReactMarkdown from "react-markdown";

interface Props {
  content: string;
}

export default function TheoryPanel({ content }: Props) {
  return (
    <div className="panel theory-panel">
      <div className="panel-header">
        <span className="panel-icon">&#128218;</span> Theory &amp; Concepts
      </div>
      <div className="panel-body markdown-body">
        <ReactMarkdown>{content}</ReactMarkdown>
      </div>
    </div>
  );
}
