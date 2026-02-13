import { useState } from "react";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism";
import type { CodeBlock } from "../types/tutorial";

interface Props {
  codeBlocks: CodeBlock[];
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  return (
    <button className="copy-btn" onClick={handleCopy}>
      {copied ? "Copied!" : "Copy"}
    </button>
  );
}

export default function CodePanel({ codeBlocks }: Props) {
  return (
    <div className="panel code-panel">
      <div className="panel-header">
        <span className="panel-icon">&#128187;</span> Python Code
      </div>
      <div className="panel-body">
        {codeBlocks.map((block, i) => (
          <div key={i} className="code-block">
            <div className="code-block-header">
              <span className="code-filename">{block.filename}</span>
              <CopyButton text={block.code} />
            </div>
            <p className="code-description">{block.description}</p>
            <SyntaxHighlighter
              language={block.language === "text" ? "plaintext" : block.language}
              style={vscDarkPlus}
              customStyle={{
                margin: 0,
                borderRadius: "0 0 8px 8px",
                fontSize: "13px",
              }}
              showLineNumbers
            >
              {block.code}
            </SyntaxHighlighter>
          </div>
        ))}
        {codeBlocks.length === 0 && (
          <div className="no-code">
            <p>No code for this step — focus on the theory and tasks!</p>
          </div>
        )}
      </div>
    </div>
  );
}
