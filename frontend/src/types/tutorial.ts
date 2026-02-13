export interface CodeBlock {
  filename: string;
  language: string;
  code: string;
  description: string;
}

export interface Task {
  title: string;
  instructions: string;
}

export interface TutorialStep {
  id: number;
  title: string;
  subtitle: string;
  theory: string;
  tasks: Task[];
  code_blocks: CodeBlock[];
}

export interface TutorialOverview {
  id: number;
  title: string;
  subtitle: string;
}
