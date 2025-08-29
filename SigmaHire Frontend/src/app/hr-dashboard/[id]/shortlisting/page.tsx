import StagePageUI from '../Components/StagePageUI';

const candidates = [
  { name: 'Ananya Mehta', email: 'ananya@example.com', score: 92 },
  { name: 'Rahul Sharma', email: 'rahul@example.com', score: 88 },
  { name: 'Priya Das', email: 'priya@example.com', score: 84 },
  { name: 'Kunal Roy', email: 'kunal@example.com', score: 78 },
];

export default function ShortlistingStage() {
  return (
    <StagePageUI
      title="Shortlisting"
      description="Ask questions to help filter and prioritize candidates based on their resume."
      prompts={[
      ]}
      nextStageName="Assessment Stage"
      candidates={candidates}
    />
  );
}