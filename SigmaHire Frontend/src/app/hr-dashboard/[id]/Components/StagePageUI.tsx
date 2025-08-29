'use client';

import { useState, useEffect } from 'react';
import Navbar from '@/app/components/Navbar';
import ChatBox from './ChatBox';
import toast from 'react-hot-toast'; // ✅ import
import { useParams } from 'next/navigation';

type Candidate = {
  candidate_id: number;
  full_name: string;
  primary_email: string;
  job_score: number;
};

export default function StagePageUI({
  stageTitle = 'Shortlisting Assistant',
  subtitle = 'Ask questions to help filter and prioritize candidates based on their resume.',
  prompts = [],
  nextStageLabel = 'Move to Assessment Stage',
}: {
  stageTitle?: string;
  subtitle?: string;
  prompts?: string[];
  nextStageLabel?: string;
}) {
  const [selected, setSelected] = useState<number[]>([]);
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [loading, setLoading] = useState(true);
  const [updating, setUpdating] = useState(false);

    const params = useParams();
    const jobId = params?.id as string;

  const fetchCandidates = async () => {
    try {
      setLoading(true);
      const res = await fetch(`/api/candidates?stage=applied&job_id=${jobId}`);
      const data = await res.json();
      if (Array.isArray(data)) {
        setCandidates(data);
      } else {
        setCandidates([]);
      }
    } catch (err) {
      console.error('Error fetching candidates:', err);
      toast.error('Failed to fetch candidates');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchCandidates();
  }, []);

  const toggleSelect = (id: number) => {
    setSelected((prev) =>
      prev.includes(id) ? prev.filter((i) => i !== id) : [...prev, id]
    );
  };

  const shortlisted = candidates.filter((c) =>
    selected.includes(c.candidate_id)
  );

  const handleMoveToAssessment = async () => {
    if (shortlisted.length === 0) return;

    setUpdating(true);
    try {
      const emails = shortlisted.map((c) => c.primary_email);

      const res = await fetch('/api/candidates/update-stage', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ emails }),
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(`HTTP ${res.status}: ${text}`);
      }

      await res.json();

      toast.success('Candidates moved to Assessment successfully');
      await fetchCandidates();
      setSelected([]);
    } catch (err) {
      console.error('Error moving candidates to assessment:', err);
      toast.error('Failed to move candidates');
    } finally {
      setUpdating(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#f9fafb] flex flex-col">
      <Navbar showBackButton={true} showActionButton={false} />

      <main className="flex-grow flex pt-20 px-6 gap-6">
        {/* LEFT: Candidates & Actions */}
        <div className="w-1/3 flex flex-col gap-4 sticky top-24 h-[calc(100vh-6rem)] overflow-y-auto">
          <div className="bg-white rounded-xl shadow p-4 overflow-auto max-h-[320px]">
            <h2 className="text-md font-semibold text-gray-800 mb-2">
              Candidate List (sorted by score)
            </h2>
            <div className="space-y-3">
              {loading ? (
                <p className="text-sm text-gray-500">Loading candidates...</p>
              ) : candidates.length === 0 ? (
                <p className="text-sm text-gray-500">No candidates found.</p>
              ) : (
                candidates.map((c) => (
                  <label
                    key={c.candidate_id}
                    className="flex items-start gap-3 p-3 border rounded-lg shadow-sm cursor-pointer hover:bg-gray-50"
                  >
                    <input
                      type="checkbox"
                      checked={selected.includes(c.candidate_id)}
                      onChange={() => toggleSelect(c.candidate_id)}
                      className="mt-1"
                    />
                    <div className="flex flex-col">
                      <span className="font-medium text-gray-800">
                        {c.full_name}
                      </span>
                      <span className="text-sm text-gray-500">
                        {c.primary_email}
                      </span>
                      <span className="text-sm font-semibold text-purple-600">
                        Score: {c.job_score}
                      </span>
                    </div>
                  </label>
                ))
              )}
            </div>
          </div>

          {/* Shortlisted Display */}
          <div className="bg-white rounded-xl shadow p-4">
            <h2 className="text-md font-semibold text-gray-800 mb-2">
              Shortlisted Candidates
            </h2>
            {shortlisted.length === 0 ? (
              <p className="text-sm text-gray-500">No candidates selected.</p>
            ) : (
              <ul className="list-disc list-inside text-sm text-gray-700 max-h-[160px] overflow-auto">
                {shortlisted.map((c) => (
                  <li key={c.candidate_id}>
                    {c.full_name} ({c.primary_email}) – Score: {c.job_score}
                  </li>
                ))}
              </ul>
            )}

            <button
              onClick={handleMoveToAssessment}
              disabled={shortlisted.length === 0 || updating}
              className="mt-4 w-full py-2 rounded-lg text-white bg-gradient-to-r from-purple-500 to-purple-600 hover:opacity-90 transition disabled:opacity-50"
            >
              {updating ? 'Updating...' : nextStageLabel}
            </button>
          </div>
        </div>

        {/* RIGHT: Chatbox */}
        <div className="flex-grow">
          <ChatBox title={stageTitle} subtitle={subtitle} prompts={prompts} />
        </div>
      </main>
    </div>
  );
}
