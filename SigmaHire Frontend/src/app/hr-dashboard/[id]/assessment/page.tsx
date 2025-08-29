'use client';

import { useEffect, useState } from 'react';
import Navbar from '@/app/components/Navbar';
import { useParams } from 'next/navigation';

type Candidate = {
  candidate_id: number;
  full_name: string;
  primary_email: string;
  job_score: number;
  stage_of_process: string; // e.g. "assessment"
};

export default function AssessmentPage() {
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [selected, setSelected] = useState<number[]>([]);
  const [loading, setLoading] = useState(true);
  const params = useParams();
  const jobId = params?.id as string;

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await fetch(`/api/candidates?stage=assessment&job_id=${jobId}`);
        const data = await res.json();
        setCandidates(data || []);
      } catch (err) {
        console.error('Error fetching candidates', err);
      } finally {
        setLoading(false);
      }
    };
     if (jobId) fetchData();
  }, [jobId]);

  const toggleSelect = (id: number) => {
    setSelected((prev) =>
      prev.includes(id) ? prev.filter((i) => i !== id) : [...prev, id]
    );
  };

  const handleMoveNext = async () => {
    try {
      await fetch('/api/candidates/move-stage', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ ids: selected, nextStage: 'interview' }),
      });
      alert('Selected candidates moved to Interview stage!');
      setSelected([]);
      // Refresh candidates
      const res = await fetch('/api/candidates?stage=assessment');
      const data = await res.json();
      setCandidates(data || []);
    } catch (err) {
      console.error('Error moving candidates:', err);
    }
  };

  return (
    <div className="min-h-screen bg-[#f9fafb] flex flex-col">
      <Navbar showBackButton={true} showActionButton={false} />

      <main className="flex-grow pt-20 px-6">
        <div className="bg-white rounded-xl shadow p-6">
          <h2 className="text-lg font-semibold text-gray-800 mb-4">
            Assessment Stage Candidates
          </h2>

          <table className="w-full border border-gray-200 rounded-lg">
            <thead>
              <tr className="bg-gray-100 text-left">
               <th className="p-3 text-gray-800">Select</th>
                <th className="p-3 text-gray-800">Full Name</th>
                <th className="p-3 text-gray-800">Email</th>
                <th className="p-3 text-gray-800">Score</th>
                <th className="p-3 text-gray-800">Stage</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr key="loading">
                  <td colSpan={5} className="p-4 text-center text-gray-500">
                    Loading candidates...
                  </td>
                </tr>
              ) : candidates.length === 0 ? (
                <tr key="no-candidates">
                  <td colSpan={5} className="p-4 text-center text-gray-500">
                    No candidates found in this stage.
                  </td>
                </tr>
              ) : (
                candidates.map((c) => (
                  <tr key={c.candidate_id} className="border-t">
                    <td className="p-3">
                      <input
                        type="checkbox"
                        checked={selected.includes(c.candidate_id)}
                        onChange={() => toggleSelect(c.candidate_id)}
                      />
                    </td>
                    <td className="p-3 font-medium text-gray-800">
                      {c.full_name}
                    </td>
                    <td className="p-3 text-gray-600">{c.primary_email}</td>
                    <td className="p-3 font-semibold text-purple-600">
                      {c.job_score}
                    </td>
                    <td className="p-3 text-gray-500">{c.stage_of_process}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>

          <button
            onClick={handleMoveNext}
            disabled={selected.length === 0}
            className="mt-6 w-full py-2 rounded-lg text-white bg-gradient-to-r from-green-500 to-green-600 hover:opacity-90 transition"
          >
            Move to Interview Stage
          </button>
        </div>
      </main>
    </div>
  );
}
