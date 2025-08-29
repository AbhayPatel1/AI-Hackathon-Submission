'use client';

import { useEffect, useState } from 'react';
import { useParams } from 'next/navigation';
import Navbar from '@/app/components/Navbar';

type Candidate = {
  candidate_id: string;
  full_name: string;
  primary_email: string;
  job_score: number;
  stage_of_process: string;
};

type Result = {
  candidate_id: string;
  interviewer_id?: string;
  slot?: { start_ts: string; end_ts: string };
  status: 'scheduled' | 'failed';
  reason?: string;
};

export default function SchedulingPage() {
  const { id: jobId } = useParams<{ id: string }>();

  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [windowStart, setWindowStart] = useState('');
  const [windowEnd, setWindowEnd] = useState('');
  const [duration, setDuration] = useState(60); // minutes
  const [results, setResults] = useState<Result[]>([]);
  const [loading, setLoading] = useState(false);

  // Fetch candidates in "assessment" stage
  useEffect(() => {
    const fetchCandidates = async () => {
      try {
        const res = await fetch(`/api/candidates?stage=assessment&job_id=${jobId}`);
        const data = await res.json();
        if (Array.isArray(data)) {
          setCandidates(data);
        }
      } catch (err) {
        console.error('Error fetching candidates:', err);
      }
    };
    if (jobId) fetchCandidates();
  }, [jobId]);

  const toggleSelect = (id: string) => {
    setSelected((prev) =>
      prev.includes(id) ? prev.filter((i) => i !== id) : [...prev, id]
    );
  };

  const handleSchedule = async () => {
    try {
      setLoading(true);
      const res = await fetch('/api/scheduler-agent/auto-scheduler/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          job_id: jobId,
          window_start: windowStart,
          window_end: windowEnd,
          duration_min: duration,
          candidateIds: selected,
        }),
      });
      const data = await res.json();
      setResults(data.results || []);
    } catch (err) {
      console.error('Error running scheduler:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#f9fafb] flex flex-col">
      <Navbar showBackButton={true} showActionButton={false} />

      <main className="flex-grow pt-20 px-6 space-y-6">
        <h2 className="text-lg font-semibold text-gray-800">
          Schedule Interviews
        </h2>

        {/* Candidate selection */}
        <div className="bg-white rounded-xl shadow p-4">
          <h3 className="font-medium text-gray-800 mb-2">
            Select Candidates (Assessment Stage)
          </h3>
          {candidates.length === 0 ? (
            <p className="text-sm text-gray-500">No candidates found.</p>
          ) : (
            <div className="space-y-2 max-h-60 overflow-auto">
              {candidates.map((c) => (
                <label
                  key={c.candidate_id}
                  className="flex items-center gap-3 p-2 border rounded cursor-pointer hover:bg-gray-50"
                >
                  <input
                    type="checkbox"
                    checked={selected.includes(c.candidate_id)}
                    onChange={() => toggleSelect(c.candidate_id)}
                  />
                  <div>
                    <p className="font-medium text-gray-800">{c.full_name}</p>
                    <p className="text-sm text-gray-500">{c.primary_email}</p>
                    <p className="text-sm font-semibold text-purple-600">
                      Score: {c.job_score}
                    </p>
                  </div>
                </label>
              ))}
            </div>
          )}
        </div>

        {/* Date range & duration */}
        <div className="bg-white rounded-xl shadow p-4 space-y-4">
          <h3 className="font-medium text-gray-800 mb-2">Scheduling Window</h3>
          <div className="flex gap-4">
            <div>
              <label className="block text-sm text-gray-700">Start Date</label>
              <input
                type="datetime-local"
                value={windowStart}
                onChange={(e) => setWindowStart(e.target.value)}
                className="border p-2 rounded w-full"
              />
            </div>
            <div>
              <label className="block text-sm text-gray-700">End Date</label>
              <input
                type="datetime-local"
                value={windowEnd}
                onChange={(e) => setWindowEnd(e.target.value)}
                className="border p-2 rounded w-full"
              />
            </div>
            <div>
              <label className="block text-sm text-gray-700">Duration (minutes)</label>
              <input
                type="number"
                value={duration}
                onChange={(e) => setDuration(Number(e.target.value))}
                className="border p-2 rounded w-28"
              />
            </div>
          </div>
        </div>

        {/* Action */}
        <button
          onClick={handleSchedule}
          disabled={selected.length === 0 || !windowStart || !windowEnd || loading}
          className="px-6 py-2 rounded-lg bg-gradient-to-r from-green-500 to-green-600 text-white font-medium hover:opacity-90 transition"
        >
          {loading ? 'Scheduling...' : 'Run Auto-Scheduler'}
        </button>

        {/* Results */}
        {results.length > 0 && (
          <div className="bg-white rounded-xl shadow p-4">
            <h3 className="font-medium text-gray-800 mb-2">Results</h3>
            <table className="w-full text-sm border border-gray-200 rounded">
              <thead className="bg-gray-100 text-gray-800">
                <tr>
                  <th className="p-2 text-left">Candidate</th>
                  <th className="p-2 text-left">Status</th>
                  <th className="p-2 text-left">Interviewer</th>
                  <th className="p-2 text-left">Slot</th>
                  <th className="p-2 text-left">Reason</th>
                </tr>
              </thead>
              <tbody>
                {results.map((r) => (
                  <tr key={r.candidate_id} className="border-t">
                    <td className="p-2">{r.candidate_id}</td>
                    <td className="p-2">{r.status}</td>
                    <td className="p-2">{r.interviewer_id || '-'}</td>
                    <td className="p-2">
                      {r.slot
                        ? `${new Date(r.slot.start_ts).toLocaleString()}`
                        : '-'}
                    </td>
                    <td className="p-2 text-red-500">{r.reason || '-'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </main>
    </div>
  );
}
