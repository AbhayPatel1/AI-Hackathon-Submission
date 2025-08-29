'use client';
import { useEffect, useState } from 'react';
import Link from 'next/link';
import Navbar from '../components/Navbar';
import Footer from '../components/Footer';


type Job = {
  id: string;
  job_title: string;
  company_name?: string;
  location?: string;
  salary_range?: string;
  job_type: string;
  job_description?: string;
  requirements?: string;
  created_at?: string;
};

export default function FindJobsPage() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchJobs = async () => {
      try {
        const res = await fetch('/api/jobs');
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Failed to fetch jobs');
        setJobs(data);
      } catch (err: any) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchJobs();
  }, []);

  return (
    <div className="min-h-screen bg-[#f9fafb] flex flex-col">
      <Navbar showBackButton={true} showActionButton={false} />

      <main className="flex-grow pt-24 pb-16 px-6">
        <h1 className="text-3xl font-bold text-gray-800 mb-8">Available Job Openings</h1>

        {loading && <p className="text-gray-500">Loading jobs...</p>}
        {error && <p className="text-red-500">{error}</p>}
        {!loading && jobs.length === 0 && (
          <p className="text-gray-500">No job postings found.</p>
        )}

        <div className="flex flex-col gap-6">
          {jobs.map((job) => (
            <Link
              href={`/find-jobs/${job.id}`}
              key={job.id}
              className="bg-white rounded-xl shadow-sm p-6 hover:shadow-md transition border border-gray-200"
            >
              <div className="flex justify-between items-start mb-2">
                <div>
                  <h2 className="text-xl font-semibold text-gray-800">{job.job_title}</h2>
                  {job.company_name && (
                    <p className="text-sm text-gray-500 mb-1">{job.company_name}</p>
                  )}
                  <p className="text-sm text-gray-600 mb-3 line-clamp-2">
                    {job.job_description || 'No description provided.'}
                  </p>
                </div>

                <span className="text-xs bg-purple-100 text-purple-700 px-2 py-1 rounded-md capitalize whitespace-nowrap">
                  {job.job_type}
                </span>
              </div>

              <div className="flex flex-wrap gap-4 text-sm text-gray-600 mb-3">
                {job.location && (
                  <div className="flex items-center gap-1">
                    📍 <span>{job.location}</span>
                  </div>
                )}
                {job.salary_range && (
                  <div className="flex items-center gap-1">
                    💰 <span>{job.salary_range}</span>
                  </div>
                )}
                {job.created_at && (
                  <div className="flex items-center gap-1">
                    🗓️{' '}
                    <span>
                      Posted: {new Date(job.created_at).toLocaleDateString()}
                    </span>
                  </div>
                )}
              </div>

              {job.requirements && (
                <div className="flex flex-wrap gap-2 text-sm text-gray-700 mb-4">
                  {job.requirements.split(',').map((req, i) => (
                    <span
                      key={i}
                      className="bg-gray-100 text-gray-700 px-2 py-1 rounded-md whitespace-nowrap"
                    >
                      {req.trim()}
                    </span>
                  ))}
                </div>
              )}

              <div className="flex justify-end">
                <button
                  className="px-4 py-2 text-sm bg-gradient-to-tr from-[#7B61FF] to-[#9F80FF] text-white rounded-lg hover:opacity-90 transition"
                  onClick={(e) => {
                    e.preventDefault(); // prevent full page reload
                    window.location.href = `/find-jobs/${job.id}/apply`;
                  }}
                >
                  Apply Now
                </button>
              </div>
            </Link>
          ))}
        </div>
      </main>

      <Footer />
    </div>
  );
}