// app/hr-dashboard/components/JobList.tsx

import Link from "next/link";

type Job = {
  title: string;
  applications: number;
  id: string;
  pendingReviews: number;
};

export default function JobList({ jobs }: { jobs: Job[] }) {

  return (
    <section>
      <h2 className="text-xl font-semibold mb-4 text-gray-800">Active Job Postings</h2>
      <p className="text-sm text-gray-500 mb-6">
        Manage your current job openings and view candidate applications.
      </p>
      <div className="space-y-4">
        {jobs.map((job, index) => (
          <div key={index} className="bg-white shadow-sm rounded-xl p-4 flex items-center justify-between">
            <div>
              <h3 className="font-medium text-gray-900">{job.job_title}</h3>
              <p className="text-sm text-gray-500">{job.job_applications}</p>
            </div>
            <div className="flex items-center gap-4">
              <div className="text-sm text-gray-600">
                <span className="font-semibold">{job.pendingReviews}</span> pending review{job.pendingReviews !== 1 && 's'}
              </div>
              <Link href={`/hr-dashboard/${job.id}`}>
              <button className="px-4 py-2 text-gray-800 bg-gray-100  text-sm rounded-lg font-medium hover:bg-gray-200">
                Open Pipeline
              </button>
                </Link>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}