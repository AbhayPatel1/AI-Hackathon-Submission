'use client';
import Footer from '@/app/components/Footer';
import Navbar from '@/app/components/Navbar';
import { useState } from 'react';
import { useRouter } from 'next/navigation';


const jobTypes = ['Full-time', 'Internship', 'Part-time', 'Contract'];

export default function CreateJobPage() {
  const router = useRouter();
  const [form, setForm] = useState({
    jobTitle: '',
    location: '',
    jobType: '',
    salaryRange: '',
    jobDescription: '',
    requirements: '',
  });

  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>
  ) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e: React.FormEvent) => {
  e.preventDefault();

  if (!form.jobTitle || !form.location || !form.jobType || !form.jobDescription) {
    setError('Please fill out all required fields.');
    return;
  }

  setError(null);
  setSuccess(false);

try {
  const res = await fetch('/api/jobs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(form),
  });

  const data = await res.json();

  if (!res.ok) throw new Error(data.error || 'Something went wrong');

  router.push('/hr-dashboard');
  
  // Handle success...
} catch (err: any) {
  console.error('Failed to submit job:', err.message || err);
  setError(err.message || 'Unexpected error');
}

};

  return (
    <>
      <Navbar showBackButton={true} actionLabel="" actionHref="" />

      <main className="bg-white text-gray-900 min-h-screen pt-24 pb-20 px-6">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-2xl font-bold mb-2">Create New Job Posting</h1>
          <p className="text-sm text-gray-500 mb-8">
            Fill out the details below to create a new job posting.
          </p>

          <form onSubmit={handleSubmit} className="space-y-6">
            {/* 🧱 First Row — Job Title, Location, Salary */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div>
                <label className="block font-medium text-gray-700 mb-1">Job Title *</label>
                <input
                  name="jobTitle"
                  value={form.jobTitle}
                  onChange={handleChange}
                  className="w-full border border-gray-300 rounded-lg px-4 py-2"
                  required
                />
              </div>

              <div>
                <label className="block font-medium text-gray-700 mb-1">Location *</label>
                <input
                  name="location"
                  value={form.location}
                  onChange={handleChange}
                  className="w-full border border-gray-300 rounded-lg px-4 py-2"
                  required
                />
              </div>

              <div>
                <label className="block font-medium text-gray-700 mb-1">Salary Range</label>
                <input
                  name="salaryRange"
                  value={form.salaryRange}
                  onChange={handleChange}
                  className="w-full border border-gray-300 rounded-lg px-4 py-2"
                  placeholder="e.g. $100k/year or $25/hr"
                />
              </div>
            </div>

            {/* 🧱 Second Row — Job Type */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block font-medium text-gray-700 mb-1">Job Type *</label>
                <select
                  name="jobType"
                  value={form.jobType}
                  onChange={handleChange}
                  className="w-full border border-gray-300 rounded-lg px-4 py-2"
                  required
                >
                  <option value="">Select</option>
                  {jobTypes.map((type) => (
                    <option key={type} value={type}>
                      {type}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* 🧱 Job Description */}
            <div>
              <label className="block font-medium text-gray-700 mb-1">Job Description *</label>
              <textarea
                name="jobDescription"
                value={form.jobDescription}
                onChange={handleChange}
                className="w-full border border-gray-300 rounded-lg px-4 py-2 min-h-[120px]"
                required
              />
            </div>

            {/* 🧱 Requirements */}
            <div>
              <label className="block font-medium text-gray-700 mb-1">Requirements</label>
              <textarea
                name="requirements"
                value={form.requirements}
                onChange={handleChange}
                className="w-full border border-gray-300 rounded-lg px-4 py-2 min-h-[80px]"
                placeholder="e.g. React, TypeScript, 5+ years experience"
              />
            </div>

            {/* 🧱 Error / Success */}
            {error && <p className="text-red-500 text-sm">{error}</p>}
            {success && (
              <p className="text-green-600 text-sm">
                Job created successfully! You can now return to the dashboard.
              </p>
            )}

            {/* 🧱 Submit */}
            <button
              type="submit"
              className="px-6 py-3 bg-[#7B61FF] text-white rounded-lg hover:bg-[#654fe0] font-medium"
            >
              Submit Job
            </button>
          </form>
        </div>
      </main>

      <Footer/>
    </>
  );
}