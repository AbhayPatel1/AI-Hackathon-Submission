'use client';

import { useEffect, useState } from "react";
import Footer from "../components/Footer";
import Navbar from "../components/Navbar";
import DashboardStats from "./Components/DashboardStats";

import JobList from "./Components/JobList";


export default function HRDashboardPage() {
  // Mock data for now
  
  const [jobs, setJobs] = useState<any[]>([]);
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
    <div className="bg-[#f9fafb] min-h-screen">
      <Navbar
        showBackButton={true}
        showActionButton={true}
        actionLabel="Create Job"
        actionHref="/hr-dashboard/create-job"
      />
      <main className="pt-24 pb-16 px-6">
        <h1 className="text-2xl font-bold mb-6 text-gray-800">HR Dashboard</h1>

        <DashboardStats
          activeJobs={jobs.length}
          totalApplications={0} // ← You can update later
          pendingReviews={0}    // ← You can update later
        />

        {loading && <p className="text-gray-500 mt-4">Loading jobs...</p>}
        {error && <p className="text-red-500 mt-4">Error: {error}</p>}
        {!loading && jobs.length === 0 && (
          <p className="text-gray-600 mt-4">No job postings found.</p>
        )}
        {!loading && jobs.length > 0 && <JobList jobs={jobs} />}
      </main>
      <Footer />
    </div>
  );
}