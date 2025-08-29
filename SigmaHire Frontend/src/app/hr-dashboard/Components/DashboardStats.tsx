// components/DashboardStats.tsx
type StatCardProps = {
  title: string;
  count: number;
  subtitle: string;
  color?: string;
};

function StatCard({ title, count, subtitle, color = "text-blue-600" }: StatCardProps) {
  return (
    <div className="bg-white shadow rounded-xl p-6 w-full sm:w-72">
      <h3 className={`text-lg font-semibold mb-2 ${color}`}>{title}</h3>
      <p className="text-3xl font-bold text-gray-800">{count}</p>
      <p className="text-sm text-gray-500">{subtitle}</p>
    </div>
  );
}

type DashboardStatsProps = {
  activeJobs: number;
  totalApplications: number;
  pendingReviews: number;
};

export default function DashboardStats({
  activeJobs,
  totalApplications,
  pendingReviews,
}: DashboardStatsProps) {
  return (
    <section className="py-10 px-6 flex flex-wrap gap-6 justify-start">
      <StatCard title="Active Jobs" count={activeJobs} subtitle="Total job postings" color="text-blue-600" />
      <StatCard title="Applications" count={totalApplications} subtitle="Total applications" color="text-purple-600" />
      <StatCard title="Pending Reviews" count={pendingReviews} subtitle="Awaiting screening" color="text-green-600" />
    </section>
  );
}