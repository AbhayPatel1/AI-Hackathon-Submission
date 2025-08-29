import Link from "next/link";

export default function Hero() {
  return (
    <section className="text-center py-20 px-6 bg-gradient-to-b from-white to-[#f5f5f7]">
      <h1 className="text-4xl md:text-5xl font-bold text-[#1F1F1F]">
        Welcome to <span className="text-[#7B61FF]">SigmaHire</span>
      </h1>
      <p className="mt-4 text-lg text-gray-600 max-w-2xl mx-auto">
        Travel the upward curve towards a great data analytics career with SigmaHire. Explore opportunities, connect with top companies, and take your career to new heights.
      </p>
      <div className="mt-8 flex flex-col sm:flex-row justify-center gap-4">
        <Link href="/hr-dashboard">
        <button className="px-6 py-3 bg-[#4D53F0] text-white rounded-xl font-medium shadow">
          Navigate as HR
        </button>
        </Link>
        <Link href="/find-jobs">
        <button className="px-6 py-3 border border-gray-300 rounded-xl font-medium text-gray-800 bg-white hover:bg-gray-100">
          Find Jobs
        </button>
        </Link>
      </div>
    </section>
  );
}