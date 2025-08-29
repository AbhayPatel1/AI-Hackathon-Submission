"use client";
import Hero from "./components/Hero";
import FeatureCard from "./components/FeatureCard";
import Navbar from "./components/Navbar";
import Footer from "./components/Footer";
import { useEffect, useState } from "react";
import Loader from "./components/Loader";

export default function Home() {
   const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Prevent scroll
    document.body.style.overflow = 'hidden';
    document.documentElement.style.overflow = 'hidden';

    // Fake delay just for demo
    const timer = setTimeout(() => {
      setLoading(false);
      document.body.style.overflow = '';
      document.documentElement.style.overflow = '';
    }, 1200);

    return () => {
      clearTimeout(timer);
      document.body.style.overflow = '';
      document.documentElement.style.overflow = '';
    };
  }, []);

  return (
    <>
     {loading && <Loader />}

    <div className="flex flex-col justify-between w-full h-screen overflow-hidden">
      <Navbar
        showBackButton={false}
        showActionButton={true}
        actionLabel="Careers"
        actionHref="https://www.sigmoid.com/careers"
      />
       <main className="flex-grow pt-24 pb-16 overflow-hidden">
      <Hero />
      <section className="py-16 px-6 flex flex-wrap gap-6 justify-center bg-[#f9fafb]">
        <FeatureCard
          title="Job Management"
          color="blue-600"
          desc="Create and manage job postings with ease."
          bullets={[
            "Create detailed job descriptions",
            "Track application status",
            "Manage hiring pipeline",
          ]}
        />
        <FeatureCard
          title="AI Scoring"
          color="purple-600"
          desc="Automatically score candidates based on resume analysis and job requirements."
          bullets={["Resume analysis", "Skill matching", "Ranking algorithms"]}
        />
        <FeatureCard
          title="Smart Support"
          color="green-600"
          desc="RAG-based chatbot to answer candidate questions and provide support."
          bullets={["24/7 availability", "Instant responses", "Knowledge base integration"]}
        />
      </section>
</main>
 <Footer />
    </div>
    </>
  );
}