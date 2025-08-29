"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Navbar from "@/app/components/Navbar";
import ChatbotSection from "./Components/ChatbotSection";
import { ChevronRight, ArrowLeft, ArrowUp, ArrowDown } from "lucide-react";

const STAGES = [
  {
    id: "shortlisting",
    title: "Shortlisting",
    description: "Initial filtering of resumes.",
  },
  {
    id: "assessment",
    title: "Assessment",
    description: "Coding test / screening.",
  },
  {
    id: "interview",
    title: "Interview",
    description: "Final round of interviews.",
  },
  {
    id: "selected",
    title: "Selected",
    description: "Final chosen candidates.",
  },
] as const;

export default function OpenPipelinePage() {
  const router = useRouter();
  const params = useParams();
  const jobId = params?.id as string | undefined;

  const [activeIdx, setActiveIdx] = useState<number>(0);

  const navigateToStage = (stageId: string) => {
    if (!jobId) return;
    router.push(`/hr-dashboard/${jobId}/${stageId}`);
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "ArrowUp") {
        e.preventDefault();
        setActiveIdx((i) => (i - 1 + STAGES.length) % STAGES.length);
      } else if (e.key === "ArrowDown") {
        e.preventDefault();
        setActiveIdx((i) => (i + 1) % STAGES.length);
      } else if (e.key === "Enter") {
        const target = STAGES[activeIdx];
        if (target) navigateToStage(target.id);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [activeIdx]);

  const active = STAGES[activeIdx];

  return (
    <div className="h-screen flex flex-col bg-[#f9fafb]">
      <Navbar showBackButton={true} showActionButton={false} />

      {/* Header */}
      <header className="pt-24 px-6">
        <div className="max-w-6xl mx-auto">
          <div className="flex items-center gap-2 text-sm text-slate-500">
            <ArrowLeft className="h-4 w-4" />
            <span className="truncate">HR Dashboard</span>
            <ChevronRight className="h-4 w-4" />
            <span className="font-medium text-slate-700">Pipeline</span>
            {jobId && (
              <>
                <ChevronRight className="h-4 w-4" />
                <span className="rounded-full px-2 py-0.5 bg-slate-100 text-slate-700">Job #{jobId}</span>
              </>
            )}
          </div>
          <div className="mt-3">
            <h1 className="text-2xl font-semibold text-slate-900 tracking-tight">Hiring Pipeline</h1>
            <p className="text-slate-600 text-sm mt-1">Navigate stages and jump in quickly. Use ↑/↓ and Enter to move fast.</p>
          </div>
        </div>
      </header>

      {/* Main */}
      <main className="flex-1 min-h-0 px-6 pb-6">
        <div className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-6 h-full">
          {/* Left: Stages list */}
         <aside className="lg:col-span-6 xl:col-span-5 min-h-0">
            <div className="rounded-2xl border border-slate-200 bg-white shadow-sm p-4">
              <div className="flex items-center justify-between mb-2">
                <h2 className="text-sm font-semibold text-slate-800">Stages</h2>
                <span className="text-xs text-slate-500">{STAGES.length} total</span>
              </div>
              <div className="space-y-3 overflow-auto max-h-[60vh] pr-1">
                {STAGES.map((stage, idx) => {
                  const isActive = idx === activeIdx;
                  return (
                    <button
                      key={stage.id}
                      onClick={() => setActiveIdx(idx)}
                      className={
                        "w-full text-left group relative overflow-hidden rounded-xl border transition focus:outline-none " +
                        (isActive
                          ? "border-indigo-300 bg-indigo-50 shadow-sm"
                          : "border-slate-200 bg-white hover:shadow-sm")
                      }
                    >
                      <div className="p-4 flex items-start gap-3">
                        <div className="flex-1">
                          <div className="flex items-center justify-between">
                            <h3 className={"font-medium " + (isActive ? "text-indigo-900" : "text-slate-900")}>{stage.title}</h3>
                            <ChevronRight className={"h-4 w-4 transition " + (isActive ? "text-indigo-600" : "text-slate-400 group-hover:text-slate-500")} />
                          </div>
                          <p className="text-sm text-slate-600 mt-0.5">{stage.description}</p>
                        </div>
                      </div>
                    </button>
                  );
                })}
              </div>
              <div className="mt-3 flex items-center justify-between text-xs text-slate-500">
                <div className="flex items-center gap-2">
                  <ArrowUp className="h-3.5 w-3.5" />
                  <ArrowDown className="h-3.5 w-3.5" />
                  <span>Navigate</span>
                </div>
                <div className="flex items-center gap-2">
                  <kbd className="rounded border px-1.5 py-0.5 bg-slate-50">Enter</kbd>
                  <span>Open stage</span>
                </div>
              </div>
            </div>
          </aside>

          {/* Right: Active stage preview + Chatbot */}
<section className="lg:col-span-6 xl:col-span-7 min-h-0 flex flex-col gap-6">
            {/* Active stage summary */}
            <div className="rounded-2xl border border-slate-200 bg-white shadow-sm p-6 h-fit">
              <div className="flex items-start gap-4">
                <div className="flex-1">
                  <h3 className="text-lg font-semibold text-slate-900">{active?.title}</h3>
                  <p className="text-slate-600 mt-1">{active?.description}</p>
                  <div className="mt-4 flex flex-wrap gap-3">
                    <button
                      onClick={() => active && navigateToStage(active.id)}
                      className="inline-flex items-center gap-2 rounded-lg bg-indigo-600 px-3.5 py-2 text-sm font-medium text-white shadow-sm transition hover:bg-indigo-700 focus:outline-none"
                    >
                      Go to {active?.title}
                      <ChevronRight className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              </div>
            </div>

            {/* Copilot / Chatbot Section */}
            {/* <div className="rounded-2xl border border-slate-200 bg-white shadow-sm p-0 overflow-hidden min-h-[40vh]">
              <ChatbotSection />
            </div> */}
          </section>
        </div>
      </main>
    </div>
  );
}
