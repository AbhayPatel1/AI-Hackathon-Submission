// src/app/api/auto-scheduler/run/route.ts
import { runMatcher } from '@/libs/scheduler/matcher';
import { NextResponse } from 'next/server';

type AutoSchedulerPayload = {
  job_id: string;
  window_start: string;   // ISO
  window_end: string;     // ISO
  duration_min: number;
  candidateIds: string[]; // recruiter-selected candidates
};

export async function POST(req: Request) {
  try {
    const body: AutoSchedulerPayload = await req.json();

    // Basic validation
    if (!body.job_id) {
      return NextResponse.json({ error: 'job_id is required' }, { status: 400 });
    }
    if (!body.window_start || !body.window_end) {
      return NextResponse.json({ error: 'window_start and window_end are required' }, { status: 400 });
    }
    if (!body.duration_min || body.duration_min <= 0) {
      return NextResponse.json({ error: 'duration_min must be greater than 0' }, { status: 400 });
    }
    if (!body.candidateIds || body.candidateIds.length === 0) {
      return NextResponse.json({ error: 'candidateIds[] must include at least one candidate' }, { status: 400 });
    }

    // Run matcher
    const results = await runMatcher(
      body.job_id,
      body.window_start,
      body.window_end,
      body.duration_min,
      body.candidateIds
    );

    return NextResponse.json(
      { message: 'Auto-scheduler completed', results },
      { status: 200 }
    );
  } catch (err: any) {
    console.error('Error in auto-scheduler run:', err);
    return NextResponse.json(
      { error: 'Internal Server Error', details: err.message },
      { status: 500 }
    );
  }
}
