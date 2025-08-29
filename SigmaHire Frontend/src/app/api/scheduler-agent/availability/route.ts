// src/app/api/availability/route.ts
import { NextResponse } from 'next/server';


type AvailabilitySlot = {
  id: string;
  interviewer_id: string;
  job_id: string;
  start_ts: string;
  end_ts: string;
  status: 'free' | 'booked' | 'tentative';
  collected_at: string;
};

export async function GET(req: Request) {
  try {
    const { searchParams } = new URL(req.url);
    const jobId = searchParams.get('job_id');
    const interviewerId = searchParams.get('interviewer_id'); // optional

    if (!jobId) {
      return NextResponse.json(
        { error: 'job_id is required' },
        { status: 400 }
      );
    }

    const supabase = await createClient();

    let query = supabase
      .from('availability')
      .select(
        'id, interviewer_id, job_id, start_ts, end_ts, status, collected_at'
      )
      .eq('job_id', jobId)
      .order('start_ts', { ascending: true });

    if (interviewerId) {
      query = query.eq('interviewer_id', interviewerId);
    }

    const { data, error } = await query;

    if (error) {
      console.error('Error fetching availability:', error.message);
      return NextResponse.json({ error: error.message }, { status: 500 });
    }

    return NextResponse.json(data ?? [], { status: 200 });
  } catch (err: any) {
    console.error('Unexpected error in availability API:', err.message);
    return NextResponse.json(
      { error: 'Internal Server Error' },
      { status: 500 }
    );
  }
}
