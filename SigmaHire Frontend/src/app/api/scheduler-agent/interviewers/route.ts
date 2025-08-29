// src/app/api/interviewers/route.ts
import { NextResponse } from 'next/server';
import { createClient } from '../../../../../utils/supabase/server';

type Interviewer = {
  interviewer_id: string;
  full_name: string;
  email: string;
  specialization: string;
  job_id: string;
  is_active: boolean;
};

export async function GET(req: Request) {
  try {
    const { searchParams } = new URL(req.url);
    const jobId = searchParams.get('job_id');

    if (!jobId) {
      return NextResponse.json(
        { error: 'job_id is required' },
        { status: 400 }
      );
    }

    const supabase = await createClient();

    const { data, error } = await supabase
      .from('interviewer')
      .select('interviewer_id, full_name, email, specialization, job_id, is_active')
      .eq('is_active', true)
      .order('full_name', { ascending: true });

    if (error) {
      console.error('Error fetching interviewers:', error.message);
      return NextResponse.json({ error: error.message }, { status: 500 });
    }

    return NextResponse.json(data ?? [], { status: 200 });
  } catch (err: any) {
    console.error('Unexpected error in interviewers API:', err.message);
    return NextResponse.json(
      { error: 'Internal Server Error' },
      { status: 500 }
    );
  }
}
