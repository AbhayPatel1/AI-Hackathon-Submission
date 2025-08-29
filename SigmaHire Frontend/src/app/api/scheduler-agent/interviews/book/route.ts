// src/app/api/interviews/book/route.ts
import { bookOne } from '@/libs/scheduler/booker';
import { NextResponse } from 'next/server';


type BookInterviewPayload = {
  candidate_id: string;
  candidate_email: string;
  candidate_name: string;
  interviewer_id: string;
  interviewer_email: string;
  interviewer_name: string;
  job_id: string;
  start_ts: string;  // ISO
  end_ts: string;    // ISO
  title: string;
  location_url?: string;
};

export async function POST(req: Request) {
  try {
    const body: BookInterviewPayload = await req.json();

    // Quick validation
    if (
      !body.candidate_id ||
      !body.candidate_email ||
      !body.interviewer_id ||
      !body.interviewer_email ||
      !body.job_id ||
      !body.start_ts ||
      !body.end_ts ||
      !body.title
    ) {
      return NextResponse.json(
        { error: 'Missing required fields' },
        { status: 400 }
      );
    }

    const interview = await bookOne(body);

    return NextResponse.json(
      { message: 'Interview booked successfully', interview },
      { status: 200 }
    );
  } catch (err: any) {
    console.error('Error booking interview via API:', err.message);
    return NextResponse.json(
      { error: 'Internal Server Error' },
      { status: 500 }
    );
  }
}
