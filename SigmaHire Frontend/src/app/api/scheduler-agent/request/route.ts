// src/app/api/availability/request/route.ts
import { NextResponse } from 'next/server';
import { randomUUID } from 'crypto';
import { sendEmail } from '@/lib/email/send';
import { availabilityRequestTemplate } from '@/lib/email/templates';
import { createClient } from '../../../../../utils/supabase/server';

type RequestPayload = {
  job_id: string;
  window_start: string; // ISO
  window_end: string;   // ISO
  duration_min: number;
};

export async function POST(req: Request) {
  try {
    const body: RequestPayload = await req.json();

    if (!body.job_id || !body.window_start || !body.window_end || !body.duration_min) {
      return NextResponse.json(
        { error: 'job_id, window_start, window_end, duration_min are required' },
        { status: 400 }
      );
    }

    const supabase = await createClient();

    // Insert a schedule_request row
    const { data: scheduleReq, error: reqErr } = await supabase
      .from('schedule_request')
      .insert({
        job_id: body.job_id,
        window_start: body.window_start,
        window_end: body.window_end,
        duration_min: body.duration_min,
        status: 'collecting',
        created_by: 'system',
      })
      .select()
      .single();

    if (reqErr) {
      console.error('Error creating schedule_request:', reqErr.message);
      return NextResponse.json({ error: reqErr.message }, { status: 500 });
    }

    // Fetch interviewers for this job
    const { data: interviewers, error: intErr } = await supabase
      .from('interviewer')
      .select('interviewer_id, full_name, email, specialization')
      .eq('job_id', body.job_id)
      .eq('is_active', true);

    if (intErr) {
      console.error('Error fetching interviewers:', intErr.message);
      return NextResponse.json({ error: intErr.message }, { status: 500 });
    }

    // For each interviewer, generate unique token + URL
    for (const iv of interviewers ?? []) {
      const token = randomUUID();
      const expiresAt = new Date();
      expiresAt.setHours(expiresAt.getHours() + 48); // 48h expiry

      await supabase.from('availability').insert({
        id: token,
        interviewer_id: iv.interviewer_id,
        job_id: body.job_id,
        start_ts: body.window_start,
        end_ts: body.window_end,
        status: 'tentative',
        collected_at: new Date().toISOString(),
      });

      const link = `${process.env.NEXT_PUBLIC_APP_URL}/availability/fill?token=${token}`;

      // Send email with link
      await sendEmail({
        to: iv.email,
        subject: `Provide your interview availability (${body.window_start} - ${body.window_end})`,
        html: availabilityRequestTemplate({
          name: iv.full_name,
          link,
          window_start: body.window_start,
          window_end: body.window_end,
          duration: body.duration_min,
        }),
      });
    }

    return NextResponse.json(
      { message: 'Availability request sent to all interviewers', request: scheduleReq },
      { status: 200 }
    );
  } catch (err: any) {
    console.error('Unexpected error in availability request API:', err.message);
    return NextResponse.json(
      { error: 'Internal Server Error' },
      { status: 500 }
    );
  }
}
