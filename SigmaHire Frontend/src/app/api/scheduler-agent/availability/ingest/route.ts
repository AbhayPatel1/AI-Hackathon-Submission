// src/app/api/availability/ingest/route.ts
import { NextResponse } from 'next/server';
import { createClient } from '../../../../../../utils/supabase/server';


type SlotInput = {
  start_ts: string; // ISO
  end_ts: string;   // ISO
};

type RequestPayload = {
  token: string;       // unique token from availability/request
  slots: SlotInput[];  // slots interviewer marked as available
};

export async function POST(req: Request) {
  try {
    const body: RequestPayload = await req.json();

    if (!body.token || !body.slots || body.slots.length === 0) {
      return NextResponse.json(
        { error: 'token and at least one slot are required' },
        { status: 400 }
      );
    }

    const supabase = await createClient();

    // Validate token → map to interviewer & job
    const { data: tokenRow, error: tokenErr } = await supabase
      .from('availability')
      .select('id, interviewer_id, job_id, start_ts, end_ts, status')
      .eq('id', body.token)
      .single();

    if (tokenErr || !tokenRow) {
      return NextResponse.json({ error: 'Invalid or expired token' }, { status: 400 });
    }

    if (tokenRow.status !== 'tentative') {
      return NextResponse.json({ error: 'This link has already been used' }, { status: 400 });
    }

    // Insert submitted slots
    const slotRows = body.slots.map((s) => ({
      interviewer_id: tokenRow.interviewer_id,
      job_id: tokenRow.job_id,
      start_ts: s.start_ts,
      end_ts: s.end_ts,
      status: 'free',
      collected_at: new Date().toISOString(),
    }));

    const { error: insertErr } = await supabase.from('availability').insert(slotRows);
    if (insertErr) {
      console.error('Error inserting slots:', insertErr.message);
      return NextResponse.json({ error: insertErr.message }, { status: 500 });
    }

    // Expire token row
    await supabase
      .from('availability')
      .update({ status: 'used' })
      .eq('id', body.token);

    return NextResponse.json(
      { message: 'Availability submitted successfully', slots: slotRows },
      { status: 200 }
    );
  } catch (err: any) {
    console.error('Unexpected error in availability ingest:', err.message);
    return NextResponse.json(
      { error: 'Internal Server Error' },
      { status: 500 }
    );
  }
}
