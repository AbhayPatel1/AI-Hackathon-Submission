import { NextResponse } from "next/server";
import { createClient } from "../../../../utils/supabase/server";

export async function POST(req: Request) {
  const supabase = await createClient();
  const body = await req.json();

  console.log('[APPLY] Request received with body:', body);

  const {
    job_id,
    full_name,
    email,
    phone,
    cover_letter,
    resume_url,
  } = body;

  if (!job_id || !full_name || !email || !resume_url) {
    console.error('[APPLY] Missing required fields');
    return NextResponse.json({ error: 'Missing required fields' }, { status: 400 });
  }

  const { error } = await supabase.from('job_applications').insert([
    {
      job_id,
      full_name,
      email,
      phone,
      cover_letter,
      resume_url,
    },
  ]);

  if (error) {
    console.error('[DB ERROR]', error.message);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  console.log('[APPLY] Insert success');
  return NextResponse.json({ success: true });
}