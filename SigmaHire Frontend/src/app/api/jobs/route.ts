import { NextResponse } from 'next/server';
import { createClient } from '../../../../utils/supabase/server';



export async function POST(req: Request) {
  const supabase = await createClient(); 
  const body = await req.json();

  const {
    jobTitle,
    location,
    jobType,
    salaryRange,
    jobDescription,
    requirements,
  } = body;

  // Optional: get user if needed
const {
  data: { user },
  error: userError,
} = await supabase.auth.getUser();

// console.log('[DEBUG] Supabase user:', user);
// console.log('[DEBUG] Supabase userError:', userError);

//   if (!user) {
//     return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
//   }

  const { error } = await supabase.from('job_postings').insert([
    {
      job_title: jobTitle,
      location,
      job_type: jobType,
      salary_range: salaryRange,
      job_description: jobDescription,
      requirements,
    },
  ]);

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  return NextResponse.json({ success: true }, { status: 201 });
}



export async function GET() {
  const supabase = await createClient();

  const { data: jobs, error } = await supabase.from('job_postings')
    .select('*')
    .order('created_at', { ascending: false });

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  return NextResponse.json(jobs);
}