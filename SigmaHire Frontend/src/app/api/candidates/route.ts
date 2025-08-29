import { NextResponse } from "next/server"
import { createClient } from "../../../../utils/supabase/server"

export async function GET(req: Request) {
  try {
    const { searchParams } = new URL(req.url)
    const stage = searchParams.get('stage')
    const jobId = searchParams.get('job_id')

    const supabase = await createClient()

    let query = supabase
      .from('candidate')
      .select('candidate_id, full_name, primary_email, job_score, stage_of_process, job_id')
      .order('job_score', { ascending: false })

    if (stage) {
      query = query.eq('stage_of_process', stage)
    }

    if (jobId) {
      query = query.eq('job_id', jobId)   // ✅ filter by job
    }

    const { data, error } = await query
    if (error) {
      console.error('Error fetching candidates:', error.message)
      return NextResponse.json({ error: error.message }, { status: 500 })
    }

    return NextResponse.json(data ?? [], { status: 200 })
  } catch (err: any) {
    console.error('Unexpected error:', err.message)
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 })
  }
}
