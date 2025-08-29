import { createClient } from "../../../utils/supabase/server";
import { sendEmail } from "./email/send";
import { interviewConfirmTemplate } from "./email/template";


type BookInterviewInput = {
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
  location_url?: string; // optional meeting link
};

export async function bookOne(input: BookInterviewInput) {
  const supabase = await createClient();

  // 1. Insert into interviews table
  const { data: interview, error } = await supabase
    .from('interview')
    .insert({
      candidate_id: input.candidate_id,
      interviewer_id: input.interviewer_id,
      job_id: input.job_id,
      start_ts: input.start_ts,
      end_ts: input.end_ts,
      title: input.title,
      location_url: input.location_url ?? null,
      status: 'scheduled',
    })
    .select()
    .single();

  if (error) {
    console.error('Error booking interview:', error.message);
    throw new Error(error.message);
  }

  // 2. Insert notification log
  await supabase.from('notification').insert([
    {
      kind: 'email',
      to_email: input.candidate_email,
      subject: `Interview Scheduled: ${input.title}`,
      payload_json: input,
      status: 'pending',
      created_at: new Date().toISOString(),
    },
    {
      kind: 'email',
      to_email: input.interviewer_email,
      subject: `Interview Scheduled: ${input.title}`,
      payload_json: input,
      status: 'pending',
      created_at: new Date().toISOString(),
    },
  ]);

  // 3. Send emails
  const candidateHtml = interviewConfirmTemplate({
    name: input.candidate_name,
    role: input.title,
    time: input.start_ts,
    interviewer: input.interviewer_name,
    link: input.location_url ?? 'To be shared',
  });

  const interviewerHtml = interviewConfirmTemplate({
    name: input.interviewer_name,
    role: input.title,
    time: input.start_ts,
    interviewer: input.candidate_name, // flips perspective
    link: input.location_url ?? 'To be shared',
  });

  await sendEmail({
    to: input.candidate_email,
    subject: `Interview Scheduled: ${input.title}`,
    html: candidateHtml,
  });

  await sendEmail({
    to: input.interviewer_email,
    subject: `Interview Scheduled: ${input.title}`,
    html: interviewerHtml,
  });

  // 4. Mark notifications as sent
  await supabase
    .from('notification')
    .update({ status: 'sent' })
    .in('to_email', [input.candidate_email, input.interviewer_email])
    .eq('subject', `Interview Scheduled: ${input.title}`);

  return interview;
}
