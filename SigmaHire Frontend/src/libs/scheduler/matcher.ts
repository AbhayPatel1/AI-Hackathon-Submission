// src/lib/scheduler/matcher.ts

import { withinWindow, makeSlot } from './utils';
import { bookOne } from './booker';
import { createClient } from '../../../utils/supabase/server';

type Candidate = {
  candidate_id: string;
  full_name: string;
  primary_email: string;
  job_id: string;
  stage_of_process: string;
};

type Interviewer = {
  interviewer_id: string;
  full_name: string;
  email: string;
  specialization: string;
  is_active: boolean;
};

type Availability = {
  id: string;
  interviewer_id: string;
  start_ts: string;
  end_ts: string;
  status: string;
};

type MatchResult = {
  candidate_id: string;
  interviewer_id?: string;
  slot?: { start_ts: string; end_ts: string };
  status: 'scheduled' | 'failed';
  reason?: string;
};

export async function runMatcher(
  jobId: string,
  window_start: string,
  window_end: string,
  duration_min: number,
  candidateIds: string[]
): Promise<MatchResult[]> {
  const supabase = await createClient();

  // 1. Fetch only selected candidates
  const { data: candidates, error: candErr } = await supabase
    .from('candidate')
    .select('candidate_id, full_name, primary_email, job_id, stage_of_process')
    .in('candidate_id', candidateIds);

  if (candErr) throw new Error(`Candidate fetch failed: ${candErr.message}`);

  // 2. Fetch all active interviewers (global, no job filter)
  const { data: interviewers, error: intErr } = await supabase
    .from('interviewer')
    .select('interviewer_id, full_name, email, specialization, is_active')
    .eq('is_active', true);

  if (intErr) throw new Error(`Interviewer fetch failed: ${intErr.message}`);

  // 3. Fetch availability slots (global, not per job)
  const { data: slots, error: slotErr } = await supabase
    .from('availability')
    .select('id, interviewer_id, start_ts, end_ts, status')
    .eq('status', 'free')
    .order('start_ts', { ascending: true });

  if (slotErr) throw new Error(`Availability fetch failed: ${slotErr.message}`);

  const results: MatchResult[] = [];

  for (const c of candidates ?? []) {
    let chosenInterviewer: Interviewer | undefined;
    let chosenSlot: Availability | undefined;

    for (const iv of interviewers ?? []) {
      const freeSlots = (slots ?? []).filter(
        (s) =>
          s.interviewer_id === iv.interviewer_id &&
          withinWindow(
            { start: s.start_ts, end: s.end_ts },
            { start: window_start, end: window_end }
          )
      );

      if (freeSlots.length === 0) continue;

      // Sort slots by earliest
      freeSlots.sort(
        (a, b) =>
          new Date(a.start_ts).getTime() - new Date(b.start_ts).getTime()
      );

      const candidateSlot = makeSlot(freeSlots[0].start_ts, duration_min);

      if (!chosenSlot) {
        chosenInterviewer = iv;
        chosenSlot = {
          ...candidateSlot,
          id: freeSlots[0].id,
          interviewer_id: iv.interviewer_id,
          status: 'booked',
        };
      }
    }

    if (chosenInterviewer && chosenSlot) {
      // Book interview
      await bookOne({
        candidate_id: c.candidate_id,
        candidate_email: c.primary_email,
        candidate_name: c.full_name,
        interviewer_id: chosenInterviewer.interviewer_id,
        interviewer_email: chosenInterviewer.email,
        interviewer_name: chosenInterviewer.full_name,
        job_id: jobId,
        start_ts: chosenSlot.start_ts,
        end_ts: chosenSlot.end_ts,
        title: 'Interview',
      });

      // Mark slot as booked
      await supabase
        .from('availability')
        .update({ status: 'booked' })
        .eq('id', chosenSlot.id);

      // Move candidate to interview stage
      await supabase
        .from('candidate')
        .update({ stage_of_process: 'interview' })
        .eq('candidate_id', c.candidate_id);

      results.push({
        candidate_id: c.candidate_id,
        interviewer_id: chosenInterviewer.interviewer_id,
        slot: { start_ts: chosenSlot.start_ts, end_ts: chosenSlot.end_ts },
        status: 'scheduled',
      });
    } else {
      results.push({
        candidate_id: c.candidate_id,
        status: 'failed',
        reason: 'No available slots',
      });
    }
  }

  return results;
}
