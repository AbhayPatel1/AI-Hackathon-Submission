// src/lib/parsing/quickBook.ts


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
  location_url?: string;
};

// Type for chatbot free text parsing result
export type QuickBookPayload = BookInterviewInput;

/**
 * MVP parser: expects either structured JSON or a simple string format.
 * Later: plug in LLM/NLP for free-text parsing.
 */
export function parseQuickBookInput(input: any): QuickBookPayload {
  // Case 1: Already structured JSON payload
  if (
    input &&
    input.candidate_id &&
    input.candidate_email &&
    input.candidate_name &&
    input.interviewer_id &&
    input.interviewer_email &&
    input.interviewer_name &&
    input.job_id &&
    input.start_ts &&
    input.end_ts &&
    input.title
  ) {
    return {
      candidate_id: input.candidate_id,
      candidate_email: input.candidate_email,
      candidate_name: input.candidate_name,
      interviewer_id: input.interviewer_id,
      interviewer_email: input.interviewer_email,
      interviewer_name: input.interviewer_name,
      job_id: input.job_id,
      start_ts: input.start_ts,
      end_ts: input.end_ts,
      title: input.title,
      location_url: input.location_url ?? undefined,
    };
  }

  // Case 2: Extend here for free-text parsing later
  throw new Error('Invalid quick-book input. Expected structured JSON.');
}
