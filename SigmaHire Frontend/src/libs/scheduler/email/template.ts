// src/lib/email/templates.ts

type AvailabilityRequestInput = {
  name: string;
  link: string;
  window_start: string;
  window_end: string;
  duration: number;
};

export function availabilityRequestTemplate(input: AvailabilityRequestInput) {
  return `
    <div style="font-family: Arial, sans-serif; line-height: 1.5;">
      <h2>Hello ${input.name},</h2>
      <p>
        We are scheduling candidate interviews and need your availability.
      </p>
      <p>
        Please provide your free slots between:
        <br/><strong>${new Date(input.window_start).toLocaleString()} 
        and ${new Date(input.window_end).toLocaleString()}</strong>
      </p>
      <p>
        Each interview will be <strong>${input.duration} minutes</strong>.
      </p>
      <p>
        👉 Click below to enter your availability:
      </p>
      <p>
        <a href="${input.link}" 
           style="display:inline-block;padding:10px 20px;
                  background:#4f46e5;color:white;
                  text-decoration:none;border-radius:6px;">
          Provide Availability
        </a>
      </p>
      <p>
        This link will expire once you submit or after 48 hours.
      </p>
      <hr/>
      <p style="font-size: 12px; color: #666;">
        Sent by AI Scheduler · Please do not reply directly.
      </p>
    </div>
  `;
}


export function interviewConfirmTemplate({
  name,
  role,
  time,
  interviewer,
  link,
}: {
  name: string;
  role: string;
  time: string;
  interviewer: string;
  link: string;
}) {
  return `
    <div style="font-family: Arial, sans-serif; line-height: 1.5;">
      <h2>Hello ${name},</h2>
      <p>Your interview has been scheduled.</p>
      <p><strong>Role:</strong> ${role}</p>
      <p><strong>With:</strong> ${interviewer}</p>
      <p><strong>Time:</strong> ${new Date(time).toLocaleString()}</p>
      <p><strong>Meeting Link:</strong> ${link}</p>
      <hr/>
      <p style="font-size: 12px; color: #666;">
        Sent by AI Scheduler · Please do not reply directly.
      </p>
    </div>
  `;
}


type EscalationInput = {
  job_id: string;
  candidate_name?: string;
  reason: string;
  details?: string;
};

export function escalationTemplate(input: EscalationInput) {
  return `
    <div style="font-family: Arial, sans-serif; line-height: 1.5;">
      <h2>⚠️ Scheduling Issue</h2>
      <p>
        The AI Scheduler encountered a problem while booking interviews.
      </p>
      <p>
        <strong>Job:</strong> ${input.job_id}
        ${input.candidate_name ? `<br/><strong>Candidate:</strong> ${input.candidate_name}` : ''}
      </p>
      <p>
        <strong>Reason:</strong> ${input.reason}
      </p>
      ${input.details ? `<p><strong>Details:</strong> ${input.details}</p>` : ''}
      <p>
        Please review this case and take manual action if needed.
      </p>
      <hr/>
      <p style="font-size: 12px; color: #666;">
        Sent by AI Scheduler · Escalation Alert
      </p>
    </div>
  `;
}
