// app/api/candidates/update-stage/route.ts
import { NextResponse } from 'next/server'
import { createClient } from '../../../../../utils/supabase/server'
import nodemailer from 'nodemailer'

export async function POST(req: Request) {
  try {
    const { emails } = await req.json()

    if (!Array.isArray(emails) || emails.length === 0) {
      return NextResponse.json({ error: 'Emails must be a non-empty array' }, { status: 400 })
    }

    const supabase = await createClient()

    // Update stage_of_process
    const { data, error } = await supabase
      .from('candidate')
      .update({ stage_of_process: 'assessment' })
      .in('primary_email', emails)
      .eq('stage_of_process', 'applied')
      .select('candidate_id, full_name, primary_email, stage_of_process')

    if (error) {
      console.error('Error updating candidates:', error.message)
      return NextResponse.json({ error: error.message }, { status: 500 })
    }

    // ✅ Send emails
    const transporter = nodemailer.createTransport({
      service: 'gmail',
      auth: {
        user: process.env.GMAIL_USER,       // e.g. "yourbot@gmail.com"
        pass: process.env.GMAIL_APP_PASS,   // app password
      },
    })

    for (const cand of data || []) {
      try {
        await transporter.sendMail({
  from: `"Project Bot" <${process.env.GMAIL_USER}>`,
  to: cand.primary_email,
  subject: '🎉 Congratulations! You’ve been shortlisted for the Assessment Stage',
  text: `Hi ${cand.full_name},

We’re excited to inform you that you have been shortlisted and moved to the Assessment Stage of the recruitment process.

More details regarding the assessment will be communicated to you shortly. Please keep an eye on your inbox.

Best of luck!
— Hiring Team
`,
  html: `
    <div style="font-family: Arial, sans-serif; color: #333; line-height: 1.6; max-width: 600px; margin: auto; border: 1px solid #eee; border-radius: 8px; overflow: hidden;">
      <div style="background: linear-gradient(90deg, #7c3aed, #6d28d9); color: white; padding: 16px;">
        <h2 style="margin: 0;">Congratulations 🎉</h2>
      </div>
      <div style="padding: 20px;">
        <p>Hi <strong>${cand.full_name}</strong>,</p>
        <p>We’re excited to inform you that you have been <strong>shortlisted</strong> and moved to the 
        <span style="color: #6d28d9; font-weight: bold;">Assessment Stage</span> of the recruitment process.</p>

        <p><em>More details regarding the assessment will be communicated to you shortly. 
        Please keep an eye on your inbox.</em></p>

        <p>Best of luck!</p>
        <p style="margin-top: 24px; font-weight: bold; color: #6d28d9;">— Hiring Team</p>
      </div>
      <div style="background: #f3f4f6; padding: 12px; text-align: center; font-size: 12px; color: #6b7280;">
        This is an automated email. Please do not reply.
      </div>
    </div>
  `,
})

      } catch (mailErr) {
        console.error(`Error sending email to ${cand.primary_email}:`, mailErr)
      }
    }

    return NextResponse.json(
      { message: 'Stage updated + emails sent', updated: data },
      { status: 200 }
    )
  } catch (err: any) {
    console.error('Unexpected error:', err.message)
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 })
  }
}
