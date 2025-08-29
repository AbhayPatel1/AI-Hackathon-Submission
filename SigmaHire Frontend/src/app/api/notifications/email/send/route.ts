// src/app/api/notifications/email/send/route.ts
import { sendEmail } from '@/libs/scheduler/email/send';
import { NextResponse } from 'next/server';


type EmailRequest = {
  to: string;
  subject: string;
  html: string;
};

export async function POST(req: Request) {
  try {
    const body: EmailRequest = await req.json();

    if (!body.to || !body.subject || !body.html) {
      return NextResponse.json(
        { error: 'to, subject, and html are required' },
        { status: 400 }
      );
    }

    await sendEmail(body);

    return NextResponse.json(
      { message: 'Email sent successfully' },
      { status: 200 }
    );
  } catch (err: any) {
    console.error('Error in email send API:', err.message);
    return NextResponse.json(
      { error: 'Internal Server Error' },
      { status: 500 }
    );
  }
}
