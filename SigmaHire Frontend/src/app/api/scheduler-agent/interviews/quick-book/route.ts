// src/app/api/interviews/quick-book/route.ts
import { bookOne } from '@/libs/scheduler/booker';
import { parseQuickBookInput } from '@/libs/scheduler/parsing/quickBook';
import { NextResponse } from 'next/server';


type QuickBookRequest = any; // For MVP we accept any, parseQuickBookInput validates

export async function POST(req: Request) {
  try {
    const body: QuickBookRequest = await req.json();

    // Parse recruiter/chatbot input into BookInterviewInput
    const payload = parseQuickBookInput(body);

    // Book the interview
    const interview = await bookOne(payload);

    return NextResponse.json(
      { message: 'Interview booked successfully via quick-book', interview },
      { status: 200 }
    );
  } catch (err: any) {
    console.error('Error in quick-book API:', err.message);
    return NextResponse.json(
      { error: err.message || 'Internal Server Error' },
      { status: 500 }
    );
  }
}
