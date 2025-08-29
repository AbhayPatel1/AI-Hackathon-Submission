// src/lib/scheduler/utils.ts

export type TimeRange = {
  start: string; // ISO string (UTC ideally)
  end: string;   // ISO string
};


export function overlaps(a: TimeRange, b: TimeRange): boolean {
  return new Date(a.start) < new Date(b.end) &&
         new Date(a.end) > new Date(b.start);
}

export function addMinutes(iso: string, minutes: number): string {
  const d = new Date(iso);
  d.setMinutes(d.getMinutes() + minutes);
  return d.toISOString();
}

export function withinWindow(slot: TimeRange, window: TimeRange): boolean {
  return new Date(slot.start) >= new Date(window.start) &&
         new Date(slot.end) <= new Date(window.end);
}


export function makeSlot(startISO: string, durationMin: number): TimeRange {
  return {
    start: startISO,
    end: addMinutes(startISO, durationMin),
  };
}
