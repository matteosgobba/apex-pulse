import { readCalendar } from "@/lib/racing/calendar-service";

export const dynamic = "force-dynamic";
export const maxDuration = 30;

export async function GET() {
  return Response.json(await readCalendar(), { headers: { "Cache-Control": "no-store" } });
}
