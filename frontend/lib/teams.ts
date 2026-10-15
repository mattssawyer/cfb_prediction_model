import teams from "@/data/teams.json";

export type TeamInfo = {
  abbr: string | null;
  color: string | null;
  alt: string | null;
  logo: string | null;
  mascot: string | null;
};

const TEAMS = teams as Record<string, TeamInfo>;

export function team(name: string): TeamInfo {
  return TEAMS[name] ?? { abbr: null, color: null, alt: null, logo: null, mascot: null };
}
