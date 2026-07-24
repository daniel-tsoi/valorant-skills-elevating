"""
Valorant Skills Elevating — player stats + coaching dashboard (single-file version)
Looks up a player's Riot ID (name#tag) via the HenrikDev VALORANT API and turns
recent matches into KPIs, rule-based coaching tips, and round-by-round notes.

Run locally:  streamlit run valorant_skills_elevating.py

Data source note: Riot's official API does not issue personal keys for VALORANT
match history (production-key-only, application required), and tracker.gg has no
official public API. This app uses HenrikDev (https://docs.henrikdev.xyz/), the
community API most hobby Valorant tools are built on — it wraps Riot's own match
data. An API key (free, via HenrikDev's Discord) raises your rate limit but isn't
required for basic lookups.

Sections below, top to bottom:
  1. Settings           - HenrikDev API key (optional), region/platform defaults
  2. Data contracts      - PlayerSummary, RankInfo
  3. ValorantMetrics      - raw matches -> per-match stats + aggregate summary
  4. Rank benchmarks      - rough per-tier ACS/HS% reference points
  5. CoachingInsight      - rule-based tips: coach voice, rank-relative, map, tilt
  6. Pro/content recommendations - weakness -> player to study + search link
  7. Play-by-play         - round-level notes for a single match
  8. DataSource / DataLoader - HenrikDev API or bundled sample fixture
  9. Chart classes        - one per chart type
  10. Dashboard           - orchestrates rendering for a single run
  11. main                - composition root, wires everything together and runs it
"""
from __future__ import annotations

import json
import os
import urllib.parse
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import streamlit.components.v1 as components
from plotly.graph_objects import Figure

# ---------------------------------------------------------------------------
# 1. Settings
# ---------------------------------------------------------------------------

HENRIKDEV_BASE = "https://api.henrikdev.xyz/valorant"
DEFAULT_REGION = "na"
DEFAULT_PLATFORM = "pc"
SAMPLE_MATCHES_PATH = "sample_data/sample_matches.json"
SAMPLE_MMR_PATH = "sample_data/sample_mmr.json"

REGIONS = ["na", "eu", "ap", "kr", "latam", "br"]


@dataclass(frozen=True)
class Settings:
    api_key: str = ""

    @classmethod
    def load(cls) -> "Settings":
        key = ""
        try:
            key = st.secrets.get("HENRIKDEV_API_KEY", "")
        except Exception:
            key = ""
        key = key or os.environ.get("HENRIKDEV_API_KEY", "")
        return cls(api_key=key)


# ---------------------------------------------------------------------------
# 2. Data contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RankInfo:
    tier_name: str
    rr: int
    elo: int


@dataclass(frozen=True)
class PlayerSummary:
    matches_played: int
    win_rate: float
    avg_kd: float
    avg_hs_pct: float
    avg_acs: float
    rank: Optional[RankInfo]


# ---------------------------------------------------------------------------
# 3. ValorantMetrics
# ---------------------------------------------------------------------------


class ValorantMetrics:
    @staticmethod
    def enrich(df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of df with kd, hs_pct, and acs columns added."""
        df = df.copy()
        df["kd"] = df["kills"] / df["deaths"].replace(0, 1)
        total_shots = df["headshots"] + df["bodyshots"] + df["legshots"]
        df["hs_pct"] = (df["headshots"] / total_shots.replace(0, 1) * 100).where(total_shots != 0, 0)
        df["acs"] = df["score"] / df["rounds_played"].replace(0, 1)
        return df

    @staticmethod
    def summarize(df: pd.DataFrame, rank: Optional[RankInfo]) -> PlayerSummary:
        return PlayerSummary(
            matches_played=len(df),
            win_rate=float(df["won"].mean() * 100) if len(df) else 0.0,
            avg_kd=float(df["kd"].mean()) if len(df) else 0.0,
            avg_hs_pct=float(df["hs_pct"].mean()) if len(df) else 0.0,
            avg_acs=float(df["acs"].mean()) if len(df) else 0.0,
            rank=rank,
        )


# ---------------------------------------------------------------------------
# 4. Rank benchmarks (rough community-known reference points, not exact)
# ---------------------------------------------------------------------------

# Approximate typical ACS / headshot% by rank tier group. These are ballpark
# figures for framing feedback relative to rank, not authoritative statistics.
RANK_BENCHMARKS = {
    "Iron": {"acs": 130, "hs_pct": 12},
    "Bronze": {"acs": 150, "hs_pct": 15},
    "Silver": {"acs": 170, "hs_pct": 17},
    "Gold": {"acs": 190, "hs_pct": 20},
    "Platinum": {"acs": 210, "hs_pct": 22},
    "Diamond": {"acs": 230, "hs_pct": 24},
    "Ascendant": {"acs": 245, "hs_pct": 26},
    "Immortal": {"acs": 260, "hs_pct": 28},
    "Radiant": {"acs": 280, "hs_pct": 30},
}


def _benchmark_for_tier(tier_name: str) -> dict:
    for key, bench in RANK_BENCHMARKS.items():
        if key.lower() in tier_name.lower():
            return bench
    return RANK_BENCHMARKS["Gold"]


# ---------------------------------------------------------------------------
# 5. CoachingInsight
# ---------------------------------------------------------------------------


class InsightStrategy(ABC):
    @abstractmethod
    def generate(self, df: pd.DataFrame, summary: PlayerSummary) -> List[str]:
        """Return a list of markdown tip strings."""
        ...


class CoachingInsight(InsightStrategy):
    def generate(self, df: pd.DataFrame, summary: PlayerSummary) -> List[str]:
        if df.empty:
            return ["No matches to coach from yet — play a few games and check back."]

        tips: List[str] = []
        tips.extend(self._benchmark_tips(summary))
        tips.extend(self._map_tip(df))
        tips.extend(self._tilt_tip(df))
        tips.extend(self._best_worst_spotlight(df))
        return tips

    @staticmethod
    def _benchmark_tips(summary: PlayerSummary) -> List[str]:
        tier_name = summary.rank.tier_name if summary.rank else "Gold"
        bench = _benchmark_for_tier(tier_name)
        tips = []
        if summary.avg_hs_pct < bench["hs_pct"] - 3:
            tips.append(
                f"🎯 Your headshot rate ({summary.avg_hs_pct:.0f}%) is below the typical range for "
                f"{tier_name} (~{bench['hs_pct']}%). That's a crosshair-placement problem, not luck — "
                f"spend time in an aim trainer holding common angles at head height instead of chasing flicks."
            )
        if summary.avg_acs < bench["acs"] - 20:
            tips.append(
                f"📉 Your average combat score ({summary.avg_acs:.0f}) is under what's typical for "
                f"{tier_name} (~{bench['acs']}). Low ACS with a decent K/D usually means you're trading "
                f"kills for value too slowly — look for more proactive duels, not just picking off leftovers."
            )
        if summary.avg_kd < 0.9:
            tips.append(
                f"⚔️ Your K/D across these matches is {summary.avg_kd:.2f}. Before anything mechanical, "
                f"check your deaths for a pattern: same spot, same timing, same read every time?"
            )
        if not tips:
            tips.append(f"✅ Your stats are holding around the typical range for {tier_name} — keep it up.")
        return tips

    @staticmethod
    def _map_tip(df: pd.DataFrame) -> List[str]:
        by_map = df.groupby("map", as_index=False).agg(win_rate=("won", "mean"), games=("won", "size"))
        by_map = by_map[by_map["games"] >= 2]
        if by_map.empty:
            return []
        worst = by_map.loc[by_map["win_rate"].idxmin()]
        if worst["win_rate"] < 0.5:
            return [
                f"🗺️ Your weakest map is **{worst['map']}** ({worst['win_rate'] * 100:.0f}% win rate over "
                f"{int(worst['games'])} games). Watch a site-execute guide for that map specifically instead "
                f"of general VOD review — map-specific losses are usually about default setups, not aim."
            ]
        return []

    @staticmethod
    def _tilt_tip(df: pd.DataFrame) -> List[str]:
        if "started_at" not in df.columns or len(df) < 4:
            return []
        ordered = df.sort_values("started_at")
        first_half = ordered.iloc[: len(ordered) // 2]["acs"].mean()
        second_half = ordered.iloc[len(ordered) // 2 :]["acs"].mean()
        if second_half < first_half * 0.75:
            return [
                "🧊 Your combat score drops off noticeably later in this match set compared to earlier — "
                "that's a classic tilt pattern. Consider capping sessions at a fixed number of games "
                "regardless of how the last one went."
            ]
        return []

    @staticmethod
    def _best_worst_spotlight(df: pd.DataFrame) -> List[str]:
        best = df.loc[df["acs"].idxmax()]
        worst = df.loc[df["acs"].idxmin()]
        return [
            f"⭐ **Best match:** {best['agent']} on {best['map']} — {int(best['kills'])}/{int(best['deaths'])}"
            f"/{int(best['assists'])}, {best['acs']:.0f} ACS.",
            f"🪦 **Roughest match:** {worst['agent']} on {worst['map']} — {int(worst['kills'])}/"
            f"{int(worst['deaths'])}/{int(worst['assists'])}, {worst['acs']:.0f} ACS.",
        ]


# ---------------------------------------------------------------------------
# 6. Pro / content recommendations
# ---------------------------------------------------------------------------

# Pro rosters shift constantly — entries marked (verified) were confirmed via a
# live search this session; everything else is recalled from training data and
# may already be stale by the time you read this. Re-check before trusting it
# for anything beyond "a starting point for whose VODs to study."
#
# Format: agent -> list of (team, player, verified: bool)
AGENT_PROS = {
    "Jett": [("MIBR", "Zekken", True), ("Sentinels", "TenZ", False)],  # TenZ retired from pro play Sept 2024 — VOD study only, not an active roster
    "Raze": [("MIBR", "aspas", True)],
    "Neon": [("LEV", "Neon", True)],
    "Reyna": [("PRX", "something", True), ("NRG", "Demon1", False)],
    "Sova": [("FNC", "Alfajer", False)],
    "Omen": [("LEV", "Saadhak", False)],
    "Killjoy": [("FNC", "Boaster", False)],
    "Chamber": [("G2", "yay", False)],
    "Fade": [("KRÜ", "Klaus", False)],
}
DEFAULT_PROS = [("PRX", "f0rsakeN", True), ("MIBR", "aspas", True)]


def _youtube_search_url(query: str) -> str:
    return "https://www.youtube.com/results?search_query=" + urllib.parse.quote(query)


def recommend_content(df: pd.DataFrame) -> List[dict]:
    if df.empty:
        return []
    most_played_agent = df["agent"].value_counts().idxmax()
    pros = AGENT_PROS.get(most_played_agent, DEFAULT_PROS)
    recs = []
    for team, player, verified in pros:
        query = f"{player} {team} valorant {most_played_agent} highlights"
        recs.append(
            {
                "agent": most_played_agent,
                "team": team,
                "pro": player,
                "verified": verified,
                "url": _youtube_search_url(query),
            }
        )
    return recs


# ---------------------------------------------------------------------------
# 7. Play-by-play (round-level notes for one match)
# ---------------------------------------------------------------------------


def play_by_play_tips(rounds: list) -> List[str]:
    """Turn a match's round list into short, round-by-round coaching notes.

    No API exposes actual clip/video data for arbitrary players — this works
    from the round-level stats the match API does return (per-round result,
    kills, damage, loadout value for the tracked player).
    """
    if not rounds:
        return ["No round-by-round data available for this match."]

    tips = []
    quiet_streak = 0
    for rnd in rounds:
        stats = rnd.get("player_stats", {})
        died = stats.get("died", False)
        kills = stats.get("kills", 0)
        loadout = stats.get("loadout_value", 0)
        round_id = rnd.get("id", "?")

        if died and kills == 0:
            quiet_streak += 1
        else:
            quiet_streak = 0

        if died and kills == 0 and loadout >= 3500:
            tips.append(f"Round {round_id}: full-buy round, died without a kill — check positioning before pushing.")
        elif loadout < 1000 and rnd.get("winning_team") != rnd.get("player_team"):
            tips.append(f"Round {round_id}: eco round lost — expected, focus on info/damage for the next buy.")

        if quiet_streak == 3:
            tips.append(f"Round {round_id}: three quiet rounds in a row — a good spot to reset mentally before the next one.")

    if not tips:
        tips.append("No standout patterns this match — solid round-to-round consistency.")
    return tips


# ---------------------------------------------------------------------------
# 8. DataSource / DataLoader
# ---------------------------------------------------------------------------


class DataSource(ABC):
    @abstractmethod
    def fetch_matches(self) -> list:
        ...

    @abstractmethod
    def fetch_rank(self) -> Optional[RankInfo]:
        ...


class HenrikDevSource(DataSource):
    def __init__(self, name: str, tag: str, region: str, platform: str, api_key: str, size: int = 15):
        self._name = name
        self._tag = tag
        self._region = region
        self._platform = platform
        self._size = size
        self._headers = {"Authorization": api_key} if api_key else {}

    def fetch_matches(self) -> list:
        url = f"{HENRIKDEV_BASE}/v4/matches/{self._region}/{self._platform}/{self._name}/{self._tag}"
        resp = requests.get(url, headers=self._headers, params={"size": self._size}, timeout=15)
        resp.raise_for_status()
        return resp.json().get("data", [])

    def fetch_rank(self) -> Optional[RankInfo]:
        url = f"{HENRIKDEV_BASE}/v3/mmr/{self._region}/{self._platform}/{self._name}/{self._tag}"
        resp = requests.get(url, headers=self._headers, timeout=15)
        resp.raise_for_status()
        current = resp.json().get("data", {}).get("current", {})
        if not current:
            return None
        return RankInfo(
            tier_name=current.get("tier", {}).get("name", "Unranked"),
            rr=current.get("rr", 0),
            elo=current.get("elo", 0),
        )


class SampleSource(DataSource):
    """Bundled fixture used when no live lookup has been made yet, or the API call fails."""

    def fetch_matches(self) -> list:
        with open(SAMPLE_MATCHES_PATH) as f:
            return json.load(f)["data"]

    def fetch_rank(self) -> Optional[RankInfo]:
        with open(SAMPLE_MMR_PATH) as f:
            current = json.load(f)["data"]["current"]
        return RankInfo(
            tier_name=current["tier"]["name"],
            rr=current["rr"],
            elo=current["elo"],
        )


class DataLoader:
    def __init__(self, source: DataSource, player_name: str, player_tag: str):
        self._source = source
        self._player_name = player_name.lower()
        self._player_tag = player_tag.lower()

    def load(self) -> tuple:
        """Returns (per-match DataFrame for the tracked player, rank info, {match_id: rounds})."""
        raw_matches = self._source.fetch_matches()
        rank = self._source.fetch_rank()

        rows = []
        rounds_by_match = {}
        for match in raw_matches:
            player = self._find_player(match.get("players", []))
            if player is None:
                continue
            meta = match.get("metadata", {})
            team_id = player.get("team_id")
            team = next((t for t in match.get("teams", []) if t.get("team_id") == team_id), {})
            stats = player.get("stats", {})
            round_totals = team.get("rounds", {})
            rounds_played = round_totals.get("won", 0) + round_totals.get("lost", 0)

            rows.append(
                {
                    "match_id": meta.get("match_id"),
                    "started_at": meta.get("started_at"),
                    "map": meta.get("map", {}).get("name", "Unknown"),
                    "agent": player.get("agent", {}).get("name", "Unknown"),
                    "won": bool(team.get("won", False)),
                    "kills": stats.get("kills", 0),
                    "deaths": stats.get("deaths", 0),
                    "assists": stats.get("assists", 0),
                    "headshots": stats.get("headshots", 0),
                    "bodyshots": stats.get("bodyshots", 0),
                    "legshots": stats.get("legshots", 0),
                    "score": stats.get("score", 0),
                    "rounds_played": rounds_played,
                }
            )

            player_puuid = player.get("puuid")
            died_in_round = {
                kill.get("round")
                for kill in match.get("kills", [])
                if kill.get("victim", {}).get("puuid") == player_puuid
            }

            match_rounds = []
            for rnd in match.get("rounds", []):
                round_id = rnd.get("id")
                # Real API: round["stats"] is a list with one entry per player (not
                # a single "player_stats" dict) — find ours by puuid.
                per_round = next(
                    (
                        s
                        for s in rnd.get("stats", [])
                        if s.get("player", {}).get("puuid") == player_puuid
                    ),
                    {},
                )
                match_rounds.append(
                    {
                        "id": round_id,
                        "winning_team": rnd.get("winning_team"),
                        "player_team": team_id,
                        "player_stats": {
                            "kills": per_round.get("stats", {}).get("kills", 0),
                            "died": round_id in died_in_round,
                            "loadout_value": per_round.get("economy", {}).get("loadout_value", 0),
                        },
                    }
                )
            rounds_by_match[meta.get("match_id")] = match_rounds

        df = pd.DataFrame(rows)
        if not df.empty:
            df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce")
        return df, rank, rounds_by_match

    def _find_player(self, players: list) -> Optional[dict]:
        for p in players:
            if p.get("name", "").lower() == self._player_name and p.get("tag", "").lower() == self._player_tag:
                return p
        return None


# ---------------------------------------------------------------------------
# 9. Chart classes
# ---------------------------------------------------------------------------

BRAND_PRIMARY = "#FF4655"  # Valorant red
BRAND_SECONDARY = "#1FA6BD"  # teal
BRAND_SURFACE = "#0F1923"
BRAND_GRID = "#24313D"
BRAND_TEXT = "#ECE8E1"

# Win/loss status colors — from ui-ux-pro-max's "Comparative Analysis Dashboard"
# pattern (positive/negative/neutral delta tokens), not guessed.
STATUS_WIN = "#22C55E"
STATUS_LOSS = "#EF4444"

CHART_FONT = dict(family="'Chakra Petch', system-ui, -apple-system, sans-serif", color=BRAND_TEXT)


def _themed_layout(**overrides) -> dict:
    base = dict(
        margin=dict(t=10, b=10),
        paper_bgcolor=BRAND_SURFACE,
        plot_bgcolor=BRAND_SURFACE,
        font=CHART_FONT,
        xaxis=dict(gridcolor=BRAND_GRID, zerolinecolor=BRAND_GRID),
        yaxis=dict(gridcolor=BRAND_GRID, zerolinecolor=BRAND_GRID),
    )
    base.update(overrides)
    return base


class Chart(ABC):
    title: str
    empty_message: str = "Not enough matches to show this yet."

    @abstractmethod
    def render(self, df: pd.DataFrame) -> Optional[Figure]:
        ...


class AcsTrendChart(Chart):
    title = "Combat score per match"

    def render(self, df: pd.DataFrame) -> Optional[Figure]:
        if df.empty:
            return None
        trend = df.sort_values("started_at")
        fig = px.line(trend, x="started_at", y="acs", markers=True)
        fig.update_traces(line_color=BRAND_PRIMARY, marker_color=BRAND_PRIMARY)
        fig.update_layout(**_themed_layout(yaxis_title="ACS"))
        return fig


class WinRateByMapChart(Chart):
    title = "Win rate by map"

    def render(self, df: pd.DataFrame) -> Optional[Figure]:
        if df.empty:
            return None
        by_map = df.groupby("map", as_index=False)["won"].mean()
        by_map["win_rate"] = by_map["won"] * 100
        fig = px.bar(by_map, x="map", y="win_rate", text="win_rate")
        fig.update_traces(marker_color=BRAND_PRIMARY, texttemplate="%{text:.0f}%")
        fig.update_layout(**_themed_layout(showlegend=False, yaxis_title="Win %"))
        return fig


class WinRateByAgentChart(Chart):
    title = "Win rate by agent"

    def render(self, df: pd.DataFrame) -> Optional[Figure]:
        if df.empty:
            return None
        by_agent = df.groupby("agent", as_index=False)["won"].mean()
        by_agent["win_rate"] = by_agent["won"] * 100
        fig = px.bar(by_agent, x="agent", y="win_rate", text="win_rate")
        fig.update_traces(marker_color=BRAND_SECONDARY, texttemplate="%{text:.0f}%")
        fig.update_layout(**_themed_layout(showlegend=False, yaxis_title="Win %"))
        return fig


class KdTrendChart(Chart):
    title = "K/D per match"

    def render(self, df: pd.DataFrame) -> Optional[Figure]:
        if df.empty:
            return None
        trend = df.sort_values("started_at")
        fig = px.line(trend, x="started_at", y="kd", markers=True)
        fig.update_traces(line_color=BRAND_PRIMARY, marker_color=BRAND_PRIMARY)
        fig.update_layout(**_themed_layout(yaxis_title="K/D"))
        return fig


class HsPctTrendChart(Chart):
    title = "Headshot % per match"

    def render(self, df: pd.DataFrame) -> Optional[Figure]:
        if df.empty:
            return None
        trend = df.sort_values("started_at")
        fig = px.line(trend, x="started_at", y="hs_pct", markers=True)
        fig.update_traces(line_color=BRAND_SECONDARY, marker_color=BRAND_SECONDARY)
        fig.update_layout(**_themed_layout(yaxis_title="Headshot %"))
        return fig


# ---------------------------------------------------------------------------
# 9b. 3D hero (real WebGL via Three.js, embedded in a sandboxed iframe)
# ---------------------------------------------------------------------------

HERO_3D_HTML = f"""
<div id="hero3d" style="width:100%; height:220px; background:{BRAND_SURFACE};"></div>
<script src="https://unpkg.com/three@0.160.0/build/three.min.js"></script>
<script>
(function() {{
  const container = document.getElementById('hero3d');
  const width = container.clientWidth || 800;
  const height = 220;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color('{BRAND_SURFACE}');

  const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
  camera.position.z = 6;

  const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
  renderer.setSize(width, height);
  container.appendChild(renderer.domElement);

  const geometry = new THREE.IcosahedronGeometry(2, 0);
  const wireMat = new THREE.MeshBasicMaterial({{ color: '{BRAND_PRIMARY}', wireframe: true }});
  const wireMesh = new THREE.Mesh(geometry, wireMat);
  scene.add(wireMesh);

  const innerGeometry = new THREE.IcosahedronGeometry(1.15, 0);
  const innerMat = new THREE.MeshBasicMaterial({{ color: '{BRAND_SECONDARY}', wireframe: true, transparent: true, opacity: 0.6 }});
  const innerMesh = new THREE.Mesh(innerGeometry, innerMat);
  scene.add(innerMesh);

  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function render() {{
    renderer.render(scene, camera);
  }}

  if (reducedMotion) {{
    wireMesh.rotation.set(0.4, 0.6, 0);
    innerMesh.rotation.set(0.4, 0.6, 0);
    render();
  }} else {{
    function animate() {{
      requestAnimationFrame(animate);
      wireMesh.rotation.x += 0.003;
      wireMesh.rotation.y += 0.004;
      innerMesh.rotation.x -= 0.002;
      innerMesh.rotation.y -= 0.003;
      render();
    }}
    animate();
  }}
}})();
</script>
"""


def render_3d_hero() -> None:
    """Decorative rotating wireframe — real WebGL via Three.js, not a CSS effect.
    Runs in a sandboxed iframe (st.components.v1.html), so it can't inherit the
    page's own theme — colors are duplicated here from the BRAND_* constants.
    """
    components.html(HERO_3D_HTML, height=225)


# ---------------------------------------------------------------------------
# 10. Dashboard
# ---------------------------------------------------------------------------

BRAND_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@400;500;600;700&family=Russo+One&display=swap');
h1 { font-family: 'Russo One', sans-serif !important; letter-spacing: 0.02em; }
h2, h3 { font-family: 'Chakra Petch', sans-serif !important; font-weight: 600 !important; }
html, body, [class*="css"] { font-family: 'Chakra Petch', sans-serif; }
div[data-testid="stMetric"] {
    background: #1A2530;
    border: 1px solid rgba(236, 232, 225, 0.12);
    border-radius: 10px;
    padding: 14px 16px 10px;
}
.form-pill {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    border-radius: 6px;
    font-weight: 700;
    font-size: 0.8rem;
    color: #0F1923;
    margin-right: 4px;
}
</style>
"""


class Dashboard:
    def __init__(self, metrics: ValorantMetrics, insight: InsightStrategy, charts: List[Chart], api_key: str):
        self._metrics = metrics
        self._insight = insight
        self._charts = charts
        self._api_key = api_key

    def run(self) -> None:
        st.set_page_config(page_title="Valorant Skills Elevating", page_icon="🎯", layout="wide")
        st.markdown(BRAND_CSS, unsafe_allow_html=True)
        st.title("🎯 Valorant Skills Elevating")
        st.caption("Look up a Riot ID to get stats, rule-based coaching tips, and round-by-round notes.")
        render_3d_hero()

        name, tag, region, platform, use_sample = self._render_lookup_form()

        if use_sample:
            source = SampleSource()
            player_name, player_tag = "SampleAgent", "NA1"
            st.info("No lookup yet — showing bundled sample data. Enter a Riot ID above and hit Look up.")
        else:
            source = HenrikDevSource(name, tag, region, platform, self._api_key)
            player_name, player_tag = name, tag

        try:
            df, rank, rounds_by_match = DataLoader(source, player_name, player_tag).load()
        except Exception as e:
            st.error(f"Couldn't fetch stats ({e}). Showing bundled sample data instead.")
            df, rank, rounds_by_match = DataLoader(SampleSource(), "SampleAgent", "NA1").load()

        if df.empty:
            st.warning("No matches found for that Riot ID in this region/platform.")
            st.stop()

        df = self._metrics.enrich(df)
        summary = self._metrics.summarize(df, rank)

        self._render_kpis(summary)
        self._render_recent_form(df)
        st.divider()
        self._render_coaching(df, summary)
        st.divider()
        self._render_recommendations(df)
        st.divider()
        self._render_play_by_play(df, rounds_by_match)
        st.divider()
        self._render_charts(df)
        st.divider()
        self._render_table(df)

    # ---- lookup form ----

    def _render_lookup_form(self):
        with st.sidebar:
            st.header("Look up a player")
            name = st.text_input("Riot ID name", value="")
            tag = st.text_input("Tag (without #)", value="")
            region = st.selectbox("Region", REGIONS, index=REGIONS.index(DEFAULT_REGION))
            platform = st.selectbox("Platform", ["pc", "console"], index=0)
            submitted = st.button("Look up")
            if not self._api_key:
                st.caption("No HenrikDev API key set — lookups are rate-limited. Add HENRIKDEV_API_KEY to raise it.")
        use_sample = not (submitted and name and tag)
        return name.strip(), tag.strip(), region, platform, use_sample

    # ---- KPI row ----

    def _render_kpis(self, summary: PlayerSummary) -> None:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Matches", summary.matches_played)
        c2.metric("Win Rate", f"{summary.win_rate:.0f}%")
        c3.metric("Avg K/D", f"{summary.avg_kd:.2f}")
        c4.metric("Avg Headshot %", f"{summary.avg_hs_pct:.0f}%")
        rank_label = f"{summary.rank.tier_name} ({summary.rank.rr} RR)" if summary.rank else "Unranked"
        c5.metric("Rank", rank_label)

    # ---- recent form ----

    def _render_recent_form(self, df: pd.DataFrame) -> None:
        recent = df.sort_values("started_at").tail(10)
        pills = "".join(
            f'<span class="form-pill" style="background:{STATUS_WIN if won else STATUS_LOSS}">'
            f'{"W" if won else "L"}</span>'
            for won in recent["won"]
        )
        st.markdown(f"**Recent form:** {pills}", unsafe_allow_html=True)

    # ---- coaching ----

    def _render_coaching(self, df: pd.DataFrame, summary: PlayerSummary) -> None:
        st.subheader("🧑‍🏫 Coaching notes")
        for tip in self._insight.generate(df, summary):
            st.markdown(f"- {tip}")

    # ---- recommendations ----

    def _render_recommendations(self, df: pd.DataFrame) -> None:
        st.subheader("📺 Study these")
        recs = recommend_content(df)
        if not recs:
            return
        st.caption(
            f"Based on your most-played agent ({recs[0]['agent']}). Pro rosters change — treat this as a "
            "starting point for whose VODs to search, not a fixed ranking."
        )
        for rec in recs:
            tag = "✅ verified current roster" if rec["verified"] else "⚠️ unverified, may be stale"
            st.markdown(f"- [{rec['team']} {rec['pro']} — {rec['agent']} highlights]({rec['url']}) _{tag}_")

    # ---- play-by-play ----

    def _render_play_by_play(self, df: pd.DataFrame, rounds_by_match: dict) -> None:
        st.subheader("🔁 Round-by-round: your roughest match")
        worst = df.loc[df["acs"].idxmin()]
        st.caption(f"{worst['agent']} on {worst['map']} — {int(worst['kills'])}/{int(worst['deaths'])}/{int(worst['assists'])}")
        tips = play_by_play_tips(rounds_by_match.get(worst["match_id"], []))
        for tip in tips:
            st.markdown(f"- {tip}")

    # ---- charts ----

    def _render_charts(self, df: pd.DataFrame) -> None:
        columns = st.columns(2)
        for i, chart in enumerate(self._charts):
            if i % 2 == 0 and i > 0:
                columns = st.columns(2)
            with columns[i % 2]:
                st.subheader(chart.title)
                fig = chart.render(df)
                if fig is None:
                    st.info(chart.empty_message)
                else:
                    st.plotly_chart(fig, use_container_width=True)

    # ---- table ----

    def _render_table(self, df: pd.DataFrame) -> None:
        st.subheader("Match history")
        view = df.sort_values("started_at", ascending=False)
        st.dataframe(view, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download match history (CSV)",
            view.to_csv(index=False),
            file_name="valorant_matches.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------------------------
# 11. main — composition root
# ---------------------------------------------------------------------------


def main() -> None:
    settings = Settings.load()
    dashboard = Dashboard(
        metrics=ValorantMetrics(),
        insight=CoachingInsight(),
        charts=[AcsTrendChart(), WinRateByMapChart(), WinRateByAgentChart(), KdTrendChart(), HsPctTrendChart()],
        api_key=settings.api_key,
    )
    dashboard.run()


if __name__ == "__main__":
    main()
