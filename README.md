# Valorant Skills Elevating

A Streamlit dashboard that looks up a Riot ID (name#tag) via the [HenrikDev VALORANT API](https://docs.henrikdev.xyz/) and turns recent matches into KPIs, rule-based coaching tips, and round-by-round notes.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run valorant_skills_elevating.py
```

No lookup yet? The app falls back to bundled sample data in `sample_data/` so you can see it working immediately.

## Optional: raise your rate limit

Get a free API key from HenrikDev's Discord, then either:

```bash
export HENRIKDEV_API_KEY=your-key-here
```

or add it to `.streamlit/secrets.toml`:

```toml
HENRIKDEV_API_KEY = "your-key-here"
```

## Notes on data sources

- **Riot's official API** does not issue personal keys for VALORANT match history (production-key-only, application required) — that's why this uses HenrikDev instead.
- **tracker.gg** and **vlr.gg** have no official public APIs. vlr.gg specifically only tracks professional esports match data, not personal ranked stats.
- Pro-player recommendations in the app are labeled verified/unverified — pro rosters shift constantly, treat them as a starting point, not a fixed ranking.
