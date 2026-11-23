"""Fan attitude: the tone of the past week's college football coverage of each team (local papers, beat writers, team fan sites),
from Google News headlines, scored 0-100% (share of positive vs negative headlines).

  python fan_sentiment.py history 2025      # every week of a past season (window: 7 days before each week's first game)
  python fan_sentiment.py week 2026 2-13 2026-11-23 333,2,228   # one week for listed team ids (used by cfb_weekly.py)

Results go to sentiment/<season>.csv: season, week (e.g. 2-13), team_id, school, n, pos, neg, attitude, tone, start, end.
Game results ("X beats Y", "Y falls to X") name both teams, so win/loss words are neutral here: the score reads the mood
of the coverage (confidence, concern, injuries, turmoil), not the result, which the model already has.
"""
import csv, os, re, sys, time, html, datetime as dt, urllib.parse, urllib.request, xml.etree.ElementTree as ET

LEX = {  # adjustments to VADER's general lexicon for football coverage
    'win': 0, 'wins': 0, 'won': 0, 'winning': 0, 'beat': 0, 'beats': 0, 'victory': 0, 'defeat': 0, 'defeats': 0, 'defeated': 0,
    'loss': 0, 'losses': 0, 'lose': 0, 'loses': 0, 'lost': 0, 'losing': 0, 'upset': 0, 'upsets': 0, 'falls': 0, 'fall': 0,
    'rout': 0, 'routs': 0, 'blowout': 0, 'crush': 0, 'crushes': 0, 'kill': 0, 'killer': 0, 'attack': 0, 'defense': 0, 'offense': 0,
    'fired': -2.6, 'firing': -2.4, 'hot-seat': -2, 'injury': -1.8, 'injured': -1.8, 'injuries': -1.8, 'out': -0.3,
    'suspended': -2.2, 'suspension': -2.2, 'arrested': -3, 'decommit': -1.8, 'decommits': -1.8, 'benched': -1.8,
    'concerns': -1.4, 'concern': -1.2, 'worry': -1.5, 'worries': -1.5, 'skid': -1.8, 'slump': -1.8, 'collapse': -2.4,
    'turmoil': -2.4, 'chaos': -1.8, 'embarrassing': -2.6, 'disappointing': -2.2, 'frustrating': -2, 'mistakes': -1.4,
    'surging': 2, 'rolling': 1.2, 'momentum': 1.6, 'confident': 2, 'optimistic': 2, 'dominant': 2.2, 'dominance': 2,
    'undefeated': 1.8, 'unbeaten': 1.8, 'breakout': 1.8, 'emerging': 1.2, 'healthy': 1.6, 'returns': 0.8, 'commit': 0.8,
    'commits': 0.8, 'extension': 1.4, 'rising': 1.4, 'climbs': 1.2, 'bowl-eligible': 1.8, 'clinch': 1.8, 'clinches': 1.8,
}
UA = {'User-Agent': 'Mozilla/5.0 (compatible; cfb26-fan-attitude/1.0; +https://github.com/tynuss/cfb26)'}

def _an():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    a = SentimentIntensityAnalyzer(); a.lexicon.update(LEX); return a

def headlines(query, start, end, tries=3):
    """Google News RSS headlines for query, published start <= date < end (YYYY-MM-DD)."""
    q = f'{query} after:{start} before:{end}'
    url = 'https://news.google.com/rss/search?' + urllib.parse.urlencode({'q': q, 'hl': 'en-US', 'gl': 'US', 'ceid': 'US:en'})
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                root = ET.fromstring(r.read())
            out, seen = [], set()
            for it in root.iter('item'):
                t = html.unescape(it.findtext('title') or '')
                src = it.findtext('source') or ''
                if src and t.endswith(' - ' + src): t = t[: -len(src) - 3]
                key = re.sub(r'\W+', ' ', t.lower()).strip()
                if key and key not in seen: seen.add(key); out.append(t)
            return out
        except Exception as e:       # rate limit or network hiccup: back off and retry
            time.sleep(20 * (i + 1))
    return None

def score(titles, an):
    c = [an.polarity_scores(t)['compound'] for t in titles]
    pos = sum(x >= 0.05 for x in c); neg = sum(x <= -0.05 for x in c)
    return {'n': len(c), 'pos': pos, 'neg': neg, 'attitude': round(100 * (pos + 1) / (pos + neg + 2)),
            'tone': round(sum(c) / len(c), 3) if c else 0.0}

def team_names(season):
    import pandas as pd
    base = 'https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main/team_info/parquet/cfb_team_info_{}.parquet'
    for y in (season, season - 1, 2026):
        try: t = pd.read_parquet(base.format(y), columns=['team_id', 'school', 'mascot']); break
        except Exception: continue
    t['team_id'] = t.team_id.astype(str)
    return {r.team_id: (r.school, r.mascot) for r in t.itertuples()}

def query_for(school, mascot):
    return f'"{school} {mascot}" football' if mascot else f'"{school}" football'

FIELDS = ['season', 'week', 'team_id', 'school', 'n', 'pos', 'neg', 'attitude', 'tone', 'start', 'end']
def load(season):
    p = os.path.join('sentiment', f'{season}.csv')
    if not os.path.exists(p): return []
    with open(p, newline='') as f: return list(csv.DictReader(f))

def save(season, rows):
    os.makedirs('sentiment', exist_ok=True)
    rows = sorted(rows, key=lambda r: (r['week'].split('-')[0], int(r['week'].split('-')[1]), r['team_id']))
    with open(os.path.join('sentiment', f'{season}.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)

def run(season, jobs, delay=1.5, save_every=40):
    """jobs: list of (week_key, team_id, start, end). Skips ones already saved; saves as it goes."""
    names = team_names(season); an = _an(); rows = load(season)
    have = {(r['week'], r['team_id']) for r in rows}; done = fails = 0
    for wk, tid, start, end in jobs:
        if (wk, str(tid)) in have or str(tid) not in names: continue
        school, mascot = names[str(tid)]
        t = headlines(query_for(school, mascot), start, end)
        if t is None:
            fails += 1
            if fails >= 5: print('stopping: news search failed 5 times in a row (rate limited?)', flush=True); break
            continue
        fails = 0
        rows.append(dict(season=season, week=wk, team_id=str(tid), school=school, start=start, end=end, **score(t, an)))
        done += 1
        if done % save_every == 0: save(season, rows); print(season, wk, done, 'teams scored', flush=True)
        time.sleep(delay)
    save(season, rows); print(season, 'scored', done, 'new team-weeks', flush=True)
    return rows

def history(season):
    import pandas as pd
    s = pd.read_parquet(f'https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_cfb_schedules/cfb_schedule_{season}.parquet',
                        columns=['season_type', 'week', 'game_date', 'home_id', 'away_id', 'status'])
    s['date'] = pd.to_datetime(s.game_date, utc=True).dt.tz_convert('America/Chicago')
    t = pd.read_parquet(f'https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_cfb_team_summaries/cfb_team_summaries_{season}.parquet',
                        columns=['team_id', 'division'])
    fbs = set(t[t.division.str.lower() == 'fbs'].team_id.astype(str))
    jobs = []
    for (st, wk), d in s.groupby(['season_type', 'week']):
        first = d.date.min().date(); start = (first - dt.timedelta(days=7)).isoformat(); end = first.isoformat()
        k = f"{3 if str(st) in ('3', 'post') else 2}-{int(wk)}"
        for tid in sorted(set(d.home_id.astype(str)) | set(d.away_id.astype(str))):
            if tid in fbs: jobs.append((k, tid, start, end))
    run(season, jobs)

def week(season, wk, end_date, team_ids):
    end = dt.date.fromisoformat(end_date); start = end - dt.timedelta(days=7)
    run(season, [(wk, t, start.isoformat(), (end + dt.timedelta(days=1)).isoformat()) for t in team_ids], delay=1.0)

if __name__ == '__main__':
    a = sys.argv[1:]
    if a and a[0] == 'history': history(int(a[1]))
    elif a and a[0] == 'week': week(int(a[1]), a[2], a[3], a[4].split(','))
    else: print(__doc__)
