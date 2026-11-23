"""CFB score & spread model, weekly run.  Usage: python3 cfb_weekly.py OUT_DIR [--season 2026]
Ratings: offense/defense points ratings (preseason prior + this season's games, recency-weighted, blowouts capped), blended
with ten opponent-adjusted team stats (fit on 2022-2025), plus emphasised factors (wins over preseason expectation, turnover
margin, rush share, Vegas reaction to last game's cover margin) and each team's own home edge (previous 14 seasons).
Two models: the predictive model (all of the above, no betting line used) and the reactive model (the sportsbook line plus an
adjustment learned from the same inputs: where the market has tended to be wrong; fit on 2008-2025 games, last 4 seasons counted fully,
older seasons x0.8 per year back). Signals: Both agree (predictive 5+ and reactive 1.5+ on the same side), Strong (predictive 3+ edge
where the market overreacted to last week), Models split (predictive 5+ but reactive 1.5+ the other way).
Writes ratings.json, games.json, summary.json ({rows, meta}) and cfb_spread_model.html, an offline presentation page with DraftKings
lines (ESPN's public odds feed), both models' edges and the signals baked in.
Reads public sportsdataverse data (GitHub releases + cfbfastR-data raw files), ESPN odds and ESPN logos.
"""
import os, sys, json, subprocess, datetime as dt
import numpy as np, pandas as pd
from scipy.stats import norm as N

P = dict(H=3.2, lam=3.0, decay=0.95, cap=30.0, cap_slope=0.6, mu0=28.0,
         fcs_o=-18.0, fcs_d=-18.0, lam_fcs=3.0, sigma=16.0, w_mkt=0.0, fcs_carry=0.8)
WEAK = dict(P, lam=0.6, decay=1.0, w_mkt=0.0)            # season-end ratings used for next year's prior
PRIOR_COEF = {"o": [-1.5247, 0.4154, 1.8623, 1.0449, 0.0284], "d": [-1.2986, 0.417, 1.9915, 1.1579, 0.0221]}   # fit on 2022-2025 season-end ratings
W = dict(w_out=0.6, w_to=-1.5, w_rush=0.03, w_react=0.2)   # emphasised factor weights, tuned on 2022-2025 ATS (points per unit)
REACT_DZ = 7          # last-game cover margin beyond +/-7 counts toward the Vegas-reaction factor
STATS = ['EPA_per_play', 'yards_per_play', 'EPA_rushing_per_play', 'EPA_passing_per_play', 'EPA_explosive_rate',
         'first_downs_created_rate', 'rushing_stuff_rate', 'line_yards_per_carry', 'EPA_special_teams', 'EPA_penalty']
STAGE2 = {"c0": 0.1165, "b": {"p_margin_base": 1.16634, "home": -1.02443, "s_EPA_per_play": -32.18952, "s_yards_per_play": 1.80299,
          "s_EPA_rushing_per_play": 0.11547, "s_EPA_passing_per_play": -1.14757, "s_EPA_explosive_rate": 26.25158,
          "s_first_downs_created_rate": 39.99544, "s_rushing_stuff_rate": -22.01354, "s_line_yards_per_carry": 2.31324,
          "s_EPA_special_teams": -0.32973, "s_EPA_penalty": -0.17491}}   # ridge fit on 2022-2025 FBS games
MIN_GP = 3
ANCHOR = {"c0": 0.53693, "b": {"p_margin_base": 0.00866, "home": -0.60956, "s_EPA_per_play": -4.86057, "s_yards_per_play": 0.00318, "s_EPA_rushing_per_play": 0.52598, "s_EPA_passing_per_play": -0.87528, "s_EPA_explosive_rate": 3.06124, "s_first_downs_created_rate": 4.29072, "s_rushing_stuff_rate": -15.65407, "s_line_yards_per_carry": 1.21319, "s_EPA_special_teams": 0.08974, "s_EPA_penalty": 0.00526, "f_out": 0.04523, "f_to": -0.2819, "f_rush": 0.02264, "rx": 0.00208}}   # line adjustment: fit on 2008-2025 closing lines, recency-weighted
LEAN = 2.0            # reactive model "lean": 2+ point edge (2010-2025, each season fit on earlier ones: 1065-912, 53.9%)
BOTH_P, BOTH_R = 5.0, 1.5   # Both agree: predictive 5+ and reactive 1.5+ on the same side (2010-2025 vs closing lines: 463-385, 54.6%)
STRONG_RS = 21        # Strong: predictive 3+ edge and the picked side's last game swung the market 21+ the other way (982-900, 52.2%)

REL = 'https://github.com/sportsdataverse/sportsdataverse-data/releases/download'
RAW = 'https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main'
CACHE = os.environ.get('CFB_CACHE', os.path.join(os.getcwd(), 'cfb_cache'))
os.makedirs(CACHE, exist_ok=True)

# ---------------- data ----------------
def fetch(url, name, refresh=False):
    p = os.path.join(CACHE, name)
    if refresh or not os.path.exists(p):
        r = subprocess.run(['curl', '-sSLf', '-m', '180', '-o', p, url])
        if r.returncode != 0:
            if os.path.exists(p): os.remove(p)
            return None
    return p

def rel(tag, stem, y, refresh=False):
    return fetch(f'{REL}/{tag}/{stem}_{y}.parquet', f'{stem}_{y}.parquet', refresh)

def fbs_teams(y, refresh=False):
    p = rel('espn_cfb_team_summaries', 'cfb_team_summaries', y, refresh) or rel('espn_cfb_team_summaries', 'cfb_team_summaries', y - 1)
    s = pd.read_parquet(p,
                        columns=['team_id', 'pos_team', 'conference', 'division'])
    s = s[s.division.str.lower() == 'fbs']
    return pd.DataFrame({'team': s.pos_team.values, 'conference': s.conference.values},
                        index=s.team_id.astype(str).values)

EMPTY_COLS = ['game_id', 'season', 'week', 'season_type', 'game_date', 'neutral_site', 'home_id', 'away_id', 'home_team', 'away_team',
              'home_score', 'away_score', 'h_to', 'h_rush_att', 'h_pass_att', 'a_to', 'a_rush_att', 'a_pass_att', 'home_team_spread', 'over_under', 'date']
def completed_games(y, refresh=False):
    sp, bp = rel('espn_cfb_schedules', 'cfb_schedule', y, refresh), rel('espn_cfb_team_box', 'team_box', y, refresh)
    if not sp or not bp:   # preseason: no games yet
        e = pd.DataFrame(columns=EMPTY_COLS); e['date'] = pd.to_datetime(e['date'], utc=True); return e
    s = pd.read_parquet(sp)
    s = s[s.status == 'STATUS_FINAL'].copy()
    s['game_id'] = s.game_id.astype('int64')
    for c in ['home_id', 'away_id']: s[c] = s[c].astype(str)
    b = pd.read_parquet(bp)
    b['game_id'] = b.game_id.astype('int64'); b['team_id'] = b.team_id.astype(str)
    b['to'] = pd.to_numeric(b.turnovers, errors='coerce')
    b['rush_att'] = pd.to_numeric(b.rushingAttempts, errors='coerce')
    b['pass_att'] = pd.to_numeric(b.completionAttempts.astype(str).str.split('/').str[1], errors='coerce')
    bx = b[['game_id', 'team_id', 'to', 'rush_att', 'pass_att']]
    s = s.merge(bx.add_prefix('h_').rename(columns={'h_game_id': 'game_id', 'h_team_id': 'home_id'}), on=['game_id', 'home_id'], how='left')
    s = s.merge(bx.add_prefix('a_').rename(columns={'a_game_id': 'game_id', 'a_team_id': 'away_id'}), on=['game_id', 'away_id'], how='left')
    lp = rel('espn_cfb_betting', 'betting', y, refresh)
    if lp:
        bt = pd.read_parquet(lp)[['game_id', 'home_team_spread', 'over_under']]
        bt['game_id'] = bt.game_id.astype('int64')
        s = s.merge(bt.drop_duplicates('game_id'), on='game_id', how='left')
    else:
        s['home_team_spread'] = np.nan; s['over_under'] = np.nan
    s['date'] = pd.to_datetime(s.game_date, utc=True)
    s['season'] = y
    return s.sort_values('date').reset_index(drop=True)

def upcoming_games(y, done_ids, refresh=True):
    p = fetch(f'{RAW}/schedules/csv/cfb_schedules_{y}.csv', f'cfb_schedules_{y}.csv', refresh)
    c = pd.read_csv(p, low_memory=False)
    c = c[(c.completed != True) & (~c.game_id.astype('int64').isin(done_ids))].copy()
    c = c.dropna(subset=['home_id', 'away_id'])
    c['home_id'] = c.home_id.astype('int64').astype(str); c['away_id'] = c.away_id.astype('int64').astype(str)
    c['date'] = pd.to_datetime(c.start_date, utc=True)
    c['neutral_site'] = c.neutral_site.astype(str).str.upper().eq('TRUE')
    return c.sort_values('date')

def prior_inputs(y, R_prev, refresh=False):
    fb = fbs_teams(y, refresh)
    tp = rel('cfb_team_talent', 'cfb_team_talent', y, refresh) or rel('cfb_team_talent', 'cfb_team_talent', y - 1)
    rp = rel('cfb_returning_production', 'cfb_returning_production', y, refresh)
    t = pd.read_parquet(tp)[['team_id', 'talent_composite']]
    r = pd.read_parquet(rp)[['team_id', 'off_returning', 'def_returning']] if rp else pd.DataFrame({'team_id': [], 'off_returning': [], 'def_returning': []})
    t['team_id'] = t.team_id.astype(str); r['team_id'] = r.team_id.astype(str)
    inp = t.merge(r, on='team_id', how='outer').drop_duplicates('team_id').set_index('team_id')
    z = lambda s: (s - s.mean()) / s.std()
    df = pd.DataFrame(index=fb.index)
    fbs_prev = R_prev[R_prev.fbs]
    df['o_prev'] = R_prev.o.reindex(df.index).fillna(fbs_prev.o.quantile(0.1))
    df['d_prev'] = R_prev.d.reindex(df.index).fillna(fbs_prev.d.quantile(0.1))
    df['talent_z'] = z(inp.talent_composite.reindex(df.index)).fillna(-1.5)
    df['offret_z'] = z(inp.off_returning.reindex(df.index)).fillna(0)
    df['defret_z'] = z(inp.def_returning.reindex(df.index)).fillna(0)
    return df

def prior_from_inputs(df, coef, lam, R_prev=None, p=P):
    co, cd = coef['o'], coef['d']
    po = co[0] + co[1]*df.o_prev + co[2]*df.talent_z + co[3]*df.offret_z + co[4]*df.o_prev*df.offret_z
    pdv = cd[0] + cd[1]*df.d_prev + cd[2]*df.talent_z + cd[3]*df.defret_z + cd[4]*df.d_prev*df.defret_z
    out = pd.DataFrame({'po': po, 'pd': pdv, 'lam': lam}, index=df.index)
    if R_prev is not None and p.get('fcs_carry', 0) > 0:   # FCS teams: carry part of last season's rating
        f = R_prev[~R_prev.fbs & ~R_prev.index.isin(out.index)]
        k = p['fcs_carry']
        out = pd.concat([out, pd.DataFrame({'po': k*f.o + (1-k)*p['fcs_o'], 'pd': k*f.d + (1-k)*p['fcs_d'], 'lam': p['lam_fcs']}, index=f.index)])
    return out

# ---------------- model ----------------
def capped(ph, pa, cap, slope):
    m = ph - pa; ex = np.clip(np.abs(m) - cap, 0, None) * (1 - slope)
    return np.where(m > 0, ph - ex, ph), np.where(m < 0, pa - ex, pa)

def fit(g, prior, p=P, ref_date=None):
    if len(g) == 0:
        return pd.DataFrame({'o': prior.po, 'd': prior.pd}), p['mu0']
    teams = sorted(set(g.home_id) | set(g.away_id) | set(prior.index))
    ix = {t: i for i, t in enumerate(teams)}; n = len(teams); K = 2*n + 1
    ph, pa = capped(g.home_score.values.astype(float), g.away_score.values.astype(float), p['cap'], p['cap_slope'])
    hfa = np.where(g.neutral_site.values, 0.0, p['H']/2)
    ref_date = g.date.max() if ref_date is None else ref_date
    w = p['decay'] ** ((ref_date - g.date).dt.total_seconds().clip(lower=0).values / (7*86400))
    hi = np.array([ix[t] for t in g.home_id]); ai = np.array([ix[t] for t in g.away_id])
    A = np.zeros((K, K)); b = np.zeros(K)
    obs = list(zip(hi, ai, ph - hfa, w)) + list(zip(ai, hi, pa + hfa, w))
    for oi, di, y, wt in obs:
        cols = (oi, n + di, 2*n); vals = (1.0, -1.0, 1.0)
        for c1, v1 in zip(cols, vals):
            b[c1] += wt*v1*y
            for c2, v2 in zip(cols, vals): A[c1, c2] += wt*v1*v2
    pr = prior.reindex(teams)
    po = pr.po.fillna(p['fcs_o']).values; pdv = pr.pd.fillna(p['fcs_d']).values; lam = pr.lam.fillna(p['lam_fcs']).values
    A[np.arange(n), np.arange(n)] += lam; b[:n] += lam*po
    A[n + np.arange(n), n + np.arange(n)] += lam; b[n:2*n] += lam*pdv
    A[2*n, 2*n] += 0.5; b[2*n] += 0.5*p['mu0']
    th = np.linalg.solve(A, b)
    return pd.DataFrame({'o': th[:n], 'd': th[n:2*n]}, index=teams), th[2*n]

def predict(R, mu, home, away, neutral, p=P):
    o = lambda ids: R.o.reindex(ids).fillna(p['fcs_o']).values
    d = lambda ids: R.d.reindex(ids).fillna(p['fcs_d']).values
    h = np.where(neutral, 0.0, p['H'])
    return mu + o(home) - d(away) + h/2, mu + o(away) - d(home) - h/2

def team_factors(g, prior, p=P):
    if len(g) == 0: return None
    net = prior.po + prior.pd; fcs = p['fcs_o'] + p['fcs_d']; rows = []
    for side, opp in [('h', 'a'), ('a', 'h')]:
        tid = g.home_id if side == 'h' else g.away_id; oid = g.away_id if side == 'h' else g.home_id
        pts = g.home_score if side == 'h' else g.away_score; opts = g.away_score if side == 'h' else g.home_score
        hf = np.where(g.neutral_site, 0, p['H'] * (1 if side == 'h' else -1))
        em = net.reindex(tid).fillna(fcs).values - net.reindex(oid).fillna(fcs).values + hf
        rows.append(pd.DataFrame({'team_id': tid.values, 'win': (pts > opts).astype(float).values,
                                  'exp_win': N.cdf(em / p['sigma']), 'to_m': (g[f'{opp}_to'] - g[f'{side}_to']).values,
                                  'rush': g[f'{side}_rush_att'].values, 'pas': g[f'{side}_pass_att'].values}))
    t = pd.concat(rows)
    f = t.groupby('team_id').agg(gp=('win', 'size'), wins=('win', 'sum'), exp_wins=('exp_win', 'sum'),
                                 to_m=('to_m', 'mean'), rush=('rush', 'sum'), pas=('pas', 'sum'))
    f['wins_over_exp'] = f.wins - f.exp_wins
    f['rush_rate'] = f.rush / (f.rush + f.pas)
    return f

def factor_diffs(f, home, away, lg_rush):
    """home-minus-away factor values; a team with < MIN_GP games contributes 0."""
    def v(col, ids, fill):
        if f is None: return np.full(len(ids), fill)
        x = f[col].reindex(ids); ok = f.gp.reindex(ids).fillna(0) >= MIN_GP
        return np.where(ok, x.fillna(fill), fill)
    return (v('wins_over_exp', home, 0) - v('wins_over_exp', away, 0),
            v('to_m', home, 0) - v('to_m', away, 0),
            (v('rush_rate', home, lg_rush) - v('rush_rate', away, lg_rush)) * 100)

def season_end(y, refresh=False):
    g = completed_games(y, refresh); fb = fbs_teams(y, refresh)
    R, mu = fit(g, pd.DataFrame({'po': 0.0, 'pd': 0.0, 'lam': WEAK['lam']}, index=fb.index), WEAK)
    R['fbs'] = R.index.isin(fb.index)
    return R, mu

# ---------------- team stats (opponent-adjusted) ----------------
def adv_stats(y, refresh=False):
    p = rel('espn_cfb_adv_team', 'adv_team', y, refresh)
    if not p: return pd.DataFrame(columns=['game_id', 'team_id'] + STATS)
    a = pd.read_parquet(p); a['game_id'] = a.game_id.astype('int64'); a['team_id'] = a.pos_team_id.astype(str)
    return a[['game_id', 'team_id'] + STATS]

def stat_rows(g, a):
    """one row per team-game: offense id, defense id, date, stat values (for that offense)."""
    base = g[['game_id', 'home_id', 'away_id', 'date']]
    h = base.merge(a.rename(columns={'team_id': 'home_id'}), on=['game_id', 'home_id']).rename(columns={'home_id': 'off', 'away_id': 'dfn'})
    w = base.merge(a.rename(columns={'team_id': 'away_id'}), on=['game_id', 'away_id']).rename(columns={'away_id': 'off', 'home_id': 'dfn'})
    return pd.concat([h, w], ignore_index=True)

def stat_ratings(rows, teams, ref_date, lam=3.0, decay=0.95):
    """For each stat: value = mu + o[offense] - d[defense] (ridge, recency-weighted). Returns DataFrame of o+d per team."""
    ix = {t: i for i, t in enumerate(teams)}; n = len(teams); K = 2*n + 1
    out = pd.DataFrame(index=teams)
    rows = rows[rows.off.isin(ix) & rows.dfn.isin(ix)]
    if len(rows) == 0:
        for s in STATS: out[s] = 0.0
        return out
    wt0 = decay ** ((ref_date - rows.date).dt.total_seconds().clip(lower=0).values / (7*86400))
    oi0 = rows.off.map(ix).values; di0 = rows.dfn.map(ix).values
    for s in STATS:
        ok = rows[s].notna().values; oi, di, wt, y = oi0[ok], di0[ok], wt0[ok], rows[s].values[ok].astype(float)
        A = np.zeros((K, K)); b = np.zeros(K); m = np.full(len(oi), 2*n)
        np.add.at(A, (oi, oi), wt); np.add.at(A, (n+di, n+di), wt); A[2*n, 2*n] += wt.sum()
        np.add.at(A, (oi, n+di), -wt); np.add.at(A, (n+di, oi), -wt)
        np.add.at(A, (oi, m), wt); np.add.at(A, (m, oi), wt); np.add.at(A, (n+di, m), -wt); np.add.at(A, (m, n+di), -wt)
        np.add.at(b, oi, wt*y); np.add.at(b, n+di, -wt*y); b[2*n] += (wt*y).sum()
        A[np.arange(2*n), np.arange(2*n)] += lam; A[2*n, 2*n] += 1e-3
        th = np.linalg.solve(A, b)
        out[s] = th[:n] + th[n:2*n]          # team's net (offense + defense) for this stat
    return out

def last_ats(past):
    """each team's cover margin in its most recent game with a line (team perspective)."""
    p = past.dropna(subset=['vegas'])
    if not len(p): return pd.Series(dtype=float)
    r = pd.concat([pd.DataFrame({'team_id': p.home_id, 'date': p.date, 'ats': (p.home_score - p.away_score) + p.vegas}),
                   pd.DataFrame({'team_id': p.away_id, 'date': p.date, 'ats': (p.away_score - p.home_score) - p.vegas})]).sort_values('date')
    return r.groupby('team_id').ats.last()

def react(a):
    a = np.asarray(a, dtype=float); return -np.sign(a) * np.clip(np.abs(a) - REACT_DZ, 0, None)

def matchup(cur, R, mu, SR, f, la, lg_rush, prior, he=None):
    """stage-2 features + core margin for games in cur (home/away ids, neutral_site)."""
    c = cur.copy()
    ph, pa = predict(R, mu, c.home_id, c.away_id, c.neutral_site.values, P)
    c['base_home'] = ph; c['base_away'] = pa; c['p_margin_base'] = ph - pa
    c['home'] = (~c.neutral_site.astype(bool)).astype(float)
    for s in STATS:
        c['s_' + s] = SR[s].reindex(c.home_id).fillna(0).values - SR[s].reindex(c.away_id).fillna(0).values
    c['core'] = STAGE2['c0'] + sum(c[k] * v for k, v in STAGE2['b'].items())
    fo, ft, fr = factor_diffs(f, c.home_id, c.away_id, lg_rush)
    c['f_out'] = fo; c['f_to'] = ft; c['f_rush'] = fr
    c['h_ats1'] = la.reindex(c.home_id).values if len(la) else np.nan
    c['a_ats1'] = la.reindex(c.away_id).values if len(la) else np.nan
    c['rx'] = react(c.h_ats1.fillna(0)) - react(c.a_ats1.fillna(0))
    c['home_edge'] = (he.reindex(c.home_id).fillna(0.0).values if he is not None else 0.0) * c.home
    c['model_margin'] = c.core + W['w_out']*fo + W['w_to']*ft + W['w_rush']*fr + W['w_react']*c.rx + c.home_edge   # ratings-only line (no book)
    c['adj'] = ANCHOR['c0'] + sum(c[k] * v for k, v in ANCHOR['b'].items())                                     # points the book line is off by
    return c

def state_at(g, prior, stat_rows_all, t0):
    """everything known before time t0."""
    past = g[g.date < t0]
    R, mu = fit(past, prior, P, ref_date=t0)
    teams = sorted(set(prior.index) | set(past.home_id) | set(past.away_id))
    SR = stat_ratings(stat_rows_all[stat_rows_all.game_id.isin(set(past.game_id))], teams, t0)
    f = team_factors(past, prior, P)
    la = last_ats(past)
    return past, R, mu, SR, f, la

def walk_forward(g, prior, a, he=None, fbs=None):
    g = g.copy(); g['slate'] = g.season_type.astype(str) + '-' + g.week.astype(str).str.zfill(2)
    starts = g.groupby('slate').date.min().sort_values()
    tot = g[['h_rush_att', 'a_rush_att', 'h_pass_att', 'a_pass_att']].astype(float).sum()
    lg_rush = (tot.h_rush_att + tot.a_rush_att) / tot.sum()
    SRA = stat_rows(g, a); out = []
    for sl, t0 in starts.items():
        past, R, mu, SR, f, la = state_at(g, prior, SRA, t0 - pd.Timedelta(hours=6))
        o = matchup(g[g.slate == sl], R, mu, SR, f, la, lg_rush, prior, he)
        if fbs is not None: o['h_def_rank'] = R.d.reindex(fbs).rank(ascending=False, method='min').reindex(o.home_id).values
        out.append(o)
    o = pd.concat(out).reset_index(drop=True)
    o['margin'] = o.home_score - o.away_score
    return o

def team_power(R, SR, f, lg_rush, idx):
    """neutral-field power: core margin is linear in (team - opponent) terms, so each team gets one number."""
    b = STAGE2['b']
    net = (R.o + R.d).reindex(idx)
    pw = b['p_margin_base'] * net
    for s in STATS: pw = pw + b['s_' + s] * SR[s].reindex(idx).fillna(0)
    return pw

# ---------------- team-specific home field ----------------
HOME_EDGE = dict(window=14, decay=0.8, lam_x=60.0, lam_r=0.5)   # tuned walk-forward on 2022-2026

def team_home_edge(Y, fbs_index, refresh=False):
    """Each team's home-field edge above the FBS average, in points, from actual margins in the previous
    HOME_EDGE['window'] seasons. Margin = team-season rating(home) - team-season rating(away) + season home field
    + team home edge; recent seasons count more (decay) and the team edges are shrunk toward 0 (lam_x)."""
    from scipy import sparse as sp
    from scipy.sparse.linalg import lsqr
    cfg = HOME_EDGE; G = []
    for y in range(Y - cfg['window'], Y):
        p = rel('espn_cfb_schedules', 'cfb_schedule', y, refresh=False)
        if not p: continue
        s = pd.read_parquet(p, columns=['status', 'neutral_site', 'home_id', 'away_id', 'home_score', 'away_score'])
        s = s[s.status == 'STATUS_FINAL'].copy(); s['season'] = y; G.append(s)
    if not G: return pd.Series(0.0, index=fbs_index)
    g = pd.concat(G, ignore_index=True).dropna(subset=['home_score', 'away_score'])
    g['home_id'] = g.home_id.astype(str); g['away_id'] = g.away_id.astype(str)
    g['nn'] = (~g.neutral_site.astype(str).str.upper().eq('TRUE')).astype(float)
    y_ = np.clip((g.home_score - g.away_score).values.astype(float), -40, 40)
    tsh = g.home_id + '_' + g.season.astype(str); tsa = g.away_id + '_' + g.season.astype(str)
    ts = pd.Index(sorted(set(tsh) | set(tsa))); tm = pd.Index(sorted(set(g.home_id))); yrs = pd.Index(sorted(g.season.unique()))
    n, nr, nx, ny = len(g), len(ts), len(tm), len(yrs)
    w = np.sqrt(cfg['decay'] ** (Y - 1 - g.season.values)); r = np.arange(n); nn = g.nn.values
    A = sp.csr_matrix((np.concatenate([w, -w, w*nn, w*nn]),
                       (np.concatenate([r, r, r, r]), np.concatenate([ts.get_indexer(tsh), ts.get_indexer(tsa), nr + tm.get_indexer(g.home_id), nr + nx + yrs.get_indexer(g.season)]))),
                      shape=(n, nr + nx + ny))
    Rg = sp.diags(np.concatenate([np.full(nr, np.sqrt(cfg['lam_r'])), np.full(nx, np.sqrt(cfg['lam_x'])), np.full(ny, 1e-3)]))
    th = lsqr(sp.vstack([A, Rg]).tocsr(), np.concatenate([w * y_, np.zeros(nr + nx + ny)]), atol=1e-12, btol=1e-12, iter_lim=50000)[0]
    x = pd.Series(th[nr:nr + nx], index=tm)
    x = x - x.reindex(fbs_index).mean()
    return x.reindex(fbs_index).fillna(0.0).combine_first(x)

# ---------------- presentation page ----------------
BACKTEST = [  # season, games, predictive miss, Vegas miss, then W-L: Both agree, Strong, predictive 3+, reactive 2+, Models split (predictive side)
    ["2010", 602, 12.95, 12.4, 54, 36, 46, 47, 154, 139, 110, 82, 3, 7],
    ["2011", 672, 12.52, 12.1, 49, 35, 63, 40, 157, 145, 90, 90, 7, 2],
    ["2012", 702, 12.27, 12.09, 40, 37, 45, 47, 169, 149, 106, 95, 4, 5],
    ["2013", 707, 13.04, 12.43, 32, 43, 52, 59, 149, 149, 101, 86, 7, 12],
    ["2014", 734, 13.13, 12.48, 42, 41, 65, 57, 180, 177, 99, 83, 13, 6],
    ["2015", 737, 12.78, 12.33, 41, 28, 78, 63, 197, 166, 97, 60, 10, 12],
    ["2016", 729, 13.47, 12.81, 21, 19, 66, 65, 170, 175, 70, 76, 15, 13],
    ["2017", 746, 13.6, 12.3, 31, 31, 66, 71, 184, 200, 67, 79, 9, 15],
    ["2018", 752, 13.27, 12.54, 25, 19, 74, 59, 170, 181, 59, 39, 6, 13],
    ["2019", 748, 12.82, 12.18, 19, 16, 54, 60, 160, 168, 60, 45, 9, 11],
    ["2020", 534, 13.37, 12.88, 10, 11, 51, 46, 156, 147, 32, 15, 6, 6],
    ["2021", 770, 13.27, 12.54, 18, 24, 64, 67, 189, 189, 37, 49, 7, 6],
    ["2022", 776, 12.42, 12.02, 17, 4, 66, 50, 213, 173, 35, 27, 9, 5],
    ["2023", 792, 12.55, 12.12, 21, 13, 58, 64, 204, 188, 35, 20, 5, 7],
    ["2024", 798, 12.55, 12.07, 25, 20, 65, 48, 216, 178, 44, 52, 5, 5],
    ["2025", 808, 12.15, 11.82, 18, 8, 69, 57, 212, 171, 23, 14, 2, 4],
    ["2010–2021 total", 8433, 13.04, 12.42, 382, 340, 724, 681, 2035, 1985, 928, 799, 96, 108],
    ["2022–2025 total", 3174, 12.42, 12.01, 81, 45, 258, 219, 845, 710, 137, 113, 21, 21],
    ["2010–2025 total", 11607, 12.87, 12.3, 463, 385, 982, 900, 2880, 2695, 1065, 912, 117, 129]]
ODDS = 'https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard?dates={y}&seasontype={st}&week={wk}&groups=80&limit=400'
LOGO = 'https://a.espncdn.com/i/teamlogos/ncaa/500/{}.png'

def team_info(y):
    for yy in (y, y - 1):
        p = fetch(f'{RAW}/team_info/parquet/cfb_team_info_{yy}.parquet', f'cfb_team_info_{yy}.parquet')
        if p:
            t = pd.read_parquet(p, columns=['team_id', 'school', 'abbreviation', 'color']); t['team_id'] = t.team_id.astype(str)
            return t.drop_duplicates('team_id').set_index('team_id')
    return pd.DataFrame(columns=['school', 'abbreviation', 'color'])

def _num(x):
    s = str(x).strip().upper()
    if s in ('EVEN', 'PK', 'PICK'): return 0.0
    try: return float(s.lstrip('OU'))
    except ValueError: return None

def book_lines(Y, slates):
    """Home spread, total and book (DraftKings is ESPN's listed book) per game id, from ESPN's public odds feed."""
    out = {}
    for st, wk in slates:
        p = fetch(ODDS.format(y=Y, st=st, wk=wk), f'odds_{st}_{wk}.json', refresh=True)
        if not p: continue
        try: evs = json.load(open(p)).get('events', [])
        except ValueError: continue
        for ev in evs:
            c = ev['competitions'][0]; tm = {x.get('homeAway'): x.get('team') or {} for x in c.get('competitors', [])}
            for o in sorted(c.get('odds') or [], key=lambda o: (o.get('provider') or {}).get('priority', 99)):
                ps = (o.get('pointSpread') or {}).get('home') or {}
                ln = next((v for v in (_num((ps.get(k) or {}).get('line')) for k in ('close', 'current', 'open')) if v is not None), None)
                s = _num(o.get('spread'))
                if ln is None and s is not None and (o.get('homeTeamOdds') or {}).get('favorite') is True: ln = -abs(s)
                if ln is None and s is not None and (o.get('awayTeamOdds') or {}).get('favorite') is True: ln = abs(s)
                d = str(o.get('details') or '').rsplit(' ', 1)
                if ln is None and len(d) == 2 and _num(d[1]) is not None:
                    if d[0].upper() == str(tm.get('home', {}).get('abbreviation', '')).upper(): ln = -abs(_num(d[1]))
                    elif d[0].upper() == str(tm.get('away', {}).get('abbreviation', '')).upper(): ln = abs(_num(d[1]))
                if ln is None: continue
                out[str(ev['id'])] = {'line': ln, 'total': _num(o.get('overUnder')), 'book': (o.get('provider') or {}).get('name') or 'Sportsbook'}
                break
    return out

def logos(ids):
    """Team logos as small base64 PNGs so the page needs no internet; teams without one show their initials."""
    try:
        from PIL import Image; import base64, io
    except ImportError: return {}
    out, miss = {}, 0
    for i in sorted(ids):
        p = fetch(LOGO.format(i), f'logo_{i}.png')
        if not p:
            miss += 1
            if miss >= 3 and not out: return {}   # logo host not reachable from here
            continue
        try:
            im = Image.open(p).convert('RGBA'); im.thumbnail((72, 72)); b = io.BytesIO(); im.save(b, 'PNG', optimize=True)
            out[i] = base64.b64encode(b.getvalue()).decode()
        except Exception: pass
    return out

def write_html(out_dir, Y):
    """cfb_spread_model.html: one self-contained page (works offline) with model lines, sportsbook lines, edges and signals baked in.
    Optional next to this script: page_inputs.json {overrides: [{team_id, points, note}], weights: {w_out, w_to, w_rush, w_react}} from the
    dashboard, and lines.json {game_id: {line, total, book}} to fill or replace sportsbook lines."""
    from zoneinfo import ZoneInfo
    rd = lambda n: json.load(open(os.path.join(out_dir, n)))['rows']
    games, ratings, summary = rd('games.json'), rd('ratings.json'), rd('summary.json')
    s0 = next((r for r in summary if r['bucket'] == 'all'), {})
    pi = json.load(open('page_inputs.json')) if os.path.exists('page_inputs.json') else {}
    w = dict(W); w.update({k: float(v) for k, v in (pi.get('weights') or {}).items() if k in W and v is not None})
    adj = {str(o['team_id']): float(o['points']) for o in pi.get('overrides') or [] if o.get('points')}
    slates = sorted({(3 if str(g['season_type']).lower().startswith('post') else 2, int(g['week'])) for g in games})[:2]
    sp = os.path.join('lines', f'{Y}.json')
    try: lines = json.load(open(sp))
    except (OSError, ValueError): lines = {}
    stamp = pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%dT%H:%MZ'); fresh = book_lines(Y, slates)
    for k, v in fresh.items():        # a game that has kicked off drops out of the feed and keeps its last line; line0 = first line pulled
        prev = lines.get(k) or {}
        lines[k] = dict(v, at=stamp, line0=prev.get('line0', prev.get('line', v['line'])), at0=prev.get('at0', prev.get('at', stamp)))
    if fresh or lines:
        os.makedirs('lines', exist_ok=True); json.dump(lines, open(sp, 'w'), indent=0, sort_keys=True)
    if os.path.exists('lines.json'): lines.update(json.load(open('lines.json')))
    z = lambda v: 0.0 if v is None else float(v)
    FA = {}
    try:
        import csv
        with open(os.path.join('sentiment', f'{Y}.csv'), newline='') as fh:
            for r in csv.DictReader(fh): FA[(r['week'], r['team_id'])] = [int(r['attitude']), int(r['n'])]
    except (OSError, ValueError, KeyError): pass
    DD_RANK, DD_WEEK = 25, 5     # Defense Dog: home underdog (getting points, not neutral) with a top-25 defense, from week 5 on (or a bowl)
    def dd(g, ln, st):
        r = g.get('h_def_rank')
        return bool(ln is not None and ln > 0 and not g['neutral_site'] and r is not None and r <= DD_RANK and (st == 3 or int(g['week']) >= DD_WEEK))
    fa = lambda k, a, h: [FA.get((k, a)), FA.get((k, h))] if (FA.get((k, a)) or FA.get((k, h))) else None
    def pred(g, h, a, ov):            # predictive model line (home margin): ratings, team stats, factors, team home field
        return (z(g['core']) + w['w_out']*z(g['f_out']) + w['w_to']*z(g['f_to']) + w['w_rush']*z(g['f_rush']) + w['w_react']*z(g['rx'])
                + z(g.get('home_edge')) + (adj.get(h, 0) - adj.get(a, 0) if ov else 0))
    def sigs(mp, ln, er, ha, aa):
        if ln is None: return None, None, {}
        ep = mp + ln; f = {}
        if er is not None and abs(ep) >= BOTH_P and abs(er) >= BOTH_R: f['ba' if (ep > 0) == (er > 0) else 'sp'] = True
        pk, op = (ha, aa) if ep > 0 else (aa, ha)
        if abs(ep) >= 3 and abs(ln) < 28 and max(-z(pk), z(op)) >= STRONG_RS: f['st'] = True
        f['sig'] = 'Both' if f.get('ba') else 'Strong' if f.get('st') else 'Split' if f.get('sp') else ''
        return round(ep, 2), (None if er is None else round(er, 2)), f
    def grade(e, ln, hs, aw):
        if e is None or ln is None or abs(e) < 0.05: return None
        c = (z(hs) - z(aw)) + ln; return 'P' if c == 0 else ('W' if (e > 0) == (c > 0) else 'L')
    key = lambda g: f"{3 if str(g['season_type']).lower() in ('3', 'post', 'postseason') else 2}-{int(g['week'])}"
    try: past = rd('results.json')
    except (OSError, KeyError, ValueError): past = []
    PG = []
    for g in past:
        h, a = str(g['home_id']), str(g['away_id']); k = key(g); mp = pred(g, h, a, False)
        ln = g.get('vegas'); hs, aw = g['home_score'], g['away_score']
        ep, er, f = sigs(mp, ln, None if ln is None else z(g.get('adj')), g['h_ats1'], g['a_ats1'])
        PG.append(dict({'id': str(g['game_id']), 'k': k, 'd': g['date'], 'n': bool(g['neutral_site']), 'h': h, 'a': a,
                        'hn': g['home_team'], 'an': g['away_team'], 'm': round(mp, 2), 'line': ln, 'e': ep, 'er': er, 'sig': '',
                        'r': grade(ep, ln, hs, aw), 'rr': grade(er, ln, hs, aw), 'hs': hs, 'as': aw, 'fb': bool(g['fbs_both']),
                        'fa': fa(k, a, h), 'dd': dd(g, ln, int(k[0]))}, **f))
    done = {x['id'] for x in PG}; cur = {key(g) for g in games if g['this_week']}
    G = []
    for x in past:                       # this week's games already played: keep them on the board with the final score
        k = key(x)
        if k not in cur: continue
        h, a = str(x['home_id']), str(x['away_id']); pg = next(q for q in PG if q['id'] == str(x['game_id']))
        b = lines.get(str(x['game_id'])) or {}; ln = b.get('line', x.get('vegas')); mp = pred(x, h, a, False)
        ep, er, f = sigs(mp, ln, None if ln is None else z(x.get('adj')), x['h_ats1'], x['a_ats1'])
        G.append(dict({'id': pg['id'], 'd': x['date'], 'n': bool(x['neutral_site']), 'h': h, 'a': a, 'hn': x['home_team'], 'an': x['away_team'],
                       'm': round(mp, 2), 't': round(z(x.get('model_total')), 1), 'wp': int(round(100 * N.cdf(abs(mp) / P['sigma']))), 'line': ln,
                       'l0': b.get('line0'), 'bt': b.get('total', x.get('over_under')), 'bk': b.get('book') or ('Closing line' if ln is not None else None),
                       'e': ep, 'er': er, 'sig': '', 'ha': x['h_ats1'], 'aa': x['a_ats1'], 'he': round(z(x.get('home_edge')), 2),
                       'tw': True, 'fb': bool(x['fbs_both']), 'adj': False, 'fin': [x['away_score'], x['home_score']], 'fa': pg['fa'], 'dd': dd(x, ln, int(k[0]))}, **f))
    for g in games:
        if str(g['game_id']) in done: continue
        h, a = str(g['home_id']), str(g['away_id']); k = key(g); mp = pred(g, h, a, True)
        b = lines.get(str(g['game_id'])) or {}; ln = b.get('line')
        ep, er, f = sigs(mp, ln, None if ln is None else z(g.get('adj')) + adj.get(h, 0) - adj.get(a, 0), g['h_ats1'], g['a_ats1'])
        G.append(dict({'id': str(g['game_id']), 'd': g['date'], 'n': bool(g['neutral_site']), 'h': h, 'a': a, 'hn': g['home_team'], 'an': g['away_team'],
                       'm': round(mp, 2), 't': round(z(g['model_total']), 1), 'wp': int(round(100 * N.cdf(abs(mp) / P['sigma']))), 'line': ln,
                       'l0': b.get('line0'), 'bt': b.get('total'), 'bk': b.get('book'), 'e': ep, 'er': er, 'sig': '', 'ha': g['h_ats1'], 'aa': g['a_ats1'],
                       'he': round(z(g.get('home_edge')), 2), 'tw': bool(g['this_week']), 'fb': bool(g['fbs_both']), 'adj': bool(adj.get(h) or adj.get(a)),
                       'fa': fa(k, a, h), 'dd': dd(g, ln, int(k[0]))}, **f))
    R = []
    for r in ratings:
        i = str(r['team_id'])
        R.append({'id': i, 'tn': r['team'], 'rec': f"{int(z(r['wins']))}-{int(z(r['losses']))}" if r.get('gp') else '0-0',
                  'pw': z(r['core_power']) + w['w_out']*z(r['f_out']) + w['w_to']*z(r['f_to']) + w['w_rush']*z(r['f_rush']) + adj.get(i, 0),
                  'o': z(r['off']), 'df': z(r['def']), 'epa': r['epa_net'], 'pre': r['rank_pre'], 'woe': r['wins_over_exp'], 'to': r['to_m'],
                  'rr': r['rush_rate'], 'la': r['last_ats'], 'he': r.get('home_edge'), 'adj': adj.get(i),
                  'mv': None if r.get('core_power_lw') is None else z(r['core_power']) - z(r['core_power_lw'])})
    R.sort(key=lambda t: -t['pw'])
    for k, t in enumerate(R): t['rk'] = k + 1; t['pw'] = round(t['pw'], 2)
    ti = team_info(Y); ids = {x['h'] for x in G} | {x['a'] for x in G} | {t['id'] for t in R} | {x['h'] for x in PG} | {x['a'] for x in PG}
    im = logos({t['id'] for t in R})   # FBS logos only; other teams show initials (keeps the page small)
    sv = lambda df, c, i: df[c].get(i) if i in df.index and isinstance(df[c].get(i), str) else None
    teams = {i: {'s': sv(ti, 'school', i), 'a': sv(ti, 'abbreviation', i), 'c': sv(ti, 'color', i), 'i': im.get(i)} for i in ids}
    ct = lambda t: t.tz_convert(ZoneInfo('America/Chicago')).strftime('%a %b %-d, %-I:%M %p CT')
    books = [x['bk'] for x in G if x['line'] is not None and not x.get('fin')]
    meta = {'season': s0.get('season'), 'week': s0.get('next_week'), 'data_through': s0.get('data_through'),
            'run_at': ct(pd.Timestamp(str(s0.get('run_at', '')).replace(' UTC', ''), tz='UTC')) if s0.get('run_at') else '',
            'lines_at': ct(pd.Timestamp.now(tz='UTC')), 'n_lines': len(books), 'book': max(set(books), key=books.count) if books else None,
            'live': sorted({(3 if str(g['season_type']).lower().startswith('post') else 2, int(g['week'])) for g in games if g['this_week']}),
            'adj': [[(teams.get(i) or {}).get('s') or o.get('team') or i, float(o['points']), o.get('note') or ''] for o in pi.get('overrides') or [] if o.get('points') for i in [str(o['team_id'])]]}
    payload = json.dumps({'meta': meta, 'teams': teams, 'games': G, 'past': PG, 'ratings': R, 'summary': summary, 'backtest': BACKTEST},
                         separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    p = os.path.join(out_dir, 'cfb_spread_model.html')
    open(p, 'w', encoding='utf-8').write(PAGE.replace('__DATA__', payload, 1))
    print('cfb_spread_model.html', os.path.getsize(p), 'bytes;', len(books), 'games with book lines;', len(im), 'logos;', len(PG), 'past games')
    return p

PAGE = r'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CFB spread model</title>
<style>
:root{--ink:#14213d;--paper:#f2f4f6;--surface:#fff;--line:#d8dde3;--muted:#5a6575;--strong:#0b7a52;--strong-bg:#e3f3ec;--lean:#9a5b0b;--lean-bg:#fbf0dd;--pass:#6b7280;--neg:#b42318;--focus:#2f6fdb;
--cond:"Barlow Condensed","Avenir Next Condensed","Roboto Condensed","Arial Narrow",system-ui,sans-serif;--body:"Barlow","Avenir Next",system-ui,-apple-system,"Segoe UI",sans-serif}
@media (prefers-color-scheme:dark){:root{--ink:#e8edf4;--paper:#0f1622;--surface:#172131;--line:#2a3648;--muted:#9aa6b6;--strong:#3fcf98;--strong-bg:#11352a;--lean:#f0b450;--lean-bg:#3a2b12;--pass:#9aa3af;--neg:#ff7b6b;--focus:#7aa7ff}}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.45 var(--body)}
button,input,select{font:inherit;color:inherit}:focus-visible{outline:2px solid var(--focus);outline-offset:2px}
.top{background:var(--surface);border-bottom:1px solid var(--line)}
.in{max-width:1440px;margin:0 auto;padding:18px 20px 0}
h1{font:700 34px/1 var(--cond);margin:0;letter-spacing:-.01em}h1 span{color:var(--muted);font-weight:600}
.stamp{color:var(--muted);font-size:13px;margin-top:8px}
nav{display:flex;gap:2px;overflow-x:auto;margin-top:12px;scrollbar-width:none}
nav button{border:0;background:none;padding:10px 14px;font:600 17px var(--cond);color:var(--muted);border-bottom:3px solid transparent;cursor:pointer;white-space:nowrap}
nav button[aria-selected=true]{color:var(--ink);border-bottom-color:var(--ink)}
main{max-width:1440px;margin:0 auto;padding:18px 20px 48px}section[hidden]{display:none}
.ctl{display:flex;flex-wrap:wrap;gap:8px 10px;align-items:center;margin-bottom:12px}
.ctl select,.ctl input[type=search]{border:1px solid var(--line);background:var(--surface);border-radius:6px;padding:7px 9px;min-width:0}
.ctl input[type=search]{flex:1 1 180px;max-width:280px}
.chk{display:inline-flex;gap:6px;align-items:center;font-size:14px;color:var(--muted)}
.note{color:var(--muted);font-size:13px;max-width:78ch;margin:10px 0 0}
.tw{overflow-x:auto;background:var(--surface);border:1px solid var(--line);border-radius:8px}
table{border-collapse:separate;border-spacing:0;width:100%;font-variant-numeric:tabular-nums;font-size:14px}
th{font:600 14px/1.2 var(--cond);color:var(--muted);text-align:right;padding:10px 12px;border-bottom:1px solid var(--line);white-space:nowrap;background:var(--surface);position:sticky;top:0;z-index:2}
td{padding:10px 12px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap;vertical-align:middle}
tr:last-child td{border-bottom:0}.l{text-align:left}
.st{position:sticky;left:0;background:var(--surface);z-index:1;width:1%;box-shadow:1px 0 0 var(--line)}th.st{z-index:3}
.mu{display:grid;grid-template-columns:76px 22px 76px;align-items:start;justify-items:center;gap:0 2px}
.sd{display:flex;flex-direction:column;align-items:center;gap:4px;width:76px}
.lg{width:40px;height:40px;object-fit:contain;display:block}
.mo{width:40px;height:40px;border-radius:50%;display:grid;place-items:center;color:#fff;font:700 14px var(--cond);flex:none}
.sc{font:600 13px/1.1 var(--cond);text-align:center;white-space:normal;max-width:76px}
.at{align-self:center;margin-top:-16px;font:600 13px var(--cond);color:var(--muted)}
.ms{display:none}
tr.Strong .st{box-shadow:inset 4px 0 0 var(--strong),1px 0 0 var(--line)}tr.Lean .st{box-shadow:inset 4px 0 0 var(--strong),1px 0 0 var(--line)}
.big{font:600 18px/1.1 var(--cond)}.sub{display:block;color:var(--muted);font-size:12px;margin-top:2px}
.pill{display:inline-block;border-radius:999px;padding:2px 10px;font:600 14px var(--cond)}
.pill.Strong{background:var(--strong-bg);color:var(--strong)}.pill.Lean{background:var(--strong-bg);color:var(--strong)}.pill.Pass{color:var(--pass);border:1px solid var(--line)}
.pos{color:var(--strong)}.neg{color:var(--neg)}
.tm{display:inline-flex;align-items:center;gap:8px}.tm .lg,.tm .mo{width:24px;height:24px;font-size:10px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-bottom:18px}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:14px 16px}
.tile .k{color:var(--muted);font-size:13px}.tile .v{font:700 30px/1.1 var(--cond);margin:4px 0 2px}
h2{font:700 22px var(--cond);margin:22px 0 10px}.how{max-width:78ch;line-height:1.6}.how li{margin-bottom:8px}
.live{color:var(--neg);font-weight:600}.live:before{content:"";display:inline-block;width:7px;height:7px;border-radius:50%;background:currentColor;margin-right:5px;vertical-align:1px}
.pill.W{background:var(--strong-bg);color:var(--strong)}.pill.L{background:#fde8e6;color:var(--neg)}.pill.P{color:var(--pass);border:1px solid var(--line)}
@media (prefers-color-scheme:dark){.pill.L{background:#3d1c19}}
.pill.Both{background:var(--strong-bg);color:var(--strong)}.pill.Strong{background:var(--lean-bg);color:var(--lean)}.pill.Split{color:var(--neg);border:1px solid var(--line)}
tr.Both .st{box-shadow:inset 4px 0 0 var(--strong),1px 0 0 var(--line)}tr.Strong .st{box-shadow:inset 4px 0 0 var(--lean),1px 0 0 var(--line)}tr.Split .st{box-shadow:inset 4px 0 0 var(--neg),1px 0 0 var(--line)}
.pills{display:flex;flex-direction:column;align-items:flex-start;gap:4px}.tw2{color:var(--strong)}.ag{color:var(--neg)}
.pill.dd{background:#e6eefc;color:#1d4ed8;margin-top:4px}@media (prefers-color-scheme:dark){.pill.dd{background:#172a4d;color:#9cbcff}}
.wk{font-size:14px;color:var(--ink);margin:0 0 10px}
.empty{padding:22px 14px;color:var(--muted);text-align:left;white-space:normal}
@media (max-width:640px){.in{padding:14px 14px 0}main{padding:14px 12px 40px}h1{font-size:28px}
.mu{grid-template-columns:58px 16px 58px}.sd{width:58px}.sc{max-width:58px;font-size:12px}.mu .lg,.mu .mo{width:32px;height:32px;font-size:12px}
.ms{display:block;text-align:center;margin-top:6px;font:600 13px/1.3 var(--cond);white-space:normal;max-width:150px}td,th{padding:9px}}
@media print{nav,.ctl{display:none}body{background:#fff}.tw{border:0;overflow:visible}th{position:static}}
</style></head><body>
<header class="top"><div class="in">
<h1>CFB spread model <span id="wk"></span></h1>
<div class="stamp" id="stamp"></div>
<nav role="tablist" aria-label="Sections">
<button role="tab" data-t="games" aria-selected="true">Games</button><button role="tab" data-t="ratings" aria-selected="false">Power ratings</button>
<button role="tab" data-t="record" aria-selected="false">Record</button><button role="tab" data-t="how" aria-selected="false">How it works</button>
</nav></div></header>
<main>
<section id="games">
<div class="ctl">
<select id="scope" aria-label="Which week"></select>
<select id="sort" aria-label="Sort games"><option value="time">By kickoff</option><option value="edge">Biggest edge</option><option value="signal">Signals first</option><option value="close">Closest games</option></select>
<input type="search" id="q" placeholder="Find a team" aria-label="Find a team">
<label class="chk"><input type="checkbox" id="fbs" checked> FBS vs FBS only</label>
</div>
<p class="wk" id="wsum" hidden></p>
<div class="tw"><table><thead id="bh"></thead><tbody id="board"></tbody></table></div>
<p class="note" id="gnote"></p>
</section>
<section id="ratings" hidden>
<div class="ctl"><select id="rshow" aria-label="How many teams"><option value="25">Top 25</option><option value="all">All FBS</option></select>
<input type="search" id="rq" placeholder="Find a team" aria-label="Find a team in ratings"></div>
<div class="tw"><table><thead><tr><th>#</th><th class="l st">Team</th><th class="l">Record</th><th>Power</th><th>Off</th><th>Def</th><th>EPA/play net</th><th>Preseason #</th><th>Wins vs preseason exp.</th><th>TO margin / g</th><th>Rush %</th><th>Last game vs spread</th><th>1-week move</th><th>Home field</th><th>Adjustment</th></tr></thead><tbody id="rt"></tbody></table></div>
<p class="note">Power = points better than an average FBS team on a neutral field, from scoring plus opponent-adjusted team stats and the emphasised factors. Off and Def are scoring ratings in points. Home field is each team's own extra (or missing) home edge on top of the league average. A team that missed the spread by 21+ last game (green) is a Vegas-overreaction candidate.</p>
</section>
<section id="record" hidden>
<div class="tiles" id="tiles"></div>
<h2>Backtest by season</h2>
<div class="tw"><table><thead><tr><th class="l">Season</th><th>Games</th><th>Predictive miss</th><th>Vegas miss</th><th>Both agree</th><th>Strong</th><th>Predictive 3+</th><th>Reactive 2+</th><th>Models split</th></tr></thead><tbody id="bt"></tbody></table></div>
<p class="note">FBS vs FBS, graded against the closing line; 52.4% breaks even at standard odds. Every game is predicted from earlier games only. The reactive model's weights for each season were fit only on earlier seasons. The predictive model's blend weights were tuned on 2022–25, so its numbers for those seasons (and anything that uses it) flatter it; 2010–21 and 2026 are the fair test. Models split is graded on the predictive model's side.</p>
</section>
<section id="how" hidden><ul class="how">
<li><b>Scoring ratings.</b> Each team has an offense and a defense rating in points, adjusted for opponent and home field, starting from a preseason prior (last season, roster talent, returning production) that fades as games come in. Recent games count more; blowouts past 30 are mostly ignored.</li>
<li><b>Team stats.</b> Ten opponent-adjusted stats (EPA per play, yards per play, rushing and passing EPA, explosive plays, first-down rate, run-stuff rate, line yards, special teams, penalties) are blended with the scoring rating by a regression fit on 2022–2025 FBS games.</li>
<li><b>Factors</b> (after 3+ games): wins over preseason expectation, turnover margin, rush rate, and how the market reacted to last week's result.</li>
<li><b>Predictive model</b> = everything above combined into its own line, without looking at the betting line. This is the model line shown for every game.</li>
<li><b>Reactive model</b> starts from the sportsbook line and adjusts it by what the same inputs say the market tends to miss (fit on 2008–2025 games, the last 4 seasons counted fully and older ones a little less each year back). Its edge is how far it moves off the line.</li>
<li><b>Home field.</b> League average plus each team's own home edge from its previous 14 seasons, pulled toward average.</li>
<li><b>Both agree</b> = the predictive model is 5+ points off the line and the reactive model 1.5+ on the same side. 2010–2025 against closing lines: 463–385 (54.6%); 2010–21 alone 52.9%.</li>
<li><b>Strong</b> = a predictive 3+ edge where the market just overreacted the other way (your side missed by 21+ last game, or the opponent covered by 21+), lines under 28. A "look closer" flag: 982–900 (52.2%).</li>
<li><b>Models split</b> = the predictive model is 5+ off the line but the reactive model leans 1.5+ the other way. The predictive side went 117–129 (47.6%), so treat its edge with caution.</li>
<li><b>Line move</b> compares the first DraftKings line pulled for the game (usually Monday's) with the latest one (live while the page is online). From 2012–2021 opening lines, the line moved toward the predictive model's 5+ point picks 63% of the time.</li>
<li><b>Lines</b> are the DraftKings spreads pulled when this page was built. They don't change after that.</li>
<li><b>Live scores</b> for this week's games come from ESPN's public scoreboard while the page is open and online, refreshed every minute during games. Offline, the page works the same without them.</li>
<li><b>Defense Dog</b> tags a home underdog with a top-25 defense (model rating at kickoff), from week 5 on. Tested on 2010–2026 closing lines, these teams covered 136–143 (49%), so the tag is tracked, not used in the model lines; past weeks show how they did.</li>
<li><b>Fan attitude</b> is the mood of the week's coverage of each team (local papers, beat writers, fan sites) from Google News headlines in the 7 days before the week's games: the share of positive vs negative headlines, 50% = neutral. Game-result words are ignored, since the model already has the results. It is tracked only, not used in the model lines.</li>
<li><b>Past weeks</b> show what both models would have said before each game, using only games played earlier, graded against the closing line. Pick = the side the predictive model liked against that line.</li>
<li><b>Win %</b> assumes the final margin lands within about ±16 points of the model line two times out of three.</li>
</ul></section>
</main>
<script id="d" type="application/json">__DATA__</script>
<script>
(function(){'use strict';
var D=JSON.parse(document.getElementById('d').textContent),M=D.meta,T=D.teams,$=function(s){return document.querySelector(s)};
function esc(s){return String(s==null?'':s).replace(/[&<>"']/g,function(c){return{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]})}
function f1(x){return x==null||!isFinite(x)?'—':(x>0?'+':'')+(+x).toFixed(1)}
function pct(x){return x==null?'—':(100*x).toFixed(1)+'%'}
function nm(id,fb){return(T[id]&&T[id].s)||fb||id}
function logo(id){var t=T[id]||{},c=/^#[0-9a-f]{6}$/i.test(t.c||'')?t.c:'#5a6575',mo='<span class="mo" style="background:'+c+'" aria-hidden="true">'+esc((t.a||nm(id)||'?').slice(0,3))+'</span>';
if(t.i)return'<img class="lg" alt="'+esc(nm(id))+'" src="data:image/png;base64,'+t.i+'">';
if(!/^\d+$/.test(id))return mo;
return'<img class="lg" alt="'+esc(nm(id))+'" src="https://a.espncdn.com/i/teamlogos/ncaa/500/'+id+'.png" data-mo="'+encodeURIComponent(mo)+'">'}
document.addEventListener('error',function(e){var i=e.target;if(i.dataset&&i.dataset.mo)i.outerHTML=decodeURIComponent(i.dataset.mo)},true);
function sg(x,hi,lo){return x==null?'—':'<span class="'+(x>hi?'pos':x<lo?'neg':'')+'">'+f1(x)+'</span>'}
var S={scope:'week',sort:'time',q:'',fbs:true,rshow:'25',rq:''},RK={Both:0,Strong:1,Split:2,'':3},L={},RW={W:'Covered',L:'Missed',P:'Push'};
var PW={};(D.past||[]).forEach(function(g){(PW[g.k]=PW[g.k]||[]).push(g)});
function wlab(k){var p=k.split('-');return p[0]=='3'?(p[1]=='1'?'Bowls':'Postseason '+p[1]):'Week '+p[1]}
(function(){var o='<option value="week">This week</option><option value="all">All upcoming</option>',ks=Object.keys(PW).sort(function(a,b){var x=a.split('-'),y=b.split('-');return(y[0]-x[0])||(y[1]-x[1])});
if(ks.length)o+='<optgroup label="Past weeks: results">'+ks.map(function(k){return'<option value="p:'+k+'">'+wlab(k)+' results</option>'}).join('')+'</optgroup>';$('#scope').innerHTML=o})();
var HU='<tr><th class="l st">Matchup</th><th>Score</th><th>Predictive line</th><th>Projected score</th><th>Fav. win %</th><th>Book line</th><th>Predictive edge</th><th>Reactive edge</th><th class="l">Pick</th><th class="l">Signal</th><th>Line move</th><th>Fan attitude</th><th>Total: model / book</th><th>Last game vs spread</th><th>Home field</th><th class="l">Kickoff</th></tr>',
HP='<tr><th class="l st">Matchup</th><th>Final</th><th>Predictive line</th><th>Closing line</th><th>Predictive edge</th><th>Reactive edge</th><th class="l">Pick</th><th class="l">Against the spread</th><th class="l">Signal</th><th>Fan attitude</th><th class="l">Kickoff</th></tr>';
function ae(g){return g.e==null?-1:Math.abs(g.e)}
function mu(g,hs,as,extra){return'<td class="l st"><div class="mu"><div class="sd">'+logo(g.a)+'<span class="sc">'+as+'</span></div><span class="at">'+(g.n?'vs':'at')+'</span><div class="sd">'+logo(g.h)+'<span class="sc">'+hs+'</span></div></div>'+(extra||'')+'</td>'}
function ml(g,hs,as){var ln=Math.round(Math.abs(g.m)*2)/2;return ln==0?'Pick':(g.m>=0?hs:as)+' −'+ln}
function spr(v,hs,as){return v==null?'—':v==0?'Pick':(v<0?hs+' −'+(-v):as+' −'+v)}
function kt(g){var d=new Date(g.d);return isNaN(d)?'TBD':d.toLocaleString(undefined,{weekday:'short',month:'short',day:'numeric',hour:'numeric',minute:'2-digit'})}
function rs(e,line,hs,as){if(e==null||Math.abs(e)<.05||line==null)return null;var c=hs-as+line;return c==0?'P':((e>0)==(c>0)?'W':'L')}
function lv(g){return L[g.id]||(g.fin?{s:'post',t:'Final',as:g.fin[0],hs:g.fin[1]}:null)}
function score(g){var x=lv(g);if(!x||x.s=='pre')return'—';var t='<span class="big">'+x.as+'–'+x.hs+'</span>';
if(x.s=='in')return t+'<span class="sub live">'+esc(x.t)+'</span>';var r=rs(g.e,g.line,x.hs,x.as);return t+'<span class="sub">Final</span>'+(r?'<span class="pill '+r+'" style="margin-top:4px">'+RW[r]+'</span>':'')}
function edg(v,big){return v==null?'—':'<span class="big'+(Math.abs(v)>=big?' pos':'')+'">'+Math.abs(v).toFixed(1)+'</span><span class="sub">'+(v>0?'home':'away')+'</span>'}
function ddp(g){return g.dd?'<span class="pill dd" title="Home underdog with a top-25 defense (tracked, not in the model)">Defense Dog</span>':''}
function fat(g){if(!g.fa)return'—';var f=function(x){return x?'<span title="'+x[1]+' headlines">'+x[0]+'%</span>':'—'};return'<span class="big">'+f(g.fa[0])+' – '+f(g.fa[1])+'</span><span class="sub">away – home</span>'}
function pills(g){var p=[];if(g.ba)p.push('<span class="pill Both">Both agree</span>');if(g.st)p.push('<span class="pill Strong">Strong</span>');if(g.sp)p.push('<span class="pill Split" title="Predictive 5+ but the reactive model leans the other way">Models split</span>');if(g.dd)p.push(ddp(g));return p.length?'<div class="pills">'+p.join('')+'</div>':'—'}
function mv(g){var x=L[g.id],now=x&&x.s=='pre'&&x.ln!=null?x.ln:g.line,from=g.l0!=null?g.l0:g.line,hs=esc(nm(g.h,g.hn)),as=esc(nm(g.a,g.an));if(now==null||from==null)return'—';
var d=Math.round((now-from)*2)/2;if(!d)return'<span class="big">No move</span><span class="sub">'+spr(now,hs,as)+'</span>';
var tw=g.e!=null&&Math.abs(g.e)>=.05&&((g.e>0&&d<0)||(g.e<0&&d>0));return'<span class="big '+(tw?'tw2':'ag')+'">'+Math.abs(d)+' '+(tw?'toward':'away from')+' pick</span><span class="sub">'+spr(from,hs,as)+' → '+spr(now,hs,as)+'</span>'}
function sorter(){var bt=function(a,b){return(a.d||'').localeCompare(b.d||'')};
return S.sort=='edge'?function(a,b){return ae(b)-ae(a)||bt(a,b)}:S.sort=='signal'?function(a,b){return RK[a.sig||'']-RK[b.sig||'']||ae(b)-ae(a)}:S.sort=='close'?function(a,b){return Math.abs(a.m)-Math.abs(b.m)}:bt}
function match(g){return(!S.fbs||g.fb)&&(!S.q||(nm(g.h,g.hn)+' '+nm(g.a,g.an)).toLowerCase().indexOf(S.q)>=0)}
function pastBoard(k){var all=PW[k]||[],r=all.filter(match);r.sort(sorter());$('#bh').innerHTML=HP;
var cnt=function(f,col){var w=0,l=0;col=col||'r';all.forEach(function(g){if((!S.fbs||g.fb)&&f(g)){if(g[col]=='W')w++;else if(g[col]=='L')l++}});return w+l?w+'–'+l:'0–0'};
var ddr=function(){var w=0,l=0;all.forEach(function(g){if(g.dd&&(!S.fbs||g.fb)){var c=g.hs-g.as+g.line;if(c>0)w++;else if(c<0)l++}});return w+l?' Defense Dogs covered '+w+'–'+l+'.':''};
$('#wsum').hidden=false;$('#wsum').innerHTML='<b>'+wlab(k)+'</b>'+(S.fbs?', FBS vs FBS':'')+', against the closing line: Both agree '+cnt(function(g){return g.ba})+'; Strong '+cnt(function(g){return g.st})+'; Models split '+cnt(function(g){return g.sp})+' (predictive side); predictive 3+ edges '+cnt(function(g){return g.e!=null&&Math.abs(g.e)>=3&&Math.abs(g.line)<28})+'; reactive 2+ edges '+cnt(function(g){return g.er!=null&&Math.abs(g.er)>=2},'rr')+'; every predictive pick '+cnt(function(){return true})+'.'+ddr();
if(!r.length){$('#board').innerHTML='<tr><td class="empty" colspan="11">No games match. Clear the search or untick "FBS vs FBS only".</td></tr>';return}
$('#board').innerHTML=r.map(function(g){var hs=esc(nm(g.h,g.hn)),as=esc(nm(g.a,g.an)),pk=g.e==null||Math.abs(g.e)<.05?null:(g.e>0?g.h:g.a);
return'<tr class="'+(g.sig||'')+'">'+mu(g,hs,as,'<span class="ms">'+g.as+'–'+g.hs+(g.r?' <span class="pill '+g.r+'">'+RW[g.r]+'</span>':'')+(g.sig?'<br>'+pills(g):'')+'</span>')
+'<td><span class="big">'+g.as+'–'+g.hs+'</span></td><td><span class="big">'+ml(g,hs,as)+'</span></td><td>'+spr(g.line,hs,as)+'</td>'
+'<td>'+edg(g.e,5)+'</td><td>'+edg(g.er,1.5)+'</td><td class="l">'+(pk==null?'—':'<span class="tm">'+logo(pk)+esc(nm(pk))+'</span>')+'</td>'
+'<td class="l">'+(g.r?'<span class="pill '+g.r+'">'+RW[g.r]+'</span>':'—')+'</td><td class="l">'+pills(g)+'</td><td>'+fat(g)+'</td><td class="l">'+kt(g)+'</td></tr>'}).join('')}
function board(){if(S.scope.slice(0,2)=='p:')return pastBoard(S.scope.slice(2));$('#wsum').hidden=true;$('#bh').innerHTML=HU;
var r=D.games.filter(function(g){return(S.scope=='all'||g.tw)&&match(g)});r.sort(sorter());
if(!r.length){$('#board').innerHTML='<tr><td class="empty" colspan="16">No games match. Try "All upcoming" or clear the search.</td></tr>';return}
$('#board').innerHTML=r.map(function(g){var hs=esc(nm(g.h,g.hn)),as=esc(nm(g.a,g.an)),mt=ml(g,hs,as),
lt=g.e==null||Math.abs(g.e)<.05?null:(g.e>0?g.h:g.a),lx=lv(g),ls=lx&&lx.s!='pre'?'<br>'+lx.as+'–'+lx.hs+' '+(lx.s=='in'?'<span class="live">'+esc(lx.t)+'</span>':'Final'):'';
return'<tr class="'+(g.sig||'')+'">'+mu(g,hs,as,'<span class="ms">Model '+mt+(g.e!=null?', edge '+Math.abs(g.e).toFixed(1):'')+(g.sig||g.dd?'<br>'+pills(g):'')+ls+'</span>'+(g.adj?'<span class="sub" style="text-align:center">includes adjustment</span>':''))
+'<td>'+score(g)+'</td><td><span class="big">'+mt+'</span></td><td>'+as+' '+Math.round((g.t-g.m)/2)+', '+hs+' '+Math.round((g.t+g.m)/2)+'</td><td>'+g.wp+'%</td>'
+'<td><span class="big">'+spr(g.line,hs,as)+'</span>'+(g.line!=null?'<span class="sub">'+esc(g.bk||'')+'</span>':'')+'</td>'
+'<td>'+edg(g.e,5)+'</td><td>'+edg(g.er,1.5)+'</td>'
+'<td class="l">'+(lt==null?'—':'<span class="tm">'+logo(lt)+esc(nm(lt))+'</span>')+'</td><td class="l">'+pills(g)+'</td><td>'+mv(g)+'</td><td>'+fat(g)+'</td>'
+'<td>'+g.t.toFixed(1)+' / '+(g.bt==null?'—':(+g.bt).toFixed(1))+'</td><td>'+as+' '+f1(g.aa)+'<span class="sub">'+hs+' '+f1(g.ha)+'</span></td>'
+'<td>'+(g.n?'Neutral':sg(g.he,.5,-.5))+'</td><td class="l">'+kt(g)+'</td></tr>'}).join('')}
function ratings(){var r=D.ratings;if(S.rq)r=r.filter(function(t){return nm(t.id,t.tn).toLowerCase().indexOf(S.rq)>=0});else if(S.rshow=='25')r=r.slice(0,25);
$('#rt').innerHTML=r.map(function(t){return'<tr><td>'+t.rk+'</td><td class="l st"><span class="tm">'+logo(t.id)+'<b>'+esc(nm(t.id,t.tn))+'</b></span></td><td class="l">'+t.rec+'</td><td><b>'+t.pw.toFixed(1)+'</b></td><td>'+t.o.toFixed(1)+'</td><td>'+t.df.toFixed(1)+'</td>'
+'<td>'+(t.epa==null?'—':(t.epa>0?'+':'')+t.epa.toFixed(3))+'</td><td>'+(t.pre||'—')+'</td><td>'+sg(t.woe,.5,-.5)+'</td><td>'+(t.to==null?'—':(t.to>0?'+':'')+t.to.toFixed(2))+'</td><td>'+(t.rr==null?'—':Math.round(100*t.rr)+'%')+'</td>'
+'<td>'+(t.la==null?'—':'<span class="'+(t.la<=-21?'pos':t.la>=21?'neg':'')+'">'+f1(t.la)+'</span>')+'</td><td>'+sg(t.mv,.5,-.5)+'</td><td>'+sg(t.he,.5,-.5)+'</td><td>'+(t.adj?sg(t.adj,0,0):'—')+'</td></tr>'}).join('')||'<tr><td class="empty" colspan="15">No team matches that search.</td></tr>'}
function record(){var B=function(b){return D.summary.filter(function(r){return r.bucket==b})[0]||{}},rec=function(r){return r.ats_w==null?'—':r.ats_w+'–'+r.ats_l},
pr=function(r){return r.ats_w==null||r.ats_w+r.ats_l==0?'—':pct(r.ats_w/(r.ats_w+r.ats_l))},
tile=function(k,v,s){return'<div class="tile"><div class="k">'+k+'</div><div class="v">'+v+'</div><div class="k">'+s+'</div></div>'},a=B('all'),ba=B('both'),st=B('strong'),p3=B('pred_3'),r2=B('react_2'),sp=B('split'),lb=B('last_week_both');
$('#tiles').innerHTML=tile(M.season+' Both agree',rec(ba),pr(ba)+' vs closing lines, 52.4% breaks even')+tile('Strong spots',rec(st),pr(st)+' this season')+tile('Predictive 3+ edges',rec(p3),pr(p3)+' this season')
+tile('Reactive 2+ edges',rec(r2),pr(r2)+' this season')+tile('Models split (predictive side)',rec(sp),pr(sp)+' this season')+tile('Average miss on the margin',(+a.mae_model||0).toFixed(2)+' pts','Predictive model; Vegas closing line: '+(+a.mae_vegas||0).toFixed(2));
var r3=function(w,l){var p=w/(w+l);return w+l?'<span class="'+(p>=.524?'pos':'')+'">'+pct(p)+' ('+w+'–'+l+')</span>':'—'};
var cur=a.ats_w==null?[]:[[M.season+' (to date)',a.games,+a.mae_model,+a.mae_vegas,ba.ats_w,ba.ats_l,st.ats_w,st.ats_l,p3.ats_w,p3.ats_l,r2.ats_w,r2.ats_l,sp.ats_w,sp.ats_l]];
$('#bt').innerHTML=cur.concat(D.backtest).map(function(b){var tot=/total/.test(b[0]);return'<tr'+(tot?' style="font-weight:600"':'')+'><td class="l">'+esc(b[0])+'</td><td>'+b[1]+'</td><td>'+b[2].toFixed(2)+'</td><td>'+b[3].toFixed(2)+'</td><td><b>'+r3(b[4],b[5])+'</b></td><td>'+r3(b[6],b[7])+'</td><td>'+r3(b[8],b[9])+'</td><td>'+r3(b[10],b[11])+'</td><td>'+r3(b[12],b[13])+'</td></tr>'}).join('')}
document.querySelectorAll('nav button').forEach(function(b){b.onclick=function(){document.querySelectorAll('nav button').forEach(function(x){x.setAttribute('aria-selected',x==b)});document.querySelectorAll('main>section').forEach(function(s){s.hidden=s.id!=b.dataset.t})}});
$('#scope').onchange=function(){S.scope=this.value;board()};$('#sort').onchange=function(){S.sort=this.value;board()};$('#q').oninput=function(){S.q=this.value.trim().toLowerCase();board()};
$('#fbs').onchange=function(){S.fbs=this.checked;board()};$('#rshow').onchange=function(){S.rshow=this.value;ratings()};$('#rq').oninput=function(){S.rq=this.value.trim().toLowerCase();ratings()};
$('#wk').textContent=M.week?'Week '+M.week:'';
$('#stamp').textContent='Model run '+M.run_at+'. Results through '+M.data_through+'. '+(M.n_lines?M.book+' lines for '+M.n_lines+' games, last pulled '+M.lines_at+' (a game keeps its last line from before kickoff).':'No sportsbook lines were available when this page was built.');
$('#gnote').innerHTML='Edges use the '+esc(M.book||'sportsbook')+' spread pulled when this page was built. Predictive edge = how far the predictive model is from the line; reactive edge = how far the reactive model moves off it. <b>Both agree</b> = predictive 5+ and reactive 1.5+ on the same side. <b>Strong</b> = a predictive 3+ edge after a market overreaction. <b>Models split</b> = predictive 5+ but the reactive model leans the other way.'+(M.adj&&M.adj.length?' Team adjustments included: '+M.adj.map(function(a){return esc(a[0])+' '+f1(a[1])+(a[2]?' ('+esc(a[2])+')':'')}).join(', ')+'.':'');
var LV=M.live||[],fails=0,tm=null;
function poll(){clearTimeout(tm);if(!LV.length||fails>=3||!window.fetch||document.hidden)return;
Promise.all(LV.map(function(s){return fetch('https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard?dates='+M.season+'&seasontype='+s[0]+'&week='+s[1]+'&groups=80&limit=400').then(function(r){if(!r.ok)throw r.status;return r.json()})}))
.then(function(js){fails=0;var on=false,soon=false,now=Date.now();js.forEach(function(j){(j.events||[]).forEach(function(ev){var c=(ev.competitions||[])[0];if(!c)return;var st=((ev.status||c.status||{}).type)||{},sc={};
(c.competitors||[]).forEach(function(t){sc[t.homeAway]=t.score});var od=(c.odds||[])[0]||{},ps=((od.pointSpread||{}).home)||{},q=(ps.current||ps.close||ps.open||{}).line,ln=q==null?null:(String(q).toUpperCase()=='EVEN'||String(q).toUpperCase()=='PK'?0:parseFloat(q));
if(ln==null&&od.spread!=null){var hf=(od.homeTeamOdds||{}).favorite,af=(od.awayTeamOdds||{}).favorite;if(hf===true)ln=-Math.abs(od.spread);else if(af===true)ln=Math.abs(od.spread)}
L[ev.id]={s:st.state,t:st.shortDetail||'',hs:+sc.home||0,as:+sc.away||0,ln:isNaN(ln)?null:ln};if(st.state=='in')on=true;if(st.state=='pre'&&new Date(ev.date)-now<3600000)soon=true})});
if(S.scope=='week'||S.scope=='all')board();tm=setTimeout(poll,on||soon?60000:600000)})
.catch(function(){fails++;tm=setTimeout(poll,120000)})}
document.addEventListener('visibilitychange',function(){if(!document.hidden)poll()});
board();ratings();record();poll()})();
</script></body></html>
'''

# ---------------- weekly run ----------------
def main(out_dir, season=None):
    now = pd.Timestamp.now(tz='UTC')
    Y = season or (now.year if now.month >= 7 else now.year - 1)
    os.makedirs(out_dir, exist_ok=True)
    code = open(os.path.abspath(__file__)).read()
    meta = [{'label': 'cfb_weekly.py', 'value': code, 'type': 'code', 'lang': 'python'},
            {'label': 'Input', 'value': 'sportsdataverse-data releases (ESPN schedules, box scores, advanced team stats, closing lines, talent, returning production, team summaries) and cfbfastR-data schedules CSV (upcoming games)'},
            {'label': 'Run at', 'value': now.strftime('%Y-%m-%d %H:%M UTC')}]
    def dump(name, rows):
        rows = json.loads(pd.DataFrame(rows).replace({np.nan: None}).to_json(orient='records'))
        json.dump({'rows': rows, 'meta': meta}, open(os.path.join(out_dir, name), 'w'))
        print(name, len(rows))

    R_prev, _ = season_end(Y - 1)
    prior = prior_from_inputs(prior_inputs(Y, R_prev, refresh=True), PRIOR_COEF, P['lam'], R_prev)
    fb = fbs_teams(Y, refresh=True)
    he = team_home_edge(Y, fb.index)
    g = completed_games(Y, refresh=True); g['vegas'] = g.home_team_spread
    a = adv_stats(Y, refresh=True)
    tot = g[['h_rush_att', 'a_rush_att', 'h_pass_att', 'a_pass_att']].astype(float).sum()
    lg_rush = (tot.h_rush_att + tot.a_rush_att) / tot.sum() if tot.sum() > 0 else 0.5
    SRA = stat_rows(g, a)
    past, R, mu, SR, f, la = state_at(g, prior, SRA, now)
    lw_t = (g.date.max() if len(g) else now) - pd.Timedelta(days=6)
    _, R_lw, _, SR_lw, f_lw, _ = state_at(g, prior, SRA, lw_t)

    # ---- ratings.json
    T = fb.copy()
    T['off'] = R.o.reindex(T.index); T['def'] = R.d.reindex(T.index); T['net'] = T['off'] + T['def']
    T['pre_net'] = (prior.po + prior.pd).reindex(T.index)
    T['core_power'] = team_power(R, SR, f, lg_rush, T.index)
    T['core_power_lw'] = team_power(R_lw, SR_lw, f_lw, lg_rush, T.index)
    T['epa_net'] = SR['EPA_per_play'].reindex(T.index); T['ypp_net'] = SR['yards_per_play'].reindex(T.index)
    for c in ['gp', 'wins', 'exp_wins', 'wins_over_exp', 'to_m', 'rush_rate']:
        T[c] = f[c].reindex(T.index) if f is not None else np.nan
    T['losses'] = T.gp - T.wins
    ok = T.gp.fillna(0) >= MIN_GP
    T['f_out'] = np.where(ok, T.wins_over_exp, 0.0); T['f_to'] = np.where(ok, T.to_m, 0.0)
    T['f_rush'] = np.where(ok, (T.rush_rate - lg_rush) * 100, 0.0)
    T['last_ats'] = la.reindex(T.index) if len(la) else np.nan
    T['react_pts'] = react(T.last_ats.fillna(0))
    T['home_edge'] = he.reindex(T.index).fillna(0.0)
    T['power'] = T.core_power + W['w_out']*T.f_out + W['w_to']*T.f_to + W['w_rush']*T.f_rush   # team strength (Vegas reaction is game-level, kept out)
    T['rank'] = T.power.rank(ascending=False, method='min').astype(int)
    T['rank_pre'] = T.pre_net.rank(ascending=False, method='min').astype(int)
    T = T.reset_index().rename(columns={'index': 'team_id'}).sort_values('rank')
    T[T.select_dtypes('number').columns] = T.select_dtypes('number').round(3)
    dump('ratings.json', T)

    # ---- games.json (upcoming)
    U = upcoming_games(Y, set(g.game_id))
    U = U[U.home_id.isin(fb.index) | U.away_id.isin(fb.index)].copy()
    U = matchup(U, R, mu, SR, f, la, lg_rush, prior, he)
    U['h_def_rank'] = R.d.reindex(fb.index).rank(ascending=False, method='min').reindex(U.home_id).values
    U['model_total'] = U.base_home + U.base_away
    U['home_win_prob'] = N.cdf(U.model_margin / P['sigma'])
    nxt = U.week.min() if len(U) else None
    U['this_week'] = U.week == nxt
    U['fbs_both'] = U.home_id.isin(fb.index) & U.away_id.isin(fb.index)
    keep = ['game_id', 'season', 'week', 'season_type', 'date', 'neutral_site', 'home_id', 'home_team', 'away_id', 'away_team',
            'core', 'f_out', 'f_to', 'f_rush', 'h_ats1', 'a_ats1', 'rx', 'home_edge', 'model_margin', 'model_total', 'home_win_prob', 'this_week', 'fbs_both', 'h_def_rank', 'adj']
    U['date'] = U.date.dt.strftime('%Y-%m-%dT%H:%MZ')
    U = U[keep].copy(); U[U.select_dtypes('number').columns] = U.select_dtypes('number').round(3)
    U['game_id'] = U.game_id.astype(str)
    dump('games.json', U)

    # ---- this season's out-of-sample record (every game predicted from earlier games only)
    gt, pt, at_ = g, prior, a
    if len(g) == 0:
        Rp2 = season_end(Y - 2)[0]
        gt = completed_games(Y - 1); gt['vegas'] = gt.home_team_spread
        pt = prior_from_inputs(prior_inputs(Y - 1, Rp2), PRIOR_COEF, P['lam'], Rp2); at_ = adv_stats(Y - 1)
        print('no games yet this season; track/summary show last season')
        he_t = team_home_edge(Y - 1, fbs_teams(Y - 1).index)
    else:
        he_t = he
    wf = walk_forward(gt, pt, at_, he_t, fbs_teams(gt.season.iloc[0]).index if len(g) == 0 else fb.index)
    wf = wf[wf.home_id.isin(fb.index) | wf.away_id.isin(fb.index)].copy()
    wf['fbs_both'] = wf.home_id.isin(fb.index) & wf.away_id.isin(fb.index)
    wf['ep'] = wf.model_margin + wf.vegas                # predictive edge, >0: likes home vs the line
    wf['er'] = wf.adj.where(wf.vegas.notna())             # reactive edge
    wf['ats_margin'] = wf.margin + wf.vegas               # >0: home covered
    res = lambda e: np.where(wf.vegas.isna() | (wf.ats_margin == 0) | e.isna() | (e.abs() < 0.05), None,
                             np.where(np.sign(e) == np.sign(wf.ats_margin), 'W', 'L'))
    wf['pred_ats'] = res(wf.ep); wf['react_ats'] = res(wf.er)
    pk = np.where(wf.ep > 0, wf.h_ats1, wf.a_ats1); op = np.where(wf.ep > 0, wf.a_ats1, wf.h_ats1)
    wf['strong'] = (wf.ep.abs() >= 3) & (wf.vegas.abs() < 28) & (np.fmax(-np.nan_to_num(pk), np.nan_to_num(op)) >= STRONG_RS)
    big = (wf.ep.abs() >= BOTH_P) & (wf.er.abs() >= BOTH_R)
    wf['both'] = big & (np.sign(wf.ep) == np.sign(wf.er)); wf['split'] = big & (np.sign(wf.ep) != np.sign(wf.er))
    wf['su_correct'] = (np.sign(wf.model_margin) == np.sign(wf.margin)).astype(int)
    wf['abs_err_model'] = (wf.margin - wf.model_margin).abs(); wf['abs_err_vegas'] = (wf.margin + wf.vegas).abs()

    # ---- results.json: this season's finished games as the model saw them beforehand (for the page's past-weeks view)
    wf['model_total'] = wf.base_home + wf.base_away
    rc = ['game_id', 'season_type', 'week', 'date', 'neutral_site', 'home_id', 'away_id', 'home_team', 'away_team', 'home_score', 'away_score',
          'vegas', 'over_under', 'model_total', 'h_def_rank', 'adj', 'core', 'f_out', 'f_to', 'f_rush', 'rx', 'home_edge', 'h_ats1', 'a_ats1', 'fbs_both']
    rs = wf[rc].copy() if len(g) else pd.DataFrame(columns=rc)
    if len(rs):
        rs['date'] = rs.date.dt.strftime('%Y-%m-%dT%H:%MZ'); rs['game_id'] = rs.game_id.astype(str)
        rs[rs.select_dtypes('number').columns] = rs.select_dtypes('number').round(3)
    dump('results.json', rs)

    # ---- summary.json (one row per bucket; 'all' first)
    S = []
    v = wf[wf.fbs_both].dropna(subset=['vegas'])   # graded on FBS-vs-FBS games
    def summ(label, d, col='pred_ats'):
        dd = d[d[col].isin(['W', 'L'])]
        return dict(bucket=label, games=len(d), ats_w=int((dd[col] == 'W').sum()), ats_l=int((dd[col] == 'L').sum()),
                    ats_pct=round((dd[col] == 'W').mean(), 3) if len(dd) else None,
                    mae_model=round(d.abs_err_model.mean(), 2) if len(d) else None, mae_vegas=round(d.abs_err_vegas.mean(), 2) if len(d) else None,
                    su_pct=round(d.su_correct.mean(), 3) if len(d) else None)
    S.append(summ('all', v)); S.append(summ('both', v[v.both])); S.append(summ('strong', v[v.strong]))
    S.append(summ('pred_3', v[(v.ep.abs() >= 3) & (v.vegas.abs() < 28)])); S.append(summ('react_2', v[v.er.abs() >= LEAN], 'react_ats'))
    S.append(summ('split', v[v.split]))
    lw = v[v.date >= v.date.max() - pd.Timedelta(days=6)]
    S.append(summ('last_week', lw)); S.append(summ('last_week_both', lw[lw.both])); S.append(summ('last_week_strong', lw[lw.strong]))
    s = pd.DataFrame(S)
    s['season'] = int(gt.season.iloc[0]) if len(gt) else Y
    s['data_through'] = gt.date.max().strftime('%Y-%m-%d'); s['run_at'] = now.strftime('%Y-%m-%d %H:%M UTC')
    s['next_week'] = None if nxt is None or pd.isna(nxt) else int(nxt)
    s['hfa'] = P['H']; s['sigma'] = P['sigma']
    s['home_edge_sd'] = round(float(he.reindex(fb.index).std()), 2)
    dump('summary.json', s)
    if os.environ.get('CFB_FAN') and len(U):
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import fan_sentiment
            tw = U[U.this_week]
            for st, d in tw.groupby(tw.season_type.astype(str).str.lower().str.startswith('post')):
                ids = sorted((set(d.home_id) | set(d.away_id)) & set(fb.index))
                fan_sentiment.week(Y, f"{3 if st else 2}-{int(d.week.min())}", now.tz_convert('America/Chicago').date().isoformat(), ids)
        except Exception as e:
            print('fan attitude skipped:', e)
    write_html(out_dir, Y)

if __name__ == '__main__':
    args = sys.argv[1:]
    season = int(args[args.index('--season') + 1]) if '--season' in args else None
    main(args[0] if args else 'out', season)
