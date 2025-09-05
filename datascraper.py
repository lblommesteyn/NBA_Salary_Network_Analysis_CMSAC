# datascraper.py — frequent CSV checkpoints, robust roster/player parsing (td/th),
#                   salaries from team-year (salaries/salaries2) with y{year} support,
#                   contracts fallback, and detailed logging.

import os
import sys
import re
import time
import random
import subprocess
import argparse
from datetime import datetime
from time import perf_counter

# Ensure deps
REQUIRED = ["requests", "beautifulsoup4", "pandas", "numpy", "lxml", "urllib3"]
for pkg in REQUIRED:
    try:
        __import__(pkg if pkg != "beautifulsoup4" else "bs4")
    except ImportError:
        print(f"Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

import logging
from logging.handlers import RotatingFileHandler

import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup, Comment
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import http.client as http_client


# ---------- Logging ----------
def setup_logging(level_str="INFO", log_dir="logs", log_name="datascraper.log"):
    os.makedirs(log_dir, exist_ok=True)
    level = getattr(logging, level_str.upper(), logging.INFO)

    logger = logging.getLogger("nba_scraper")
    logger.setLevel(level)

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)

    fh = RotatingFileHandler(os.path.join(log_dir, log_name),
                             maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)

    logger.handlers = []
    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.propagate = False

    # urllib3 wire logs when DEBUG
    for name in ("urllib3", "urllib3.connectionpool"):
        ulog = logging.getLogger(name)
        ulog.handlers = []
        ulog.addHandler(ch)
        ulog.addHandler(fh)
        ulog.setLevel(level if level <= logging.DEBUG else logging.INFO)
        ulog.propagate = False

    http_client.HTTPConnection.debuglevel = 1 if level <= logging.DEBUG else 0
    return logger


# ---------- Collector ----------
class NBADataCollector:
    def __init__(self, logger, base_delay=6, jitter=3,
                 max_connect_read_retries=3, max_retry_after=7200,
                 output_dir="nba_data", checkpoint_mode="team"):
        self.log = logger
        self.base_url_bref = "https://www.basketball-reference.com"
        self.headers = {
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/124.0.0.0 Safari/537.36"),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Connection": "close",
        }
        self.teams = [
            'ATL', 'BOS', 'BRK', 'CHO', 'CHI', 'CLE', 'DAL', 'DEN', 'DET',
            'GSW', 'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN',
            'NOP', 'NYK', 'OKC', 'ORL', 'PHI', 'PHO', 'POR', 'SAC', 'SAS',
            'TOR', 'UTA', 'WAS'
        ]
        self.base_delay = base_delay
        self.jitter = jitter
        self.session = self._create_session(max_connect_read_retries)
        self.domain_cooldown_until = 0.0
        self.max_retry_after = max_retry_after
        self.output_dir = output_dir
        self.checkpoint_mode = checkpoint_mode

        self.schemas = {
            'salaries': ['player','salary','team','year'],
            'team_stats': [
                'team','year',
                'reg_games','reg_wins','reg_losses','reg_win_pct',
                'playoff_games','playoff_wins','playoff_losses',
                'total_games',
                'pts_per_game','opp_pts_per_game','net_rating','srs'
            ],
            'player_stats': ['player','team','year','age','games','games_started','minutes','minutes_per_game'],
            'network_metrics': [
                'team','year',
                'gini_coefficient','top_1_share','top_3_share','top_5_share',
                'salary_mean','salary_std','salary_cv','num_players'
            ],
            'coaches': ['coach','seasons','games','wins','losses','win_pct'],
        }

        self.files = {
            'salaries': f'{self.output_dir}/nba_salaries.csv',
            'team_stats': f'{self.output_dir}/nba_team_stats.csv',
            'player_stats': f'{self.output_dir}/nba_player_stats.csv',
            'network_metrics': f'{self.output_dir}/nba_network_metrics.csv',
            'coaches': f'{self.output_dir}/nba_coaches.csv',
            'merged': f'{self.output_dir}/nba_complete_dataset.csv',
        }

    # ----- HTTP -----
    def _create_session(self, max_connect_read_retries):
        retry = Retry(
            total=None,
            connect=max_connect_read_retries,
            read=max_connect_read_retries,
            redirect=max_connect_read_retries,
            backoff_factor=2.0,
            allowed_methods=["GET", "HEAD", "OPTIONS"],
            raise_on_status=False,
            respect_retry_after_header=False,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_maxsize=10)
        s = requests.Session()
        s.headers.update(self.headers)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        self.log.debug("HTTP session created (retries=%s backoff_factor=%.1f pool_maxsize=10)",
                       max_connect_read_retries, 2.0)
        return s

    def _polite_pause(self, why="request"):
        delay = self.base_delay + random.uniform(0, self.jitter)
        self.log.debug("Sleeping %.2fs after %s.", delay, why)
        time.sleep(delay)

    def _get(self, url, **kwargs):
        if time.time() < self.domain_cooldown_until:
            wait = self.domain_cooldown_until - time.time()
            self.log.warning("In domain cooldown for %.1fs before requesting %s", wait, url)
            time.sleep(wait)

        kwargs.setdefault("timeout", (10, 30))
        self.log.info("GET %s", url)
        t0 = perf_counter()
        try:
            resp = self.session.get(url, **kwargs)
        except requests.exceptions.RequestException as e:
            self.log.error("RequestException for %s: %s", url, repr(e))
            raise

        dt = perf_counter() - t0
        self.log.debug("Response status=%s in %.2fs (history=%s)",
                       resp.status_code, dt, [h.status_code for h in resp.history])

        if resp.status_code == 429:
            ra_raw = resp.headers.get("Retry-After")
            self.log.warning("429 Too Many Requests (Retry-After=%s) for %s", ra_raw, url)
            sleep_seconds = 600
            if ra_raw:
                try:
                    sleep_seconds = min(int(float(ra_raw)), self.max_retry_after)
                except Exception:
                    pass
            jitter = random.uniform(2, 8)
            sleep_total = sleep_seconds + jitter
            self.domain_cooldown_until = time.time() + sleep_total
            self.log.warning("Honoring Retry-After: sleeping %.1fs (cooldown until %s)",
                             sleep_total, datetime.fromtimestamp(self.domain_cooldown_until).isoformat(timespec='seconds'))
            time.sleep(sleep_total)
            self.log.info("Retrying after cooldown: %s", url)
            resp = self.session.get(url, **kwargs)
            if resp.status_code == 429:
                self.log.error("Still 429 after honoring Retry-After. Aborting %s.", url)
                resp.raise_for_status()

        resp.raise_for_status()
        self.log.debug("Headers: server=%s cf-ray=%s content-type=%s content-length=%s encoding=%s",
                       resp.headers.get("server"), resp.headers.get("cf-ray"),
                       resp.headers.get("content-type"), resp.headers.get("content-length"),
                       resp.encoding)
        if resp.cookies:
            self.log.debug("Set-Cookies: %s", resp.cookies.get_dict())

        self.log.debug("Received %d bytes from %s", len(resp.content or b""), resp.url)
        self._polite_pause("request")
        return resp

    # ----- helpers -----
    def _soup(self, content):
        return BeautifulSoup(content, 'lxml')

    def _find_table(self, soup, table_id):
        tbl = soup.find('table', id=table_id)
        if tbl:
            self.log.debug("Found table #%s directly.", table_id)
            return tbl
        for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
            if table_id in c:
                inner = BeautifulSoup(c, 'lxml').find('table', id=table_id)
                if inner:
                    self.log.debug("Found table #%s inside HTML comment.", table_id)
                    return inner
        self.log.debug("Table #%s not found.", table_id)
        return None

    def _find_table_near_anchor(self, soup, anchor_id):
        anchor = soup.find(id=anchor_id)
        if not anchor:
            return None
        for el in anchor.next_elements:
            if isinstance(el, Comment):
                inner = BeautifulSoup(el, 'lxml').find('table')
                if inner:
                    return inner
            n = getattr(el, 'name', None)
            if n == 'table':
                return el
            if n == 'div' and 'section_heading' in (el.get('class') or []):
                break
        return None

    def _safe_float(self, text):
        try:
            return float(str(text).replace(',', '').strip())
        except (ValueError, TypeError, AttributeError):
            return None

    def _safe_int(self, text):
        try:
            return int(float(str(text).replace(',', '').strip()))
        except (ValueError, TypeError, AttributeError):
            return None

    def _parse_money(self, txt: str):
        if not txt:
            return None
        cleaned = (txt.replace('\xa0', ' ')
                      .split('(')[0]
                      .replace('$', '')
                      .replace(',', '')
                      .strip())
        if not cleaned or cleaned in {'—', '-', '–'}:
            return None
        try:
            return int(float(cleaned))
        except ValueError:
            m = re.search(r'\d[\d\.]*', cleaned)
            if m:
                try:
                    return int(float(m.group(0)))
                except ValueError:
                    return None
            return None

    # ---------- Salaries ----------
    def get_team_salaries(self, team, year):
        # team-year first
        url = f"{self.base_url_bref}/teams/{team}/{year}.html"
        try:
            resp = self._get(url)
            soup = self._soup(resp.content)

            table = (self._find_table(soup, 'salaries')
                     or self._find_table(soup, 'salaries2')
                     or self._find_table(soup, 'payroll')
                     or self._find_table_near_anchor(soup, 'salaries2_sh')
                     or self._find_table_near_anchor(soup, 'salaries_sh'))

            out = []
            if table:
                rows = table.select('tbody tr')
                self.log.debug("Parsing %d salary rows (team-year) for %s %s", len(rows), team, year)
                for tr in rows:
                    name_cell = tr.find(['th','td'], {'data-stat': 'player'})
                    if not name_cell:
                        continue
                    # Prefer actual player rows
                    if name_cell.get('data-append-csv') is None:
                        a = name_cell.find('a') or tr.find('a')
                        if not a or '/players/' not in a.get('href', ''):
                            continue
                    name = name_cell.get_text(strip=True)

                    # 1) salary cell
                    td = tr.find('td', {'data-stat': 'salary'})
                    # 2) or year-specific column (y{year})
                    if not td:
                        td = tr.find('td', {'data-stat': f'y{year}'})
                    # 3) or any $ cell
                    if not td:
                        for tcell in tr.find_all('td'):
                            if '$' in tcell.get_text():
                                td = tcell
                                break
                    if not td:
                        continue

                    val = self._parse_money(td.get_text(" ", strip=True))
                    if val is None:
                        continue
                    out.append({'player': name, 'salary': val, 'team': team, 'year': year})

            if out:
                self.log.info("Salaries: %s %s -> %d players (team-year page)", team, year, len(out))
                return out
            else:
                self.log.warning("No salaries on team-year page for %s %s; falling back to contracts.", team, year)

        except requests.RequestException as e:
            self.log.warning("Error fetching team-year salaries for %s %s: %s", team, year, e)

        # contracts fallback
        return self._get_team_salaries_from_contracts(team, year)

    def _get_team_salaries_from_contracts(self, team, year):
        url = f"{self.base_url_bref}/contracts/{team}.html"
        try:
            resp = self._get(url)
            soup = self._soup(resp.content)
            table = self._find_table(soup, 'contracts') or soup.find('table')
            if not table:
                self.log.warning("No contracts table for %s %s", team, year)
                return []

            out = []
            season_label = f"{year-1}-{str(year)[-2:]}"   # "2020-21"
            alt_label    = f"{year-1}-{year}"

            # header index if possible
            season_col_idx = None
            thead = table.find('thead')
            if thead:
                for tr in thead.find_all('tr')[::-1]:
                    ths = tr.find_all(['th','td'])
                    for idx, th in enumerate(ths):
                        txt = th.get_text(strip=True).replace('–','-')
                        if txt in (season_label, alt_label):
                            season_col_idx = idx
                            break
                    if season_col_idx is not None:
                        break

            rows = table.select('tbody tr')
            for tr in rows:
                name_cell = tr.find(['th','td'], {'data-stat': 'player'})
                if not name_cell:
                    continue
                if name_cell.get('data-append-csv') is None:
                    a = name_cell.find('a') or tr.find('a')
                    if not a or '/players/' not in a.get('href',''):
                        continue
                name = name_cell.get_text(strip=True)

                salary_val = None
                if season_col_idx is not None:
                    cells = tr.find_all(['th','td'])
                    if season_col_idx < len(cells):
                        salary_val = self._parse_money(cells[season_col_idx].get_text(" ", strip=True))

                if salary_val is None:
                    td = tr.find('td', {'data-stat': f'y{year}'})
                    if td:
                        salary_val = self._parse_money(td.get_text(" ", strip=True))

                if salary_val is None:
                    # last resort: any $ cell
                    for tcell in tr.find_all('td'):
                        if '$' in tcell.get_text():
                            salary_val = self._parse_money(tcell.get_text(" ", strip=True))
                            if salary_val is not None:
                                break

                if salary_val is not None:
                    out.append({'player': name, 'salary': salary_val, 'team': team, 'year': year})

            self.log.info("Salaries: %s %s -> %d players (contracts page%s)",
                          team, year, len(out),
                          " header-index" if season_col_idx is not None else " fallback")
            if len(out) == 0:
                self.log.warning("0 salary values captured from contracts for %s %s", team, year)
            return out

        except requests.RequestException as e:
            self.log.warning("Error fetching contracts for %s %s: %s", team, year, e)
            return []

    # ---------- Team stats ----------
    def get_team_stats(self, year):
        url_season = f"{self.base_url_bref}/leagues/NBA_{year}.html"
        reg_wl_map, pts_map, opp_map, srs_map = {}, {}, {}, {}
        try:
            resp = self._get(url_season)
            soup = self._soup(resp.content)

            # standings (reg season)
            for tid in ['divs_standings_E', 'divs_standings_W', 'expanded_standings']:
                tbl = self._find_table(soup, tid)
                if not tbl:
                    continue
                for row in tbl.select('tbody tr'):
                    a = row.find('a')
                    if not a or '/teams/' not in a.get('href', ''):
                        continue
                    abbr = a['href'].split('/')[2]
                    def g(stat):
                        td = row.find('td', {'data-stat': stat})
                        return self._safe_float(td.get_text()) if td else None
                    wins = g('won') or g('wins')
                    losses = g('lost') or g('losses')
                    win_pct = g('win_loss_pct')
                    if wins is None and losses is None and win_pct is None:
                        continue
                    reg_wl_map.setdefault(abbr, {})
                    if wins is not None: reg_wl_map[abbr]['wins'] = int(wins)
                    if losses is not None: reg_wl_map[abbr]['losses'] = int(losses)
                    if win_pct is not None: reg_wl_map[abbr]['win_pct'] = float(win_pct)

            # team/offense per-game
            for tid in ['team-stats-per_game', 'per_game-team', 'team_per_game', 'team-opponent-per_game']:
                tbl = self._find_table(soup, tid)
                if not tbl:
                    continue
                for row in tbl.select('tbody tr'):
                    a = row.find('a')
                    if not a or '/teams/' not in a.get('href', ''):
                        continue
                    abbr = a['href'].split('/')[2]
                    td = row.find('td', {'data-stat': 'pts_per_g'}) or row.find('td', {'data-stat': 'pts'})
                    if td:
                        v = self._safe_float(td.get_text())
                        if v is not None:
                            pts_map[abbr] = v

            # opponent per-game
            for tid in ['opponent-stats-per_game', 'per_game-opponent', 'team-opponent-per_game']:
                tbl = self._find_table(soup, tid)
                if not tbl:
                    continue
                for row in tbl.select('tbody tr'):
                    a = row.find('a')
                    if not a or '/teams/' not in a.get('href', ''):
                        continue
                    abbr = a['href'].split('/')[2]
                    td = row.find('td', {'data-stat': 'pts_per_g'}) or row.find('td', {'data-stat': 'opp_pts'})
                    if td:
                        v = self._safe_float(td.get_text())
                        if v is not None:
                            opp_map[abbr] = v

            # SRS
            for tid in ['misc_stats', 'advanced-team', 'team-stats-advanced']:
                tbl = self._find_table(soup, tid)
                if not tbl:
                    continue
                for row in tbl.select('tbody tr'):
                    a = row.find('a')
                    if not a or '/teams/' not in a.get('href', ''):
                        continue
                    abbr = a['href'].split('/')[2]
                    srs_td = row.find('td', {'data-stat': 'srs'})
                    if srs_td:
                        v = self._safe_float(srs_td.get_text())
                        if v is not None:
                            srs_map[abbr] = v

        except requests.RequestException as e:
            self.log.warning("Error fetching season page for %s: %s", year, e)

        # playoff record
        playoff_wl = self._get_playoff_record(year)

        teams = set(reg_wl_map) | set(pts_map) | set(opp_map) | set(srs_map) | set(playoff_wl)
        out = []
        for abbr in sorted(teams):
            rw = reg_wl_map.get(abbr, {}).get('wins')
            rl = reg_wl_map.get(abbr, {}).get('losses')
            rwp = reg_wl_map.get(abbr, {}).get('win_pct')
            reg_games = (rw + rl) if (rw is not None and rl is not None) else None

            pw = playoff_wl.get(abbr, {}).get('wins', 0)
            pl = playoff_wl.get(abbr, {}).get('losses', 0)
            playoff_games = pw + pl if (pw is not None and pl is not None) else 0

            total_games = (reg_games or 0) + playoff_games if reg_games is not None else None

            pts = pts_map.get(abbr)
            opp = opp_map.get(abbr)
            net = (pts - opp) if (pts is not None and opp is not None) else None

            out.append({
                'team': abbr,
                'year': year,
                'reg_games': reg_games,
                'reg_wins': rw,
                'reg_losses': rl,
                'reg_win_pct': rwp,
                'playoff_games': playoff_games,
                'playoff_wins': pw,
                'playoff_losses': pl,
                'total_games': total_games,
                'pts_per_game': pts,
                'opp_pts_per_game': opp,
                'net_rating': net,
                'srs': srs_map.get(abbr)
            })
        self.log.info("Team stats (robust): %s -> %d teams", year, len(out))
        return out

    def _get_playoff_record(self, year):
        url = f"{self.base_url_bref}/playoffs/NBA_{year}.html"
        out = {}
        try:
            resp = self._get(url)
            soup = self._soup(resp.content)
            for el in soup.find_all(['p','li']):
                text = el.get_text(" ", strip=True)
                if not text:
                    continue
                m = re.search(r'\((\d+)\s*-\s*(\d+)\)', text)
                if not m:
                    continue
                a_wins, b_wins = int(m.group(1)), int(m.group(2))
                if a_wins != 4 and b_wins != 4:
                    continue
                links = el.find_all('a', href=re.compile(rf'^/teams/[A-Z]{{3}}/{year}\.html$'))
                if len(links) < 2:
                    continue
                t1, t2 = links[0]['href'].split('/')[2], links[1]['href'].split('/')[2]
                # winner heuristic
                low = text.lower()
                if ' over ' in low or 'defeated' in low or ' beat ' in low or ' beats ' in low:
                    winner, loser = t1, t2
                    w_wins, l_wins = a_wins, b_wins
                else:
                    b1 = links[0].find_parent(['strong','b']) is not None
                    b2 = links[1].find_parent(['strong','b']) is not None
                    if b1 and not b2:
                        winner, loser = t1, t2; w_wins, l_wins = a_wins, b_wins
                    elif b2 and not b1:
                        winner, loser = t2, t1; w_wins, l_wins = b_wins, a_wins
                    else:
                        continue
                out.setdefault(winner, {'wins': 0, 'losses': 0})
                out.setdefault(loser,  {'wins': 0, 'losses': 0})
                out[winner]['wins']  += w_wins; out[winner]['losses'] += l_wins
                out[loser]['wins']   += l_wins;  out[loser]['losses']  += w_wins
        except requests.RequestException as e:
            self.log.warning("Error fetching playoffs page for %s: %s", year, e)
        return out

    # ---------- Player stats ----------
    def get_player_stats(self, team, year):
        url = f"{self.base_url_bref}/teams/{team}/{year}.html"
        totals_ct = per_game_ct = roster_ct = 0
        try:
            resp = self._get(url)
            soup = self._soup(resp.content)

            players = []

            # totals
            totals = self._find_table(soup, 'totals') or self._find_table(soup, 'team_totals')
            totals_map = {}
            if totals:
                for row in totals.select('tbody tr'):
                    name_cell = row.find(['th','td'], {'data-stat': 'player'})
                    if not name_cell:
                        continue
                    name = name_cell.get_text(strip=True)
                    def gi(stat):
                        td = row.find('td', {'data-stat': stat})
                        return self._safe_int(td.get_text()) if td else None
                    totals_map[name] = {
                        'age': self._safe_int((row.find('td', {'data-stat': 'age'}) or {}).get_text()
                                              if row.find('td', {'data-stat': 'age'}) else None),
                        'games': gi('g'),
                        'games_started': gi('gs'),
                        'minutes': gi('mp'),
                    }
                totals_ct = len(totals_map)

            # per-game
            per_game = self._find_table(soup, 'per_game')
            mpg_map, age_pg = {}, {}
            if per_game:
                for row in per_game.select('tbody tr'):
                    name_cell = row.find(['th','td'], {'data-stat': 'player'})
                    if not name_cell:
                        continue
                    pname = name_cell.get_text(strip=True)
                    mp_td = row.find('td', {'data-stat': 'mp_per_g'}) or row.find('td', {'data-stat': 'mp'})
                    if mp_td:
                        mpg_map[pname] = self._safe_float(mp_td.get_text())
                    age_td = row.find('td', {'data-stat': 'age'})
                    if age_td:
                        age_pg[pname] = self._safe_int(age_td.get_text())
                per_game_ct = len(mpg_map)

            # merge
            names = set(totals_map) | set(mpg_map)
            for name in sorted(names):
                t = totals_map.get(name, {})
                age = t.get('age') if t else None
                if age is None and name in age_pg:
                    age = age_pg[name]
                g = t.get('games')
                mp = t.get('minutes')
                mpg = mpg_map.get(name)
                if mpg is None and isinstance(mp, int) and isinstance(g, int) and g > 0:
                    mpg = round(mp / g, 1)
                players.append({
                    'player': name,
                    'team': team,
                    'year': year,
                    'age': age,
                    'games': g,
                    'games_started': t.get('games_started') if t else None,
                    'minutes': mp,
                    'minutes_per_game': mpg
                })

            # roster fallback if still empty
            if not players:
                roster = self._find_table(soup, 'roster')
                if roster:
                    for row in roster.select('tbody tr'):
                        name_cell = row.find(['th','td'], {'data-stat': 'player'})
                        if not name_cell:
                            continue
                        name = name_cell.get_text(strip=True)
                        def gi(stat):
                            td = row.find('td', {'data-stat': stat})
                            return self._safe_int(td.get_text()) if td else None
                        age = gi('age')
                        g   = gi('g')
                        gs  = gi('gs')
                        mp  = gi('mp')
                        mpg = round(mp / g, 1) if isinstance(mp, int) and isinstance(g, int) and g > 0 else None
                        players.append({
                            'player': name,
                            'team': team,
                            'year': year,
                            'age': age,
                            'games': g,
                            'games_started': gs,
                            'minutes': mp,
                            'minutes_per_game': mpg
                        })
                    roster_ct = len(players)

            self.log.info("Players: %s %s -> %d (totals=%d, per_game=%d, roster_fallback=%d)",
                          team, year, len(players), totals_ct, per_game_ct,
                          max(0, roster_ct - (totals_ct or 0)))
            if not players:
                self.log.warning("No player rows found for %s %s (totals/per_game/roster all empty).", team, year)
            elif self.log.level <= logging.DEBUG:
                self.log.debug("First row: %s", players[0])
            return players

        except requests.RequestException as e:
            self.log.warning("Error fetching player stats for %s %s: %s", team, year, e)
            return []

    # ---------- Metrics + orchestration ----------
    def calculate_network_metrics(self, salary_data):
        if not salary_data:
            return {}
        salaries = [p['salary'] for p in salary_data if p.get('salary', 0) > 0]
        if not salaries:
            return {}
        s = sorted(salaries)
        n = len(s)
        cumsum = np.cumsum(s)
        gini = (2 * np.sum([(i + 1) * s[i] for i in range(n)]) / (n * cumsum[-1])) - (n + 1) / n
        total = float(sum(salaries))
        s_desc = sorted(salaries, reverse=True)
        return {
            'gini_coefficient': gini,
            'top_1_share': (s_desc[0] / total) if n >= 1 else 0.0,
            'top_3_share': (sum(s_desc[:3]) / total) if n >= 3 else 0.0,
            'top_5_share': (sum(s_desc[:5]) / total) if n >= 5 else 0.0,
            'salary_mean': float(np.mean(salaries)),
            'salary_std': float(np.std(salaries)),
            'salary_cv': float(np.std(salaries) / np.mean(salaries)) if np.mean(salaries) > 0 else 0.0,
            'num_players': n,
        }

    def collect_all_data(self, start_year=2021, end_year=2025):
        all_data = {
            'salaries': [],
            'team_stats': [],
            'player_stats': [],
            'network_metrics': [],
        }
        os.makedirs(self.output_dir, exist_ok=True)

        for year in range(start_year, end_year + 1):
            if time.time() < self.domain_cooldown_until:
                wait = self.domain_cooldown_until - time.time()
                self.log.warning("Season %s-%s delayed by %.1fs (cooldown)", year-1, year, wait)
                time.sleep(wait)

            self.log.info("=== Season %s-%s ===", year - 1, year)

            # team stats first
            self.log.info("Fetching team statistics for %s…", year)
            ts = self.get_team_stats(year)
            all_data['team_stats'].extend(ts)
            if self.checkpoint_mode in ("season","team"):
                self._checkpoint_save(all_data, f"post-season-stats-{year}")

            # per team
            for i, team in enumerate(self.teams, 1):
                t0 = perf_counter()
                self.log.info("[%02d/%02d] %s %s", i, len(self.teams), team, year)

                sals = self.get_team_salaries(team, year)
                all_data['salaries'].extend(sals)

                pls = self.get_player_stats(team, year)
                all_data['player_stats'].extend(pls)

                if sals:
                    m = self.calculate_network_metrics(sals)
                    m['team'] = team; m['year'] = year
                    all_data['network_metrics'].append(m)

                if self.checkpoint_mode == "team":
                    self._checkpoint_save(all_data, f"{team}-{year}")

                time.sleep(1 + random.uniform(0, 1))
                self.log.debug("Team %s %s done in %.2fs", team, year, perf_counter() - t0)

            if self.checkpoint_mode in ("season","team"):
                self._checkpoint_save(all_data, f"end-of-season-{year}")

        # coaches optional
        self.log.info("Fetching coaching data…")
        coaches = self.get_coaching_data()
        all_data['coaches'] = coaches

        return all_data

    def get_coaching_data(self):
        url = f"{self.base_url_bref}/coaches/NBA_stats.html"
        try:
            resp = self._get(url)
            soup = self._soup(resp.content)
            tbl = self._find_table(soup, 'NBA_coaches') or soup.find('table', {'id': 'NBA_coaches'})
            out = []
            if tbl:
                for row in tbl.select('tbody tr'):
                    a = row.find('a')
                    if not a:
                        continue
                    name = a.get_text(strip=True)
                    def g(stat):
                        td = row.find('td', {'data-stat': stat})
                        return td.get_text(strip=True) if td else None
                    out.append({
                        'coach': name,
                        'seasons': self._safe_int(g('years')),
                        'games': self._safe_int(g('g')),
                        'wins': self._safe_int(g('wins')),
                        'losses': self._safe_int(g('losses')),
                        'win_pct': self._safe_float(g('win_loss_pct')),
                    })
            self.log.info("Coaches parsed: %d", len(out))
            return out
        except requests.RequestException as e:
            self.log.warning("Error fetching coaching data: %s", e)
            return []

    # ----- saving -----
    def _atomic_to_csv(self, df, path):
        tmp = path + ".tmp"
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)

    def save_to_csv(self, data):
        os.makedirs(self.output_dir, exist_ok=True)
        for key in ('salaries','team_stats','player_stats','network_metrics','coaches'):
            rows = data.get(key) or []
            df = pd.DataFrame(rows)
            cols = self.schemas[key]
            for c in cols:
                if c not in df.columns:
                    df[c] = np.nan
            df = df[cols]
            self._atomic_to_csv(df, self.files[key])
            self.log.info("Saved %d %s rows -> %s", len(df), key, self.files[key])

    def _checkpoint_save(self, data, note=""):
        self.log.info("Checkpoint save (%s) -> %s", note, self.output_dir)
        try:
            self.save_to_csv(data)
        except Exception as e:
            self.log.error("Checkpoint save failed: %s", repr(e))

    def merge_datasets(self, data):
        if not data.get('salaries') or not data.get('player_stats'):
            self.log.warning("Merge skipped: missing salaries or player_stats.")
            return pd.DataFrame()
        df_s = pd.DataFrame(data['salaries'])
        df_p = pd.DataFrame(data['player_stats'])
        merged = pd.merge(df_s, df_p, on=['player','team','year'], how='outer')
        if data.get('team_stats'):
            df_t = pd.DataFrame(data['team_stats'])
            merged = pd.merge(merged, df_t, on=['team','year'], how='left')
        self._atomic_to_csv(merged, self.files['merged'])
        self.log.info("Saved complete dataset -> %s", self.files['merged'])
        return merged


# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser(description="NBA data collector with robust salaries + player parsing and frequent checkpoints.")
    ap.add_argument("--start-year", type=int, default=2021, help="Season end year label (e.g., 2021 for 2020-21).")
    ap.add_argument("--end-year", type=int, default=2025, help="Season end year label.")
    ap.add_argument("--output-dir", type=str, default="nba_data", help="Where to write CSVs.")
    ap.add_argument("--log-level", type=str, default="INFO", help="DEBUG, INFO, WARNING, ERROR.")
    ap.add_argument("--log-dir", type=str, default="logs", help="Directory for log files.")
    ap.add_argument("--log-file", type=str, default="datascraper.log", help="Log file name.")
    ap.add_argument("--base-delay", type=float, default=6.0, help="Base polite delay.")
    ap.add_argument("--jitter", type=float, default=3.0, help="Random jitter added to delay.")
    ap.add_argument("--checkpoint", type=str, default="team", choices=["none","season","team"],
                    help="How often to write CSV checkpoints.")
    return ap.parse_args()


def main():
    args = parse_args()
    logger = setup_logging(args.log_level, args.log_dir, args.log_file)

    logger.info("=" * 60)
    logger.info("NBA Data Collection Script")
    logger.info("=" * 60)
    logger.info("Python=%s | requests=%s | pandas=%s | numpy=%s | lxml",
                sys.version.split()[0], requests.__version__, pd.__version__, np.__version__)
    logger.info("Seasons: %s-%s through %s-%s",
                args.start_year - 1, args.start_year, args.end_year - 1, args.end_year)

    collector = NBADataCollector(
        logger,
        base_delay=args.base_delay,
        jitter=args.jitter,
        max_connect_read_retries=3,
        max_retry_after=7200,
        output_dir=args.output_dir,
        checkpoint_mode=args.checkpoint,
    )

    data = collector.collect_all_data(args.start_year, args.end_year)
    logger.info("Saving data to CSV files…")
    collector.save_to_csv(data)
    logger.info("Creating merged dataset…")
    collector.merge_datasets(data)
    logger.info("=" * 60)
    logger.info("Data collection complete!")
    logger.info("=" * 60)
    logger.info("Generated files in %s:", args.output_dir)
    logger.info("  - nba_salaries.csv")
    logger.info("  - nba_team_stats.csv")
    logger.info("  - nba_player_stats.csv")
    logger.info("  - nba_network_metrics.csv")
    logger.info("  - nba_coaches.csv")
    logger.info("  - nba_complete_dataset.csv")


if __name__ == "__main__":
    main()
