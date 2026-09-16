import os, re, unicodedata, math
from collections import defaultdict, deque
from datetime import datetime, timezone
from functools import lru_cache
from fractions import Fraction
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

LEAGUES={
    "Premier League":{"of":"en.1","odds":"soccer_epl"},
    "Championship":{"of":"en.2","odds":"soccer_efl_champ"},
    "Bundesliga":{"of":"de.1","odds":"soccer_germany_bundesliga"},
    "La Liga":{"of":"es.1","odds":"soccer_spain_la_liga"},
    "Serie A":{"of":"it.1","odds":"soccer_italy_serie_a"},
    "Ligue 1":{"of":"fr.1","odds":"soccer_france_ligue_one"},
}

# V15 league-aware promotion policy. These choices are based on the V14
# chronological unseen-data calibration tests at the 62% audit threshold.
# No league is allowed to inherit another league's calibration result.
V15_POLICY={
    "Premier League":{"engine":"calibrated","status":"APPROVED","evidence":"V14 gap +6.1pp → +2.1pp"},
    "Championship":{"engine":"raw","status":"RAW FALLBACK","evidence":"V14 calibrated validation unavailable"},
    "Bundesliga":{"engine":"calibrated","status":"APPROVED","evidence":"V14 gap +3.4pp → -1.5pp"},
    "La Liga":{"engine":"raw","status":"APPROVED RAW","evidence":"Raw gap -0.7pp; calibration worsened to -11.7pp"},
    "Serie A":{"engine":"raw","status":"APPROVED RAW","evidence":"Calibration improvement too small for 56% fewer selections"},
    "Ligue 1":{"engine":"raw","status":"APPROVED RAW","evidence":"Raw gap -2.1pp; calibration worsened to -9.9pp"},
}
# Only leagues actually present in OpenFootball's 2026/27 JSON repository are exposed.
SEASONS=["2018-19","2019-20","2020-21","2021-22","2022-23","2023-24","2024-25","2025-26","2026-27"]
FEATURES=["h_pts","a_pts","h_gf","a_gf","h_ga","a_ga","elo_diff","elo_home"]

HEADERS={"User-Agent":"Mozilla/5.0 FootballPredictorV15/1.0","Accept":"application/json"}

def get_json(url):
    r=requests.get(url,headers=HEADERS,timeout=25)
    r.raise_for_status()
    data=r.json()
    if not isinstance(data,dict) or "matches" not in data:
        raise ValueError("Unexpected OpenFootball JSON format.")
    return data

@lru_cache(maxsize=64)
def season_json(season,code):
    return get_json(f"{RAW}/{season}/{code}.json")

def team_name(x):
    if isinstance(x,str): return x
    if isinstance(x,dict):
        return str(x.get("name") or x.get("title") or x.get("code") or "")
    return str(x)

def score_ft(m):
    s=m.get("score")
    if isinstance(s,dict):
        ft=s.get("ft")
        if isinstance(ft,(list,tuple)) and len(ft)>=2:
            try: return int(ft[0]),int(ft[1])
            except: pass
    return None

def elo_p(d): return 1/(1+10**(-d/400))

def make_training(code,w=8):
    hist=defaultdict(lambda:deque(maxlen=w))
    elo=defaultdict(lambda:1500.0)
    rows=[]
    used=0
    for season in SEASONS:
        try: matches=season_json(season,code)["matches"]
        except Exception: continue
        matches=sorted(matches,key=lambda m:str(m.get("date","")))
        for m in matches:
            sc=score_ft(m)
            if sc is None: continue
            h=team_name(m.get("team1","")).strip()
            a=team_name(m.get("team2","")).strip()
            if not h or not a: continue
            hg,ag=sc; eh,ea=elo[h],elo[a]
            def av(t,k):
                return float(np.mean([x[k] for x in hist[t]])) if hist[t] else 0.
            vals={"h_pts":av(h,"pts"),"a_pts":av(a,"pts"),
                  "h_gf":av(h,"gf"),"a_gf":av(a,"gf"),
                  "h_ga":av(h,"ga"),"a_ga":av(a,"ga"),
                  "elo_diff":eh-ea,"elo_home":elo_p(eh+55-ea)}
            if hg>ag: y=0; hp,ap,s=3,0,1.
            elif hg<ag: y=2; hp,ap,s=0,3,0.
            else: y=1; hp,ap,s=1,1,.5
            rows.append({**vals,"y":y})
            hist[h].append({"pts":hp,"gf":hg,"ga":ag})
            hist[a].append({"pts":ap,"gf":ag,"ga":hg})
            ex=elo_p(eh-ea)
            elo[h]+=24*(s-ex); elo[a]+=24*((1-s)-(1-ex))
            used+=1
    return pd.DataFrame(rows),hist,elo,used



def fixture_context(code, fixture_date, home, away):
    """Transparent context layer from OpenFootball completed matches only. No injury claims are invented."""
    season = SEASONS[-1]
    try:
        matches=season_json(season,code)["matches"]
    except Exception:
        return {"available":False,"reason":"Current-season context unavailable"}
    cutoff=pd.to_datetime(fixture_date,errors="coerce")
    rec=defaultdict(list); table=defaultdict(lambda:{"p":0,"pts":0,"gf":0,"ga":0,"w":0,"d":0,"l":0})
    for m in sorted(matches,key=lambda z:str(z.get("date",""))):
        dt=pd.to_datetime(m.get("date"),errors="coerce"); sc=score_ft(m)
        if sc is None or pd.isna(dt) or (not pd.isna(cutoff) and dt>=cutoff): continue
        h=team_name(m.get("team1","")).strip(); a=team_name(m.get("team2","")).strip(); hg,ag=sc
        if not h or not a: continue
        hp,ap=(3,0) if hg>ag else ((0,3) if hg<ag else (1,1))
        for t,gf,ga,pts,venue in [(h,hg,ag,hp,"H"),(a,ag,hg,ap,"A")]:
            x=table[t]; x["p"]+=1; x["pts"]+=pts; x["gf"]+=gf; x["ga"]+=ga
            x["w"]+=pts==3; x["d"]+=pts==1; x["l"]+=pts==0
            rec[t].append({"date":dt,"gf":gf,"ga":ga,"pts":pts,"venue":venue,"cs":ga==0,"btts":gf>0 and ga>0})
    order=sorted(table,key=lambda t:(table[t]["pts"],table[t]["gf"]-table[t]["ga"],table[t]["gf"]),reverse=True)
    pos={t:i+1 for i,t in enumerate(order)}
    def snap(t,venue):
        rr=rec.get(t,[]); last=rr[-8:]; va=[x for x in rr if x["venue"]==venue][-5:]
        def form(xs): return "".join("W" if x["pts"]==3 else "D" if x["pts"]==1 else "L" for x in xs) or "—"
        return {"position":pos.get(t),"played":table[t]["p"],"points":table[t]["pts"],"form":form(last),
                "ppg":round(sum(x["pts"] for x in last)/len(last),2) if last else None,
                "gf":round(sum(x["gf"] for x in last)/len(last),2) if last else None,
                "ga":round(sum(x["ga"] for x in last)/len(last),2) if last else None,
                "clean_sheet":round(100*sum(x["cs"] for x in last)/len(last),1) if last else None,
                "btts":round(100*sum(x["btts"] for x in last)/len(last),1) if last else None,
                "venue_form":form(va),"venue_ppg":round(sum(x["pts"] for x in va)/len(va),2) if va else None}
    hs,as_=snap(home,"H"),snap(away,"A")
    # Simple transparent scoreline context, not a replacement for the 1X2 model.
    import math
    lh=max(.15, ((hs.get("gf") or 1.2)+(as_.get("ga") or 1.2))/2)
    la=max(.15, ((as_.get("gf") or 1.0)+(hs.get("ga") or 1.0))/2)
    scores=[]
    for i in range(6):
        for j in range(6):
            pr=math.exp(-lh)*lh**i/math.factorial(i)*math.exp(-la)*la**j/math.factorial(j)
            scores.append((pr,f"{i}-{j}"))
    scores=sorted(scores,reverse=True)[:3]
    return {"available":True,"home":hs,"away":as_,"xg_like_home":round(lh,2),"xg_like_away":round(la,2),
            "scorelines":[{"score":x[1],"prob":round(x[0]*100,1)} for x in scores],
            "injuries":"UNAVAILABLE — no verified injury/suspension provider connected"}


@lru_cache(maxsize=16)
def train(lname,code):
    """V15 live engine: conservative model with league-specific calibration policy."""
    f,hist,elo,used=make_training(code)
    if len(f)<120:
        raise RuntimeError(f"Only {len(f)} completed historical matches available.")
    X=f[FEATURES].fillna(0); y=f["y"]
    policy=V15_POLICY[lname]

    def conservative():
        return HistGradientBoostingClassifier(
            max_iter=100,max_leaf_nodes=7,learning_rate=.045,
            min_samples_leaf=38,l2_regularization=8,random_state=42)

    if policy["engine"]=="calibrated":
        # Strict chronology: base model sees the earlier 82%; sigmoid calibrator
        # sees only the later 18%. No future fixture/result enters either stage.
        cut=max(250,int(len(f)*.82))
        proper=f.iloc[:cut]; cal=f.iloc[cut:]
        base=conservative()
        base.fit(proper[FEATURES].fillna(0),proper["y"])
        model=base
        if len(cal)>=50 and cal["y"].nunique()==3:
            try:
                calibrated=CalibratedClassifierCV(base,method="sigmoid",cv="prefit")
                calibrated.fit(cal[FEATURES].fillna(0),cal["y"])
                model=calibrated
            except Exception:
                # Fail closed to raw rather than pretending calibration succeeded.
                model=base
        engine_label="V15 CALIBRATED CONSERVATIVE" if model is not base else "V15 RAW SAFETY FALLBACK"
    else:
        model=conservative()
        model.fit(X,y)
        engine_label="V15 RAW CONSERVATIVE"

    return model,hist,elo,used,engine_label,policy["status"],policy["evidence"]

@lru_cache(maxsize=16)
def fixtures_for(code):
    return season_json("2026-27",code)["matches"]


ODDS_BASE="https://api.the-odds-api.com/v4/sports"

def norm(s):
    import re, unicodedata
    s=unicodedata.normalize("NFKD",str(s)).encode("ascii","ignore").decode().lower()
    s=re.sub(r"\b(fc|cf|afc|ac|calcio|club)\b"," ",s)
    return re.sub(r"[^a-z0-9]","",s)

# Explicit aliases are safer than increasingly loose fuzzy matching for money data.
# Keys and values are normalized forms. Add only verified club-name variants here.
TEAM_ALIASES = {
    "rayovallecanodemadrid":"rayovallecano",
    "rcdespanyoldebarcelona":"espanyol",
    "reialclubdeportiuespanyoldebarcelona":"espanyol",
    "rcdespanyol":"espanyol",
    "deportivoalaves":"alaves",
    "deportivoalaves":"alaves",
    "athleticclubbilbao":"athleticclub",
    "athleticbilbao":"athleticclub",
    "realbetisbalompie":"realbetis",
    "rcdmalorca":"mallorca",
    "realclubdeportivomallorca":"mallorca",
    "rceltadevigo":"celtavigo",
    "realclubceltadevigo":"celtavigo",
    "realoviedo":"oviedo",
}

def canonical_team(s):
    x=norm(s)
    return TEAM_ALIASES.get(x,x)

def team_match(a,b):
    """Fail-closed team match for market data: canonical equality or safe long prefix."""
    x,y=canonical_team(a),canonical_team(b)
    if not x or not y: return False
    if x==y: return True
    return min(len(x),len(y))>=8 and (x.startswith(y) or y.startswith(x))


def odds_fetch(api_key,sport_key):
    # V15.4: preserve the actual HTTP/API response diagnostics. Never expose apiKey.
    url=f"{ODDS_BASE}/{sport_key}/odds/"
    safe_params={"regions":"uk","markets":"h2h","oddsFormat":"decimal","dateFormat":"iso"}
    params={"apiKey":api_key,**safe_params}
    r=requests.get(url,params=params,timeout=25)
    meta={
        "endpoint":f"/v4/sports/{sport_key}/odds/",
        "sport_key":sport_key,
        "region":"uk",
        "market":"h2h",
        "http_status":r.status_code,
        "remaining":r.headers.get("x-requests-remaining"),
        "used":r.headers.get("x-requests-used"),
        "last":r.headers.get("x-requests-last"),
        "events":0,
        "api_message":"",
    }
    try:
        payload=r.json()
    except Exception:
        payload=None
        meta["api_message"]=(r.text or "")[:300]
    if isinstance(payload,dict):
        meta["api_message"]=str(payload.get("message") or payload.get("error") or payload.get("code") or "")[:300]
    if r.status_code in (401,403):
        raise RuntimeError(f"Odds API key rejected (HTTP {r.status_code}).")
    if r.status_code==429:
        raise RuntimeError("Odds API usage allowance reached (HTTP 429).")
    if r.status_code>=400:
        raise RuntimeError(f"Odds API HTTP {r.status_code}: {meta['api_message'] or 'request failed'}")
    data=payload if isinstance(payload,list) else []
    meta["events"]=len(data)
    if not isinstance(payload,list) and not meta["api_message"]:
        meta["api_message"]="Unexpected non-list response from odds endpoint"
    return data,meta

def _event_date_utc(e):
    try:
        return pd.to_datetime(e.get("commence_time"),utc=True).date()
    except Exception:
        return None

def kickoff_uk_from_iso(value):
    """Convert The Odds API UTC kickoff to Europe/London, including BST/GMT."""
    if not value:
        return None, None
    try:
        from zoneinfo import ZoneInfo
        dt=pd.to_datetime(value,utc=True).to_pydatetime().astimezone(ZoneInfo("Europe/London"))
        return dt, dt.strftime("%a %d %b • %H:%M UK")
    except Exception:
        return None, None

def matched_kickoff_from_diag(matchdiag):
    """Recover kickoff from the unique event trace even if bookmaker validation later fails."""
    if not isinstance(matchdiag,dict): return None
    hits=[t for t in matchdiag.get("trace",[]) if t.get("Home match") and t.get("Away match") and t.get("Date match")]
    if len(hits)==1:
        return hits[0].get("API time") or None
    return None

def consensus_for(events,home,away,fixture_date=None):
    """Fail-closed UK soccer 1X2 consensus plus rejection diagnostics."""
    trace=[]; candidates=[]
    for e in events:
        eh=e.get("home_team",""); ea=e.get("away_team",""); ed=_event_date_utc(e)
        hm=team_match(home,eh); am=team_match(away,ea)
        date_ok=(fixture_date is None) or (ed is not None and abs((ed-fixture_date).days)<=1)
        trace.append({"API event":f"{eh} v {ea}","API time":e.get("commence_time",""),
                      "Home match":hm,"Away match":am,"Date match":date_ok,
                      "OpenFootball":f"{home} v {away}"})
        if hm and am and date_ok: candidates.append(e)
    if len(candidates)!=1:
        return None,{"stage":"event","reason":f"Expected exactly 1 matching event; found {len(candidates)}",
                    "trace":trace,"rejected_books":[]}
    event=candidates[0]

    valid=[]; rejected=[]
    for b in event.get("bookmakers",[]):
        title=b.get("title",b.get("key","Unknown"))
        h2h=[m for m in b.get("markets",[]) if m.get("key")=="h2h"]
        if len(h2h)!=1:
            rejected.append({"Bookmaker":title,"Reason":f"h2h count {len(h2h)}"}); continue
        vals={"H":[],"D":[],"A":[]}; seen=[]
        for o in h2h[0].get("outcomes",[]):
            n=str(o.get("name","")).strip(); p=o.get("price"); seen.append(f"{n}={p}")
            if not isinstance(p,(int,float)) or not np.isfinite(p) or p<=1.01 or p>100: continue
            if n.casefold()=="draw": vals["D"].append(float(p))
            elif team_match(home,n): vals["H"].append(float(p))
            elif team_match(away,n): vals["A"].append(float(p))
        if any(len(vals[k])!=1 for k in ("H","D","A")):
            rejected.append({"Bookmaker":title,"Reason":"H/D/A mapping failed","Outcomes":" | ".join(seen)}); continue
        prices=np.array([vals["H"][0],vals["D"][0],vals["A"][0]],dtype=float)
        overround=float((1/prices).sum())
        if not (0.95 <= overround <= 1.20):
            rejected.append({"Bookmaker":title,"Reason":f"overround {overround:.3f}","Outcomes":" | ".join(seen)}); continue
        fair=(1/prices)/overround
        valid.append({"book":title,"prices":prices,"fair":fair,"overround":overround})

    if not valid:
        return None,{"stage":"bookmaker","reason":"Event matched, but no bookmaker passed 1X2 integrity checks",
                    "trace":trace,"rejected_books":rejected}
    arr=np.vstack([v["prices"] for v in valid]); fairs=np.vstack([v["fair"] for v in valid])
    median_prices=np.median(arr,axis=0)
    fair=np.median(fairs,axis=0); fair=fair/fair.sum(); best_prices=np.max(arr,axis=0)
    q25=np.percentile(arr,25,axis=0); q75=np.percentile(arr,75,axis=0)
    dispersion=np.max((q75-q25)/np.maximum(median_prices,1e-9))
    integrity="OK" if dispersion<=0.35 else "VERIFY"
    detail=[{"Bookmaker":v["book"],"Home":round(v["prices"][0],3),"Draw":round(v["prices"][1],3),
             "Away":round(v["prices"][2],3),"Overround %":round(v["overround"]*100,1)} for v in valid]
    market={"median":median_prices,"fair":fair,"best":best_prices,"books":len(valid),
            "integrity":integrity,"detail":detail,"event":event,"rejected":rejected}
    return market,{"stage":"accepted","reason":f"Matched event; {len(valid)} valid bookmaker(s), {len(rejected)} rejected",
                   "trace":trace,"rejected_books":rejected}


def secondary_auto_verify(market, pick_idx, conf, min_edge_pp, validation_status, matchdiag, kickoff_iso):
    """Fail-closed secondary audit for candidates that would previously need manual verification.
    Uses only independently observed bookmaker rows already accepted by the 1X2 parser.
    It never invents team-news/injury evidence.
    """
    checks=[]
    if validation_status == "RAW FALLBACK":
        return "PASS", "Secondary audit stopped: this league uses RAW FALLBACK because calibrated validation is unavailable.", checks
    if not market or matchdiag.get("stage") != "accepted" or not kickoff_iso:
        return "PASS", "Secondary audit failed fixture/market/kickoff identity checks.", checks
    detail=market.get("detail",[])
    if len(detail) < 8:
        return "PASS", f"Secondary audit failed bookmaker depth ({len(detail)} valid books).", checks
    col=("Home","Draw","Away")[pick_idx]
    prices=np.array([float(x[col]) for x in detail if isinstance(x.get(col),(int,float))],dtype=float)
    if len(prices)<8:
        return "PASS", "Secondary audit could not reconstruct enough independent prices.", checks
    # Robust price agreement: compare the middle 50% and remove dependence on the single best quote.
    q25,q50,q75=np.percentile(prices,[25,50,75])
    rel_iqr=(q75-q25)/max(q50,1e-9)
    checks.append(f"{len(prices)} accepted bookmakers; selected-price IQR {rel_iqr*100:.1f}%")
    if rel_iqr > .20:
        return "PASS", "Secondary audit rejected the signal because bookmaker prices are too dispersed.", checks
    # Rebuild each bookmaker's de-margined probability for the selected outcome.
    fairs=[]
    for x in detail:
        ps=np.array([x.get("Home"),x.get("Draw"),x.get("Away")],dtype=float)
        if np.all(np.isfinite(ps)) and np.all(ps>1.01):
            inv=1/ps; fairs.append(float((inv/inv.sum())[pick_idx]))
    if len(fairs)<8:
        return "PASS", "Secondary audit could not reconstruct bookmaker fair probabilities.", checks
    # Use the 75th percentile market probability: a deliberately tougher market comparison than the median.
    tough_market=float(np.percentile(fairs,75))
    robust_edge=conf-tough_market
    median_ev=conf*q50-1
    checks.append(f"Robust edge vs 75th-percentile market: {robust_edge*100:.1f}pp")
    checks.append(f"EV at median bookmaker price (not best price): {median_ev*100:.1f}%")
    rejected=len(market.get("rejected",[])); total=len(detail)+rejected
    reject_rate=(rejected/total) if total else 1
    checks.append(f"Bookmaker rejection rate: {reject_rate*100:.1f}%")
    if reject_rate > .35:
        return "PASS", "Secondary audit rejected the signal because too many bookmaker markets failed integrity checks.", checks
    if robust_edge < min_edge_pp/100 or median_ev <= 0:
        return "PASS", "Secondary audit removed the apparent value when tested against tougher consensus assumptions.", checks
    return "BET", "SECONDARY AUTO VERIFIED — the value survives a tougher bookmaker-consensus audit without relying on the best quote.", checks


def decimal_to_fractional(v, max_denominator=100):
    try:
        if v is None or not np.isfinite(float(v)) or float(v)<=1: return "—"
        frac=Fraction(float(v)-1).limit_denominator(max_denominator)
        if frac.numerator==frac.denominator: return "Evens"
        return f"{frac.numerator}/{frac.denominator}"
    except Exception:
        return "—"

def short_team(name):
    s=str(name or "").strip()
    s=re.sub(r"\b(FC|AFC|CF|AC)\b", "", s, flags=re.I)
    s=re.sub(r"\s+", " ", s).strip()
    return s

def context_signal(row):
    ctx=row.get("Context")
    if not isinstance(ctx,dict) or not ctx.get("available"): return "neutral"
    hh,aa=ctx.get("home",{}),ctx.get("away",{})
    hp,ap=hh.get("ppg"),aa.get("ppg"); hv,av=hh.get("venue_ppg"),aa.get("venue_ppg")
    if hp is None or ap is None: return "neutral"
    pred=str(row.get("Pick","")).upper()
    if pred=="HOME": support=(hp-ap)+.5*((hv if hv is not None else hp)-(av if av is not None else ap))
    elif pred=="AWAY": support=(ap-hp)+.5*((av if av is not None else ap)-(hv if hv is not None else hp))
    else: return "neutral"
    return "support" if support>=.5 else "caution" if support<=-.35 else "neutral"
