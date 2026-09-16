import os
from datetime import date, datetime
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from predictor import *

app=Flask(__name__)

DEFAULTS={"min_conf":62,"min_edge":4,"min_books":8,"max_edge":15}

def num(v, default=0):
    try: return float(v)
    except: return default

def team_for_pick(r):
    return short_team(r["Home team"] if r["Pick"]=="HOME" else r["Away team"] if r["Pick"]=="AWAY" else "Draw")

def public_row(r):
    c=context_signal(r)
    market="verified" if r.get("Market integrity")=="OK" else "missing" if r.get("Market fair %") is None else "caution"
    price=r.get("Best market odds") or r.get("Market odds")
    return {**r,
        "Display team":team_for_pick(r),
        "Display odds":decimal_to_fractional(price),
        "Context signal":c,"Market signal":market,
        "Validation short":"RAW" if "FALLBACK" in str(r.get("Validation")) else "VALIDATED" if "APPROVED" in str(r.get("Validation")) else "CHECK"}

def analyze(match_day, scope="ALL SUPPORTED LEAGUES"):
    min_conf,min_edge,min_books,max_edge=[DEFAULTS[k] for k in ("min_conf","min_edge","min_books","max_edge")]
    selected=LEAGUES if scope=="ALL SUPPORTED LEAGUES" else {scope:LEAGUES[scope]}
    odds_key=os.environ.get("ODDS_API_KEY","").strip()
    out=[]; warnings=[]; quota=None
    for lname,meta in selected.items():
        code=meta["of"]; odds_events=[]; api_diag={}
        if odds_key:
            try:
                odds_events,api_diag=odds_fetch(odds_key,meta["odds"]); quota=api_diag.get("remaining")
            except Exception as e: warnings.append(f"{lname}: odds unavailable ({e})")
        try: matches=fixtures_for(code)
        except Exception as e:
            warnings.append(f"{lname}: fixtures unavailable ({e})"); continue
        games=[m for m in matches if str(m.get("date",""))[:10]==match_day.isoformat()]
        if not games: continue
        try: model,hist,elo,ntrain,engine_label,validation_status,validation_evidence=train(lname,code)
        except Exception as e:
            warnings.append(f"{lname}: model unavailable ({e})"); continue
        def av(t,k): return float(np.mean([x[k] for x in hist[t]])) if hist[t] else 0.
        for g in games:
            h=team_name(g.get("team1","")).strip(); a=team_name(g.get("team2","")).strip()
            if not h or not a: continue
            vals={"h_pts":av(h,"pts"),"a_pts":av(a,"pts"),"h_gf":av(h,"gf"),"a_gf":av(a,"gf"),"h_ga":av(h,"ga"),"a_ga":av(a,"ga"),"elo_diff":elo[h]-elo[a],"elo_home":elo_p(elo[h]+55-elo[a])}
            x=pd.DataFrame([[vals[k] for k in FEATURES]],columns=FEATURES); pr=model.predict_proba(x)[0]
            i=int(np.argmax(pr)); labels=["HOME","DRAW","AWAY"]; conf=float(pr[i])
            market,diag=consensus_for(odds_events,h,a,match_day) if odds_events else (None,{"stage":"no-events","reason":"No current odds event returned","trace":[],"rejected_books":[]})
            kickoff_iso=(market.get("event",{}).get("commence_time") if market else None) or matched_kickoff_from_diag(diag)
            _,kickoff_label=kickoff_uk_from_iso(kickoff_iso)
            kickoff_label=kickoff_label or f"{match_day.strftime('%a %d %b')} • time unavailable"
            odd=best=mprob=edge=ev=None; books=0
            if market:
                odd=float(market["median"][i]); best=float(market["best"][i]); mprob=float(market["fair"][i]); books=market["books"]
                edge=conf-mprob; ev=conf*best-1
            context=fixture_context(code,match_day,h,a)
            secondary=[]
            if not market: decision,reason="PREDICTION ONLY","No matched current bookmaker market."
            elif conf < min_conf/100: decision,reason="PASS",f"Model confidence is below {min_conf}%."
            elif edge < min_edge/100: decision,reason="PASS",f"Market edge is below +{min_edge}pp."
            elif ev <= 0: decision,reason="PASS","Expected value is not positive at the verified price."
            elif market.get("integrity")!="OK": decision,reason="VERIFY","Bookmaker prices failed the market-integrity guard."
            elif books < min_books: decision,reason="VERIFY",f"Only {books} bookmakers passed validation."
            elif edge > max_edge/100: decision,reason,secondary=secondary_auto_verify(market,i,conf,min_edge,validation_status,diag,kickoff_iso)
            elif not kickoff_iso: decision,reason="VERIFY","Matched market has no verified kickoff timestamp."
            elif diag.get("stage")!="accepted": decision,reason="VERIFY","Fixture/market audit did not finish accepted."
            else: decision,reason="BET","Auto verified: fixture, market, kickoff, confidence, edge, EV and bookmaker depth passed."
            row={"League":lname,"Match":f"{h} v {a}","Home team":h,"Away team":a,"Pick":labels[i],"Home %":round(pr[0]*100,1),"Draw %":round(pr[1]*100,1),"Away %":round(pr[2]*100,1),"Confidence %":round(conf*100,1),"Market odds":round(odd,2) if odd else None,"Best market odds":round(best,2) if best else None,"Market fair %":round(mprob*100,1) if mprob else None,"Edge pp":round(edge*100,1) if edge is not None else None,"EV %":round(ev*100,1) if ev is not None else None,"Bookmakers":books or None,"Market integrity":market.get("integrity") if market else None,"Kickoff UK":kickoff_label,"Kickoff ISO":kickoff_iso,"Model engine":engine_label,"Validation":validation_status,"Validation evidence":validation_evidence,"Decision":decision,"Decision reason":reason,"Secondary checks":secondary,"Context":context}
            out.append(public_row(row))
    likely=sorted([r for r in out if r["Pick"] in ("HOME","AWAY")],key=lambda r:r["Confidence %"],reverse=True)[:5]
    value=sorted([r for r in out if r["Decision"]=="BET"],key=lambda r:(num(r.get("Edge pp"))+0.15*num(r.get("EV %")),r["Confidence %"]),reverse=True)[:5]
    both=sorted([r for r in value if r["Confidence %"]>=68 and r["Pick"] in ("HOME","AWAY")],key=lambda r:r["Confidence %"],reverse=True)[:5]
    return {"date":match_day.isoformat(),"analysed":len(out),"likely":likely,"value":value,"both":both,"all":sorted(out,key=lambda r:r["Confidence %"],reverse=True),"warnings":warnings,"quota":quota,"odds_connected":bool(odds_key)}

@app.get("/")
def index():
    return render_template("index.html",today=date.today().isoformat(),leagues=list(LEAGUES))

@app.get("/health")
def health(): return jsonify({"ok":True})

@app.get("/api/picks")
def picks():
    try:
        d=pd.to_datetime(request.args.get("date") or date.today().isoformat()).date()
        scope=request.args.get("league") or "ALL SUPPORTED LEAGUES"
        if scope!="ALL SUPPORTED LEAGUES" and scope not in LEAGUES: scope="ALL SUPPORTED LEAGUES"
        return jsonify(analyze(d,scope))
    except Exception as e:
        app.logger.exception("analysis failed")
        return jsonify({"error":str(e)}),500

if __name__=="__main__":
    app.run(host="0.0.0.0",port=int(os.environ.get("PORT",5000)),debug=True)
