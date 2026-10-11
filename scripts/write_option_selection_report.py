"""Report every causal option filter and audit forecast decisions independently."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import hashlib
import gzip
import json
import math
from pathlib import Path
import shutil

from scripts.write_option_budgets_report import audit_records, read_csv, pct


def audit_forecasts(result, rows, closed):
    decisions, residual = {}, 0.
    for row in rows:
        key = row['capital'], row['policy'], row['timestamp'], row['instrument']
        if key in decisions:
            raise ValueError('Duplicate forecast decision')
        decisions[key] = row
        when, cutoff = datetime.fromisoformat(row['timestamp']), datetime.fromisoformat(row['history_cutoff'])
        expiry = datetime.fromisoformat(row['expiry'])
        if cutoff != when.replace(hour=0, minute=0, second=0, microsecond=0):
            raise ValueError('Forecast cutoff includes incomplete current-day data')
        if row['last_completed_daily_close'] and datetime.fromisoformat(row['last_completed_daily_close']) + timedelta(days=1) > when:
            raise ValueError('Forecast includes a future daily close')
        if int(row['forecast_horizon_days']) != math.ceil((expiry - when).total_seconds() / 86400.):
            raise ValueError('Forecast horizon does not match known expiry')
        accepted = row['accepted'] == 'True'
        config = result['forecast_configs'][row['policy']]
        if accepted and int(row['history_daily_closes']) < config['minimum_daily_closes']:
            raise ValueError('Accepted forecast lacks its required history')
        debit = float(row['buy_debit_per_unit'])
        if debit <= 0 or not math.isfinite(debit):
            raise ValueError('Forecast has invalid purchase debit')
        if row['expected_edge']:
            edge, receipt = float(row['expected_edge']), float(row['expected_net_receipt_per_unit'])
            residual = max(residual, abs(edge - (receipt / debit - 1.)))
            if accepted != (edge >= float(row['minimum_edge'])):
                raise ValueError('Forecast acceptance does not match its fixed threshold')
            if not 0 <= float(row['estimated_win_probability']) <= 1 + 1e-12:
                raise ValueError('Invalid estimated payoff probability')
        elif accepted and row['reason'] != 'warmup_only':
            raise ValueError('Accepted priced filter lacks its estimated edge')
    for row in closed:
        if row['policy'] not in result['forecast_configs']:
            continue
        decision = decisions[row['capital'], row['policy'], row['entry'], row['instrument']]
        if decision['accepted'] != 'True':
            raise ValueError('A rejected forecast was nevertheless purchased')
        residual = max(residual, abs(float(row['paid_including_costs']) / float(row['qty'])
                                     - float(decision['buy_debit_per_unit'])))
    if residual > 1e-7:
        raise ValueError('Forecast decision/debit audit does not reconcile')
    return residual


def write_report(source, out):
    r = json.loads(source.read_text())
    out.mkdir(parents=True, exist_ok=True)
    stem = 'THETABOT_OPTION_SELECTION_2026-10-12'
    paths = {}
    for label, name in [('closed', 'closed_options'), ('budgets', 'purchase_budgets'), ('decisions', 'entry_selection')]:
        suffix = {'closed': 'CLOSED_OPTIONS', 'budgets': 'PURCHASE_BUDGETS', 'decisions': 'ENTRY_SELECTION'}[label]
        paths[label] = out / (stem + '_' + suffix + '.csv')
        shutil.copyfile(source.parent / (name + '.csv'), paths[label])
    closed, budgets, decisions = (read_csv(paths[x]) for x in ('closed', 'budgets', 'decisions'))
    cash_error = audit_records(r, closed, budgets)
    forecast_error = audit_forecasts(r, decisions, closed)
    r['report_diagnostics'] = {key + '_sha256': hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()}
    r['report_diagnostics'].update(closed_records=len(closed), budget_records=len(budgets), forecast_decisions=len(decisions),
        maximum_cash_record_residual=cash_error, maximum_forecast_record_residual=forecast_error)
    evidence = out / (stem + '.json.gz')
    payload = (json.dumps(r, separators=(',', ':'), ensure_ascii=False, allow_nan=False) + '\n').encode()
    evidence.write_bytes(gzip.compress(payload, mtime=0))
    d = r['results']['10000'];full = d['scenarios']['nominal']['full_continuous']
    chosen = d['selection']['frozen_candidate']
    value = full[chosen]
    worst = min(d['scenarios'][s]['full_continuous'][chosen]['max_intraday_drawdown_bound'] for s in d['scenarios'])
    scenarios = list(d['scenarios'])
    lines = ['# ThetaBot: filtr očekávané opční výplaty — 12. 10. 2026', '',
        '**Jde o modelový výzkum na již opakovaně použitých datech. Pozitivní odhad výplaty není ověřená tržní výhoda.**',
        'Nákup používá historický obchod s příplatkem, nikoli ověřenou ask nabídku nebo hloubku trhu.',
        'Průběžné opční ceny a propady jsou syntetické. Live způsobilost modelu: false.', '',
        '## Výsledek a kontrola přínosu', '',
        f'Volba zmrazená podle let 2022–2024 v šesti scénářích: pro účet 10 000 **`{chosen}`**.',
        f'Plný modelový výnos {pct(value["total_return"])}, CAGR {pct(value["cagr"])}, '
        f'nejhorší modelový propad ze šesti scénářů {pct(worst)}.',
        'Filtr může odmítnout opci, jejíž odhadovaná čistá výplata nepřevyšuje plné nákupní náklady o stanovenou rezervu.',
        'Hotovost se při odmítnutí nespotřebuje ani nepřerozdělí do druhého kontraktu.', '',
        '| Účet 10 000 | Nominální výnos | Náklady + zpoždění | Volatilita předpovědi ×0,75 | ×1,25 | Nejhorší model DD |',
        '|---|---:|---:|---:|---:|---:|']
    for policy in ('options_off', 'call_current1', 'call_current2', 'call_warmup', chosen, 'direction_empirical10'):
        values = {s: d['scenarios'][s]['full_continuous'][policy] for s in scenarios}
        lines.append(f'| `{policy}` | ' + ' | '.join(pct(values[s]['total_return'])
            for s in ('nominal', 'joint_stress', 'forecast_low', 'forecast_high'))
            + f' | {pct(min(x["max_intraday_drawdown_bound"] for x in values.values()))} |')
    lines += ['', f'**Samotná kontrola s čekáním na historii dává {pct(full["call_warmup"]["total_return"])}, více než zvolený filtr.**',
        'Proto nominální zvýšení proti původní call strategii nelze připsat pouze předpovědi. Filtr v tomto běhu',
        'zlepšuje společný nákladový/časový stres a modelový propad vůči této kontrole; statisticky nezávislý důkaz nemáme.', '',
        f'Zvolený filtr má {value["options"]["buys"]} nákupů a {value["options"]["winning_options"]} ziskových opcí.',
        f'Tři největší výhry tvoří {100 * value["top_three_option_pnl_fraction_of_net"]:.2f} % čistého opčního P&L.',
        f'Při jejich nezískání klesá výnos celého účtu na {pct(d["missed_top_three"][chosen]["result"]["total_return"])},',
        f'oproti {pct(full["options_off"]["total_return"])} spotu. Závislost na výhrách zůstává zásadní.',
        f'Směrový empirický filtr při vynechání svých tří největších výher dává {pct(d["missed_top_three"]["direction_empirical10"]["result"]["total_return"])};',
        'je však pouze další popisnou variantou a jeho odhad je citlivý na předpokládanou volatilitu.', '',
        'U účtu 1 000 žádná inverse varianta nekoupila opci při povolených lotech a rozpočtu. Výběr zůstává spot;',
        'ten sám při společném stresu překračuje 20% historický rozpočet propadu.', '',
        '## Co přesně předpovídáme', '',
        f'[Zmrazený protokol](OPTION_FORECAST_RESEARCH_PLAN.md), SHA-256 `{r["protocol_sha256"]}`.', '',
        '- Pouze dokončené denní závěry před UTC půlnocí dne nákupu; minimálně 252, nejvýše 730.',
        '- Empirický model používá minulé překrývající se H-denní logaritmické výnosy se stejnými vahami.',
        '- RV model používá volatilitu posledních 60 denních výnosů a lognormální rozdělení s nulovým driftem ceny.',
        '- H je počet kalendářních dnů do známé expirace, zaokrouhlený nahoru. Skutečné plnění/vypořádání zůstává 4h.',
        '- Odhadovaná výplata odečítá delivery a převodní poplatky. Předpověď předpokládá shodu indexu a spotu.',
        '  Skutečný modelový cash flow dál používá oficiální delivery index a pozorovaný spotový převod.',
        '- Výhoda 10/25 % se vztahuje k ceně prémie s náklady, nikoli k celému účtu ani k jistému výnosu.',
        '- Rozpočet nákupů: nejvýše 2 % aktuálního financovaného účtu a současně 20 % opční hotovosti.',
        '- Žádné opční ztráty se nedoplňují ze spotu. Opce se nevypisují a nevzniká dluh.',
        '- Snížení/zvýšení volatility předpovědi mění volbu nákupů. Změna IV pouze pro držené značky nemění cash profit.', '',
        'Development výběr maximalizuje nejhorší zisk v šesti scénářích při kladném zisku a DD <=20 % ve všech.',
        'Volbu ukládáme před pozdějšími přepočty. Pozdější data jsme však už studovali; není to nový holdout.', '']
    for capital, record in r['results'].items():
        nominal = record['scenarios']['nominal']
        lines += [f'## Všechny varianty: účet {capital}', '',
            f'Zmrazená volba `{record["selection"]["frozen_candidate"]}`; plný šestiscénářový rizikový screen: '
            f'**{record["frozen_full_risk_screen_passed"]}**.', '',
            '| Varianta | Plný výnos 2022–2026 | CAGR | Samostatný 2025–2026 | Nejhorší ze 6 výnosů | Nejhorší model DD |',
            '|---|---:|---:|---:|---:|---:|']
        for policy, s in nominal['full_continuous'].items():
            stresses = [record['scenarios'][x]['full_continuous'][policy] for x in scenarios]
            lines.append(f'| `{policy}` | {pct(s["total_return"])} | {pct(s["cagr"])} | '
                f'{pct(nominal["later_continuous"][policy]["total_return"])} | {pct(min(x["total_return"] for x in stresses))} | '
                f'{pct(min(x["max_intraday_drawdown_bound"] for x in stresses))} |')
        lines += ['', 'Roční účty začínají samostatně; jejich procenta se nesčítají. Počty nákupů/odmítnutí jsou za plnou historii.', '',
            '| Varianta | Rok 2025 | Leden–září 2026 | Nákupů / výher | Odmítnutí / z toho krátká historie | Bez top 3 plnění |',
            '|---|---:|---:|---|---|---:|']
        for policy, s in nominal['full_continuous'].items():
            opts = s.get('options', {})
            missed = record['missed_top_three'].get(policy)
            lines.append(f'| `{policy}` | {pct(nominal["2025"][policy]["total_return"])} | '
                f'{pct(nominal["2026"][policy]["total_return"])} | {opts.get("buys", 0)} / {opts.get("winning_options", 0)} | '
                f'{opts.get("forecast_rejections", 0)} / {opts.get("forecast_history_skips", 0)} | '
                + (pct(missed['result']['total_return']) if missed else '—') + ' |')
        lines += ['', '### Samostatný nákladový a časový stres', '',
            '| Varianta | Zdvojené sazby a vyšší prémie | +4h zpoždění | Náklady + zpoždění |', '|---|---:|---:|---:|']
        for policy in nominal['full_continuous']:
            lines.append(f'| `{policy}` | ' + ' | '.join(pct(record['scenarios'][x]['full_continuous'][policy]['total_return'])
                for x in ('cost_stress', 'delayed', 'joint_stress')) + ' |')
        lines += ['', '### Citlivost syntetických držených značek', '',
            '| Varianta | Poloviční IV: DD | Dvojnásobná IV: DD |', '|---|---:|---:|']
        for policy in record['iv_mark_sensitivity']['0.5']:
            lines.append(f'| `{policy}` | {pct(record["iv_mark_sensitivity"]["0.5"][policy]["model_dd"])} | '
                f'{pct(record["iv_mark_sensitivity"]["2.0"][policy]["model_dd"])} |')
        lines += ['', 'Koncový zisk při těchto změnách IV zůstal stejný. Průběžný propad není ověřenou historickou mark cenou.', '']
    lines += ['## Náklady a krátký horizont: účet 10 000', '',
        '| Varianta | Opční poplatky | Převodní poplatky | Převodní skluz | Příplatek nad cenu obchodu |',
        '|---|---:|---:|---:|---:|']
    for policy, s in full.items():
        costs = s.get('options', {}).get('costs')
        if costs:
            lines.append(f'| `{policy}` | ' + ' | '.join(f'{costs[x]:.2f}' for x in ('option_fee', 'conversion_fee', 'conversion_impact', 'premium_markup_cost')) + ' |')
    lines += ['', 'USD-equivalent za nominální běh; spotové náklady a všechny ostatní scénáře jsou v JSON.', '',
        '| Varianta | Nejlepší 7 dní | Nejlepší 30 dní | Nejlepší 90 dní | Zdvojnásobení 7 / 30 / 90 |',
        '|---|---:|---:|---:|---|']
    for policy in ('options_off', 'call_warmup', chosen, 'direction_empirical10'):
        gains = full[policy]['rolling_gains']
        lines.append(f'| `{policy}` | ' + ' | '.join(pct(gains[x]['maximum_return']) for x in ('7', '30', '90'))
            + ' | ' + ' / '.join(str(gains[x]['doublings']) for x in ('7', '30', '90')) + ' |')
    lines += ['', 'Okna se překrývají a používají syntetické opční značky; nejde o realizované ani garantované budoucí výnosy.', '',
        '## Ověření a reprodukce', '',
        f'Účetně ověřeno {r["reconciled_paths"]} cest; NAV reziduum {r["maximum_accounting_error"]:.3g}. '
        f'{r["archived_control_checks"]} kontrol zachovalo archivované ekonomické hodnoty.',
        f'Nezávislý audit: {len(closed)} uzavřených nákupů, {len(budgets)} měsíčních rozpočtů a {len(decisions)} rozhodnutí.',
        f'Reziduum cash záznamů {cash_error:.3g}; rozhodnutí/nákupních debetů {forecast_error:.3g}.',
        'Audit kontroluje cut-off, dokončenost dne, známou expiraci, minimální historii, fixní práh výhody a soulad s nakoupeným množstvím.',
        f'Surové zdroje: {r["source_checks"]["verified_raw_source_files"]} souborů, '
        f'{r["source_checks"]["verified_trade_windows"]} obchodních oken; původní pozorování hash `{r["original_observation_sha256"]}`.', '',
        '```bash', 'python -m scripts.download_spot_flow', 'python -m scripts.download_long_options',
        'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m scripts.research_option_selection --options-enabled',
        'python -m scripts.write_option_selection_report', 'python -m pytest -q tests', '```', '',
        f'[{evidence.name}]({evidence.name}) — úplný JSON komprimovaný gzip: všechny varianty, scénáře, volby a limity.',
        *[f'[{p.name}]({p.name})' for p in paths.values()],
        f'Evidence SHA-256 `{hashlib.sha256(evidence.read_bytes()).hexdigest()}`.', '', '## Omezení', '']
    lines += [f'- {x}.' for x in r['limitations']]
    lines += ['', 'Primární specifikace: [původní opční protokol](LONG_OPTIONS_RESEARCH_PLAN.md).', '']
    path = out / (stem + '.md');path.write_text('\n'.join(lines));return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('artifacts/option_selection_20261012/summary.json'))
    parser.add_argument('--out', type=Path, default=Path('docs/evaluation'))
    args = parser.parse_args();print(write_report(args.source, args.out))


if __name__ == '__main__':
    main()
