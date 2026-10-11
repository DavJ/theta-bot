"""Persist the fixed bounded-option study and independently audit purchase debits."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil


def pct(value):
    return f'{100 * value:+.2f}%'


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def audit_records(result, closed, budgets):
    paid = {}
    residual = 0.
    for row in closed:
        key = row['capital'], row['policy'], row['entry']
        buy = sum(float(row['buy_' + k]) for k in ('raw_premium', 'premium_markup_cost',
            'option_fee', 'conversion_impact', 'conversion_fee'))
        receipt = float(row['settlement_gross_payoff']) - sum(float(row['settlement_' + k])
            for k in ('option_fee', 'conversion_impact', 'conversion_fee'))
        if float(row['qty']) <= 0:
            raise ValueError('Nonpositive bought-option quantity')
        paid[key] = paid.get(key, 0.) + buy
        residual = max(residual, abs(buy - float(row['paid_including_costs'])),
            abs(receipt - float(row['net_receipt'])), abs(receipt - buy - float(row['net_pnl'])))
    covered = set()
    for row in budgets:
        key = row['capital'], row['policy'], row['timestamp']
        if key in covered:
            raise ValueError('Duplicate monthly purchase budget')
        covered.add(key)
        config = result['policies'][row['policy']]
        basis = float(row['account_open_equity']) if config['account_budget_basis'] == 'current' else float(row['capital'])
        allowed = min(config['monthly_cash_budget'] * float(row['cash_before']), config['monthly_account_budget'] * basis)
        spent = paid.get(key, 0.)
        residual = max(residual, abs(basis - float(row['budget_basis_equity'])),
            abs(allowed - float(row['allowed_debit'])), abs(spent - float(row['spent'])))
        if spent < 0 or spent > allowed + 1e-7:
            raise ValueError('Purchase debit exceeds funded account limits')
    for key in paid:
        if result['policies'][key[1]]['monthly_account_budget'] is not None and key not in covered:
            raise ValueError('Capped purchase is missing its monthly audit')
    for capital, record in result['results'].items():
        for policy, value in record['scenarios']['nominal']['full_continuous'].items():
            total = sum(float(x['net_pnl']) for x in closed if x['capital'] == capital and x['policy'] == policy)
            residual = max(residual, abs(total - value['option_net_pnl']))
    if residual > 1e-7:
        raise ValueError('Closed option/budget records do not reconcile')
    return residual


def write_report(source, out):
    r = json.loads(source.read_text())
    out.mkdir(parents=True, exist_ok=True)
    stem = 'THETABOT_OPTION_BUDGETS_2026-10-12'
    closed_path = out / (stem + '_CLOSED_OPTIONS.csv')
    budget_path = out / (stem + '_PURCHASE_BUDGETS.csv')
    shutil.copyfile(source.parent / 'closed_options.csv', closed_path)
    shutil.copyfile(source.parent / 'purchase_budgets.csv', budget_path)
    closed, budgets = read_csv(closed_path), read_csv(budget_path)
    residual = audit_records(r, closed, budgets)
    r['report_diagnostics'] = {'closed_options_sha256': hashlib.sha256(closed_path.read_bytes()).hexdigest(),
        'purchase_budgets_sha256': hashlib.sha256(budget_path.read_bytes()).hexdigest(),
        'closed_option_records': len(closed), 'monthly_budget_records': len(budgets),
        'maximum_independent_record_residual': residual}
    evidence = out / (stem + '.json')
    evidence.write_text(json.dumps(r, separators=(',', ':'), ensure_ascii=False, allow_nan=False) + '\n')
    d = r['results']['10000']
    full = d['scenarios']['nominal']['full_continuous']
    bounded, old = full['direction_current2'], full['legacy_direction_otm']
    missed = d['missed_top_three']['direction_current2']['result']
    frozen = d['selection']['frozen_candidate']
    lines = ['# ThetaBot: omezení opčních nákupů — 12. 10. 2026', '',
        '**Jde o další modelový výzkum na již použitých datech. Nákupní ceny nejsou ověřené historické ask nabídky.**',
        'Průběžné opční ceny i propady jsou syntetické. Výsledek není způsobilý pro živé obchodování.', '',
        '## Co se zlepšilo a co zůstává slabé', '',
        'Do výzkumného enginu přibylo omezení všech měsíčních nákupních debetů podle skutečně financovaného účtu.',
        'Platí pro prémii, obchodní poplatky, převodní poplatky i skluz dohromady. Výhry v opční části samy nemohou',
        'zvýšit příští nákup nad nastavené procento celého účtu. Spot opční ztráty nedoplňuje.', '',
        f'U účtu 10 000 směrová varianta s limitem 2 % aktuálního účtu dává modelově {pct(bounded["total_return"])}',
        f'oproti původním {pct(old["total_return"])}. Nejvyšší vykázaný měsíční nákupní debet v modelu klesl',
        f'z {100 * old["maximum_monthly_debit_account_fraction"]:.2f} % na {100 * bounded["maximum_monthly_debit_account_fraction"]:.2f} % účtu.',
        'Vyšší náklady a zpoždění nyní stresujeme i současně. Menší nominální profit je cena za omezené sázky.', '',
        '| Účet 10 000: varianta | Základní výnos | Vyšší náklady | +4h zpoždění | Náklady a zpoždění současně | Nejhorší model DD ze 4 scénářů |',
        '|---|---:|---:|---:|---:|---:|']
    for policy in ('options_off', 'legacy_direction_otm', 'direction_current2', 'direction_initial2', frozen):
        values = [d['scenarios'][s]['full_continuous'][policy] for s in ('nominal', 'cost_stress', 'delayed', 'joint_stress')]
        lines.append(f'| `{policy}` | ' + ' | '.join(pct(x['total_return']) for x in values)
                     + f' | {pct(min(x["max_intraday_drawdown_bound"] for x in values))} |')
    lines += ['', f'**Volba zmrazená podle vývoje 2022–2024 je `{frozen}`**, nikoli dodatečně vybraná nejvýnosnější',
        'varianta celého období. Směrová varianta s 2% limitem je popisný kandidát k dalšímu ověření.',
        'Tato časová hranice není nezávislý holdout: výsledky pozdějších let jsme již znali z předchozí práce.', '',
        f'Bez získání tří nejziskovějších nákupů klesá směrová 2% varianta na {pct(missed["total_return"])},',
        f'zatímco samotný spot dává {pct(full["options_off"]["total_return"])}. Koncentrace klesla z',
        f'{100 * old["top_three_option_pnl_fraction_of_net"]:.2f} % na {100 * bounded["top_three_option_pnl_fraction_of_net"]:.2f} %',
        'čistého opčního P&L, ale závislost na několika výhrách se nevyřešila. Vynechání je ex-post diagnostika',
        'selhání plnění; nepredikuje, které budoucí obchody vynechat.', '',
        'Pro účet 1 000 žádná z těchto inverse variant nekoupila opci při povolených minimálních lotech.',
        'Rezervovaný kapitál snižuje spotový výnos. I původní spot překračuje 20 % při společném stresu.', '',
        '## Přesná pravidla', '',
        f'[Zmrazený protokol](OPTION_BUDGET_RESEARCH_PLAN.md), SHA-256 `{r["protocol_sha256"]}`.', '',
        '- Dva OTM signály: call-only při kladném 30denním momentu; call/put podle znaménka momenta.',
        '- Pro každý signál tři limity: 2 % aktuálního účtu, 1 % aktuálního účtu, 2 % počátečního kapitálu.',
        '- Rozpočet je vždy také omezen 20 % aktuální opční hotovosti. Použije se menší limit.',
        '- Počáteční financování: 90 % spot / 10 % opční hotovost. Bez převodů ze spotu a bez dluhu.',
        '- Aktuální hodnota pro rozpočet: předchozí vlastní spotová pozice a hotovost při nynějším open,',
        '  plus opční hotovost. Aktuální close ani pozdější pozice velikost nákupu neovlivňují.',
        '- Nákupní limit neomezuje tržní cenu již držené opce a negarantuje budoucí maximální propad.',
        '- Kalendář, kontrakty, ocenění při nákupu/expiraci, lotech a sazby: původní opční protokol.',
        '- Volba na development: kladný zisk a DD <=20 % ve všech 4 scénářích; maximalizovat nejhorší zisk.',
        '- Nominální +5% příplatek k obchodu; nákladový stres +20% příplatek a zdvojené sazby; zpoždění 4h.',
        '- Chybějící a nedostupné kontrakty se nepřesouvají do jiných kontraktů nebo umělých zlomků.', '']
    for capital, record in r['results'].items():
        nominal = record['scenarios']['nominal']
        lines += [f'## Všechny varianty: účet {capital}', '',
            f'Zmrazená volba: `{record["selection"]["frozen_candidate"]}`. '
            f'Plný čtyřscénářový rizikový screen: **{record["frozen_full_risk_screen_passed"]}**.',
            'Výnos je za celý účet, leden 2022–září 2026. Pozdější účet začíná samostatně v lednu 2025.', '',
            '| Varianta | Plný zisk | CAGR | Zisk 2025–2026 | Nejhorší ze 4 zisků | Nejhorší model DD |',
            '|---|---:|---:|---:|---:|---:|']
        for policy, value in nominal['full_continuous'].items():
            stresses = [record['scenarios'][s]['full_continuous'][policy] for s in ('nominal', 'cost_stress', 'delayed', 'joint_stress')]
            lines.append(f'| `{policy}` | {pct(value["total_return"])} | {pct(value["cagr"])} | '
                f'{pct(nominal["later_continuous"][policy]["total_return"])} | {pct(min(x["total_return"] for x in stresses))} | '
                f'{pct(min(x["max_intraday_drawdown_bound"] for x in stresses))} |')
        lines += ['', '| Varianta | Samostatný 2025 | Leden–září 2026 | Nákupů / výher za celou historii | Malý lot / chybějící obchod |',
            '|---|---:|---:|---|---|']
        for policy, value in nominal['full_continuous'].items():
            opts = value.get('options', {})
            lines.append(f'| `{policy}` | {pct(nominal["2025"][policy]["total_return"])} | '
                f'{pct(nominal["2026"][policy]["total_return"])} | {opts.get("buys", 0)} / {opts.get("winning_options", 0)} | '
                f'{opts.get("minimum_lot_skips", 0)} / {opts.get("no_recent_trade", 0)} |')
        lines += ['', '### Bez nejlepších plnění a citlivost oceňování', '',
            '| Varianta | Zisk při vynechání top 3 výher | Poloviční IV: DD | Dvojnásobná IV: DD |',
            '|---|---:|---:|---:|']
        for policy, diagnostic in record['missed_top_three'].items():
            lines.append(f'| `{policy}` | {pct(diagnostic["result"]["total_return"])} | '
                f'{pct(record["iv_mark_sensitivity"]["0.5"][policy]["model_dd"])} | '
                f'{pct(record["iv_mark_sensitivity"]["2.0"][policy]["model_dd"])} |')
        lines += ['', 'Při změně IV zůstaly výplaty při expiraci i koncový zisk stejné. Propady zůstávají modelové.', '']
    lines += ['## Náklady a krátký horizont: účet 10 000', '',
        '| Varianta | Poplatky opcí | Poplatky převodu | Skluz převodu | Příplatek nad původní prémii |',
        '|---|---:|---:|---:|---:|']
    for policy, value in full.items():
        if policy not in d['missed_top_three']:
            continue
        costs = value['options']['costs']
        lines.append(f'| `{policy}` | ' + ' | '.join(f'{costs[k]:.2f}' for k in ('option_fee', 'conversion_fee', 'conversion_impact', 'premium_markup_cost')) + ' |')
    lines += ['', 'Částky v USD-equivalentu za plný nominální běh; spotové náklady jsou zvlášť v JSON.', '',
        '| Varianta | Nejlepší 7 dní | Nejlepší 30 dní | Nejlepší 90 dní | Zdvojnásobení 7 / 30 / 90 |',
        '|---|---:|---:|---:|---|']
    for policy in ('options_off', 'legacy_direction_otm', 'direction_current2', frozen):
        gain = full[policy]['rolling_gains']
        lines.append(f'| `{policy}` | ' + ' | '.join(pct(gain[x]['maximum_return']) for x in ('7', '30', '90'))
            + ' | ' + ' / '.join(str(gain[x]['doublings']) for x in ('7', '30', '90')) + ' |')
    lines += ['', 'Denní překrývající se okna zahrnují syntetické ceny držených opcí. Nejsou realizovanou výplatou ani odhadem budoucí četnosti.', '',
        '## Ověření a reprodukce', '',
        f'Účetně ověřeno {r["reconciled_paths"]} cest, maximální NAV reziduum {r["maximum_accounting_error"]:.3g}.',
        f'Celkem {r["archived_control_checks"]} porovnání kontrol zachovalo všechna archivovaná ekonomická pole.',
        f'Samostatně ověřeno {len(closed)} uzavřených nákupů a {len(budgets)} měsíčních rozpočtů; reziduum {residual:.3g}.',
        f'Ověřeno {r["source_checks"]["verified_raw_source_files"]} surových zdrojových souborů a '
        f'{r["source_checks"]["verified_trade_windows"]} historických obchodních oken.',
        f'Původní pozorování SHA-256 `{r["original_observation_sha256"]}`.', '',
        '```bash', 'python -m scripts.download_spot_flow', 'python -m scripts.download_long_options',
        'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m scripts.research_option_budgets --options-enabled',
        'python -m scripts.write_option_budgets_report', 'python -m pytest -q tests', '```', '',
        f'[{evidence.name}]({evidence.name}) — všechny varianty, období, scénáře, zmrazená volba a vynechané vstupy.',
        f'[{closed_path.name}]({closed_path.name}) — ceny, náklady, výplaty, ID/čas obchodu.',
        f'[{budget_path.name}]({budget_path.name}) — hotovost, základ limitu, povolený a skutečný měsíční debet.',
        f'JSON SHA-256 `{hashlib.sha256(evidence.read_bytes()).hexdigest()}`.', '',
        '## Omezení', '']
    lines += [f'- {x}.' for x in r['limitations']]
    lines += ['', 'Primární specifikace: [původní protokol](LONG_OPTIONS_RESEARCH_PLAN.md).', '']
    path = out / (stem + '.md')
    path.write_text('\n'.join(lines))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('artifacts/option_budgets_20261012/summary.json'))
    parser.add_argument('--out', type=Path, default=Path('docs/evaluation'))
    args = parser.parse_args()
    print(write_report(args.source, args.out))


if __name__ == '__main__':
    main()
