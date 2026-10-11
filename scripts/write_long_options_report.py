"""Write the full exploratory option-profit comparison without execution claims."""
from __future__ import annotations

import argparse
import csv
from datetime import date, timedelta
import hashlib
import json
import shutil
from pathlib import Path


def pct(value):
    return f'{100 * value:+.2f}%'


def settled_endpoint_gains(folder, capital, policy):
    """Descriptive rolling windows with no synthetic option marks at either end."""
    daily, held = {}, {}
    with (folder / f'{capital}_{policy}_combined_equity.csv').open() as handle:
        for row in csv.DictReader(handle):
            daily[date.fromisoformat(row['timestamp'][:10])] = float(row['equity'])
    with (folder / f'{capital}_{policy}_option_equity.csv').open() as handle:
        for row in csv.DictReader(handle):
            held[date.fromisoformat(row['timestamp'][:10])] = float(row['held_contracts'])
    result = {}
    for days in (7, 30, 90):
        pairs = [(value / daily[day - timedelta(days=days)] - 1, day)
                 for day, value in daily.items()
                 if day - timedelta(days=days) in daily and held[day] == 0
                 and held[day - timedelta(days=days)] == 0]
        best, end = max(pairs) if pairs else (None, None)
        result[str(days)] = {'windows': len(pairs), 'doublings': sum(gain >= 1 for gain, _ in pairs),
            'maximum_return': best, 'end': str(end) if end else None,
            'start': str(end - timedelta(days=days)) if end else None}
    return result


def write_report(source, out):
    r = json.loads(source.read_text())
    if not r['options_enabled']:
        raise ValueError('An option report requires the explicitly enabled offline option study')
    out.mkdir(parents=True, exist_ok=True)
    stem = 'THETABOT_LONG_OPTIONS_2026-10-10'
    evidence = out / (stem + '.json')
    ledger = out / (stem + '_CLOSED_OPTIONS.csv')
    shutil.copyfile(source.parent / 'closed_options.csv', ledger)
    with ledger.open() as handle:
        closed = list(csv.DictReader(handle))
    chosen = r['results']['10000']
    chosen_policy = chosen['selection']['frozen_candidate']
    chosen_nominal = chosen['scenarios']['nominal']['full_continuous'][chosen_policy]
    chosen_trades = [x for x in closed if x['capital'] == '10000' and x['policy'] == chosen_policy]
    concentration = (sum(sorted((float(x['net_pnl']) for x in chosen_trades), reverse=True)[:3])
                     / chosen_nominal['option_net_pnl']) if chosen_nominal['option_net_pnl'] else 0.
    cost_dd_failed = any(chosen['scenarios'][scenario]['full_continuous'][chosen_policy]['max_intraday_drawdown_bound'] < -.2
                         for scenario in ('nominal', 'cost_stress', 'delayed'))
    settled = settled_endpoint_gains(source.parent, 10000, chosen_policy)
    residual = 0.
    for row in closed:
        buy = sum(float(row['buy_' + key]) for key in ('raw_premium', 'premium_markup_cost', 'option_fee', 'conversion_impact', 'conversion_fee'))
        receipt = float(row['settlement_gross_payoff']) - sum(float(row['settlement_' + key])
            for key in ('option_fee', 'conversion_impact', 'conversion_fee'))
        residual = max(residual, abs(buy - float(row['paid_including_costs'])),
            abs(receipt - float(row['net_receipt'])), abs(receipt - buy - float(row['net_pnl'])))
    for capital, record in r['results'].items():
        for policy, value in record['scenarios']['nominal']['full_continuous'].items():
            total = sum(float(row['net_pnl']) for row in closed if row['capital'] == capital and row['policy'] == policy)
            residual = max(residual, abs(total - value['option_net_pnl']))
    if residual > 1e-7:
        raise ValueError('Closed-option audit does not reconcile with costs or account totals')
    r['report_diagnostics'] = {'closed_options_sha256': hashlib.sha256(ledger.read_bytes()).hexdigest(),
        'closed_options_count': len(closed), 'closed_options_maximum_residual': residual,
        'chosen_10000_top_three_option_pnl_fraction': concentration,
        'chosen_10000_settled_endpoint_gains': settled}
    evidence.write_text(json.dumps(r, separators=(',', ':'), ensure_ascii=False, allow_nan=False) + '\n')
    lines = ['# ThetaBot: model ziskovosti koupených opcí — 10. 10. 2026', '',
        '**Výnosy jsou modelové. Vstupní ceny vycházejí z historických obchodů, nikoli z ověřených nabídek k nákupu.**',
        'Hodnota držených opcí a průběžné propady používají syntetický oceňovací model.',
        'Výsledek není ověřeným proveditelným opčním backtestem. Live způsobilost: **false**.', '',
        '## Výsledek', '',
        'Pro účet 1 000 jednotek opce nepřekonaly původní spot. Inverse kontrakty nebyly při stanoveném rozpočtu dostupné.',
        f'Pro účet 10 000 byla před pozdějšími výpočty vybrána **`{chosen_policy}`**:',
        f'modelový čistý výnos celého účtu {pct(chosen_nominal["total_return"])}, '
        f'oproti {pct(chosen["scenarios"]["nominal"]["full_continuous"]["options_off"]["total_return"])} bez opcí.',
        'Pozdější samostatně financovaný účet 2025–2026 dává podstatně menší výnos; nesmí se zaměňovat s plnou historií.', '',
        '| Vybraná opční varianta, účet 10 000 | Čistý výnos 2022–2026 | Modelový propad |',
        '|---|---:|---:|']
    for scenario, label in [('nominal', 'Základní model'), ('cost_stress', 'Vyšší prémie a dvojnásobné náklady'),
                            ('delayed', 'Zpoždění nákupu o 4 hodiny')]:
        value = chosen['scenarios'][scenario]['full_continuous'][chosen_policy]
        lines.append(f'| {label} | {pct(value["total_return"])} | {pct(value["max_intraday_drawdown_bound"])} |')
    lines += ['', ('**Kandidát neprošel všemi scénáři 20% rozpočtu modelového propadu.**' if cost_dd_failed else
                   'Ve třech scénářích modelový propad nepřekročil 20 %; skutečný průběžný propad tím není ověřen.'),
        f'Tři nejziskovější opce tvoří {100 * concentration:.2f} % čistého opčního P&L. '
        'Výsledek je koncentrovaný do malého počtu výplat a citlivý na dostupnost vstupní ceny.',
        f'Opční část na konci tvoří {100 * chosen_nominal["options"]["final_cash"] / chosen_nominal["final_equity"]:.2f} % účtu.',
        '**Počáteční 10% oddělení kapitálu není trvalý 10% strop expozice. Počáteční 2% nákupní limit také není trvalý 2% limit účtu.**',
        'Zisky zůstávají v opční části, ale ze spotu se do ní další peníze nepřevádějí.', '',
        '### Krátký horizont bez otevřených opcí na koncích okna', '',
        'Doplňková kontrola vybrané varianty a účtu 10 000: oba koncové denní body musí mít všechny opce vypořádané.',
        'Spot zůstává oceněn tržně. Tato okna nemají syntetickou opční cenu v počáteční ani konečné hodnotě.', '',
        '| Délka | Dostupných oken | Nejlepší výnos | Zdvojnásobení |', '|---|---:|---:|---:|']
    for days, values in settled.items():
        lines.append(f'| {days} dní | {values["windows"]} | {pct(values["maximum_return"])} | {values["doublings"]} |')
    lines += ['', 'Jde o dodatečnou popisnou diagnostiku, nikoli nové pravidlo výběru nebo odhad budoucích výnosů.', '',
        '## Co přesně porovnáváme', '',
        '- Jeden účet od ledna 2022 do září 2026; výnosy zahrnují složené úročení a modelové náklady.',
        '- Vypnuto: původní capped spot strategie s celým kapitálem. Rezerva: 90 % spot / 10 % hotovost.',
        '- Zapnuto: 90 % spot / 10 % oddělená opční část, bez doplňování ztrát ze spotu.',
        '- Každý měsíc první pondělí: nákup do posledního pátečního vypořádání měsíce.',
        '- Prémie a všechny náklady nákupu celkem nejvýše 20 % aktuální opční hotovosti.',
        '  Zpočátku tedy nejvýše 2 % celého účtu. Rozpočet může po opčních ziscích růst.',
        '- BTC/ETH: call při kladném 30denním momentu, nebo call/put podle jeho znaménka;',
        '  strike nejblíže poslednímu dokončenému dennímu závěru (ATM), nebo s cílem o 5 % mimo peníze (OTM). Opce pouze kupujeme.',
        '- Žádné půjčky, vypisování opcí, samostatné futures, martingale ani živé objednávky.',
        '- Inverse ceny/výplaty v BTC/ETH se převádějí za vlastní hotovost; linear jsou v USDC.',
        '- USDC BTC/ETH produkty existují až od srpna 2025. Před dostupností zůstává jejich část v hotovosti.',
        '- Minimální velikost kontraktu respektujeme; nepřístupný nákup se přeskočí.', '',
        f'Předem zmrazený [protokol](LONG_OPTIONS_RESEARCH_PLAN.md), SHA-256 `{r["protocol_sha256"]}`.',
        'Studovaná spotová období už byla použita; nejde o nový nezávislý holdout.', '',
        '## Model cen a nákladů', '',
        'Poslední skutečný obchod ve vybraném kontraktu musí předcházet nákupu a být maximálně 4 hodiny starý.',
        'Model zaplatí jeho prémii zvýšenou o 5 %, zaokrouhlenou nahoru na tick. Stres používá +20 %.',
        'Chybějící cenu nenahrazujeme umělou prémií a nevybíráme dodatečně jiný kontrakt.',
        'Nemáme historické bid/ask nabídky, dostupnou hloubku ani zaručené plnění.',
        'Zaokrouhlení na tick může zvýšit skutečný modelový příplatek nad 5 %.', '',
        'Nákupní poplatek je scénář 0,03 % podkladového objemu, nejvýše 12,5 % prémie.',
        'Vypořádání ziskové opce: 0,015 % podkladového objemu, nejvýše 12,5 % výplaty.',
        'Inverse převody: 0,1 % poplatek + 5 bps skluz + polovina 2 bps spreadu.',
        'Linear převody: 0,1 % poplatek a předpoklad USD/USDT/USDC parity. Náklady se v účetnictví neodečítají dvakrát.',
        'Stres zdvojnásobuje sazby; 12,5% strop zůstává. Jde o jednotný scénář podle současných pravidel,',
        'nikoli rekonstrukci historických poplatků a jejich změn.', '',
        'Výplata při expiraci používá oficiální Deribit delivery index. Průběžný Black-Scholes model má',
        'nulovou sazbu a vstupní IV neměnnou do expirace. Spot OHLC slouží i pro modelové průběžné propady.',
        'Všechny opce do konce období expirují: terminální výnos nezávisí na těchto syntetických značkách.', '']
    for capital, data in r['results'].items():
        nom = data['scenarios']['nominal']
        full, later = nom['full_continuous'], nom['later_continuous']
        stress = data['scenarios']['cost_stress']['full_continuous']
        delay = data['scenarios']['delayed']['full_continuous']
        frozen = data['selection']['frozen_candidate']
        lines += [f'## Počáteční účet {int(capital):,} jednotek'.replace(',', ' '), '',
            f'Volba zmrazená pouze podle vývoje 2022–2024: **`{frozen}`**.',
            'Procenta se vztahují k celému účtu. DD je modelový 4h bound; nevydáváme ho za skutečný maximální propad.', '',
            '| Varianta | Celý účet 2022–2026 | CAGR | 2025–2026 | Opční P&L | Model DD | Nákladový stres | Zpoždění +4h |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
        for p, s in full.items():
            lines.append(f'| `{p}` | {pct(s["total_return"])} | {pct(s["cagr"])} | {pct(later[p]["total_return"])} | '
                f'{s["option_net_pnl"]:+.2f} | {pct(s["max_intraday_drawdown_bound"])} | '
                f'{pct(stress[p]["total_return"])} | {pct(delay[p]["total_return"])} |')
        lines += ['', 'Samostatně začínající roční účty; nesčítat roční procenta. Počty nákupů, výher a přeskočení jsou za celou historii 2022–2026.', '',
            '| Varianta | Rok 2025 | Leden–září 2026 | Opčních nákupů | Ziskových | Přeskočeno: malý rozpočet / bez obchodu / bez kontraktu |',
            '|---|---:|---:|---:|---:|---|']
        for p, s in full.items():
            o = s.get('options', {})
            win = f'{o.get("winning_options", 0)}/{o.get("buys", 0)}'
            skips = ' / '.join(str(o.get(k, 0)) for k in ('minimum_lot_skips', 'no_recent_trade', 'no_listed_contract'))
            lines.append(f'| `{p}` | {pct(nom["2025"][p]["total_return"])} | {pct(nom["2026"][p]["total_return"])} | '
                         f'{o.get("buys", 0)} | {win} | {skips} |')
        lines += ['', '### Výnosy za kratší období', '',
            '| Varianta | Nejlepší 7 dní | Nejlepší 30 dní | Nejlepší 90 dní | Zdvojnásobení 7 / 30 / 90 dní |',
            '|---|---:|---:|---:|---|']
        for p, s in full.items():
            rolling = s['rolling_gains']
            best = ' | '.join(pct(rolling[d]['maximum_return']) for d in ('7', '30', '90'))
            counts = ' / '.join(str(rolling[d]['doublings']) for d in ('7', '30', '90'))
            lines.append(f'| `{p}` | {best} | {counts} |')
        lines += ['', 'Okna se překrývají. Zisky a zdvojnásobení uvnitř doby držení závisí na modelových cenách opcí;',
            'nejde o realizované výnosy ani budoucí pravděpodobnosti.', '',
            '### Citlivost modelového propadu na IV', '',
            '| Varianta | Poloviční IV | Vstupní IV | Dvojnásobná IV |', '|---|---:|---:|---:|']
        for p in data['iv_mark_sensitivity'].get('0.5', {}):
            lines.append(f'| `{p}` | {pct(data["iv_mark_sensitivity"]["0.5"][p]["model_dd"])} | '
                f'{pct(full[p]["max_intraday_drawdown_bound"])} | {pct(data["iv_mark_sensitivity"]["2.0"][p]["model_dd"])} |')
        lines += ['', 'Změna oceňovací IV nemění terminální opční cash flow ani zisk účtu; mění vykazovaný průběžný propad.', '']
    lines += ['## Ověření a reprodukce', '',
        f'Účetně ověřeno {r["reconciled_paths"]} spotových/opčních cest; maximální NAV reziduum '
        f'{r["maximum_accounting_error"]:.3g}. Vypnutý přepínač deleguje přímo na původní spotový engine.',
        'Dřívější spotová čísla pro účet 1 000 se kontrolují automaticky. Hotovost a opční množství zůstávají nezáporné.',
        f'S kontrolními součty bylo ověřeno {r["source_checks"]["verified_raw_source_files"]} zdrojových souborů '
        f'a {r["source_checks"]["verified_trade_windows"]} historických obchodních oken; '
        'zapsané obchody i delivery ceny odpovídají surovým oficiálním odpovědím.',
        'Měsíční expirace odstraňuje otevřené opce před konečným vyčíslením. Součet P&L obou oddělených částí tvoří P&L účtu.', '',
        f'Uzavřená kniha obsahuje {len(closed)} modelových nákupů napříč variantami. Samostatný součet prémií, '
        f'poplatků a čistých výplat i součet P&L odpovídají výsledkům; nejvyšší reziduum {residual:.3g}.', '',
        '```bash', 'python -m scripts.download_long_options',
        'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m scripts.research_long_options --options-enabled',
        'python -m scripts.write_long_options_report', 'python -m pytest -q tests', '```', '',
        'Bez `--options-enabled` výzkumný příkaz porovnává pouze spot. Výzkum nemá napojení na objednávky.',
        f'[{stem}.json]({stem}.json) obsahuje všechny modelové výsledky, diagnostiku, hash zdrojových pozorování a protokolu.',
        f'[{ledger.name}]({ledger.name}) obsahuje každý uzavřený nákup: cenu, poplatky, výplatu, ID a čas zdrojového obchodu.',
        f'Evidence SHA-256 `{hashlib.sha256(evidence.read_bytes()).hexdigest()}`.',
        'Surové API odpovědi a každý vybraný obchod mají URL, parametry, ID/čas a hash v `data/raw/long_options/`.',
        'Účetní knihy a 4h křivky jsou reprodukovatelné v `artifacts/long_options_20261010/`.', '',
        '## Omezení výsledku', '']
    lines += [f'- {x}.' for x in r['limitations']]
    lines += ['', 'Primární specifikace a zdroje: viz [protokol](LONG_OPTIONS_RESEARCH_PLAN.md).', '']
    path = out / (stem + '.md')
    path.write_text('\n'.join(lines))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('artifacts/long_options_20261010/summary.json'))
    parser.add_argument('--out', type=Path, default=Path('docs/evaluation'))
    args = parser.parse_args()
    print(write_report(args.source, args.out))


if __name__ == '__main__':
    main()
