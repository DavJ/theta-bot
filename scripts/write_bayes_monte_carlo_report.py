"""Independently audit the frozen spot study and publish complete evidence."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t

from scripts.research_spot_flow import load_spot_flow
from scripts.research_bayes_monte_carlo import REPO, PLAN, SEEDS, SCENARIOS

STEM = 'THETABOT_BAYES_MONTE_CARLO_2026-10-12'
DESTINATION = REPO / 'docs/evaluation'


def audit_ledgers(out, result, markets):
    """Rebuild cash, quantities and NAV from each exported fill, independently."""
    maximum, paths = 0., 0
    for capital, account in result['results'].items():
        for policy, statistics in account['scenarios']['nominal']['full_continuous'].items():
            eq = pd.read_csv(out / f'{capital}_{policy}_equity.csv', float_precision='round_trip')
            trades = pd.read_csv(out / f'{capital}_{policy}_trades.csv', float_precision='round_trip')
            eq['timestamp'] = pd.to_datetime(eq.timestamp, utc=True).astype('datetime64[ns, UTC]')
            trades['timestamp'] = pd.to_datetime(trades.timestamp, utc=True).astype('datetime64[ns, UTC]')
            index = pd.DatetimeIndex(eq.timestamp)
            if not index.is_unique or not index.is_monotonic_increasing or not trades.timestamp.isin(index).all():
                raise ValueError('Invalid exported ledger clock')
            signed = np.where(trades.side == 'buy', trades.qty, -trades.qty)
            if not trades.side.isin(['buy', 'sell']).all() or (trades.qty <= 0).any():
                raise ValueError('Invalid exported fill side/quantity')
            flows = pd.Series(-signed * trades.price.to_numpy() - trades.fee.to_numpy(), index=trades.timestamp)
            cash = float(capital) + flows.groupby(level=0).sum().reindex(index, fill_value=0.).cumsum()
            nav = cash.copy()
            residuals = [float(np.max(np.abs(cash.to_numpy() - eq.usdt.to_numpy())))]
            if cash.min() < -1e-8:
                raise ValueError('Exported ledger borrowed cash')
            for symbol, market in markets.items():
                keep = trades.symbol.eq(symbol).to_numpy()
                changes = pd.Series(signed[keep], index=trades.loc[keep, 'timestamp'])
                inventory = changes.groupby(level=0).sum().reindex(index, fill_value=0.).cumsum()
                if inventory.min() < -1e-9:
                    raise ValueError('Exported ledger shorted an asset')
                residuals.append(float(np.max(np.abs(inventory.to_numpy() - eq['base_' + symbol].to_numpy()))))
                nav += inventory * market.close.reindex(index)
            residuals.extend([float(np.max(np.abs(nav.to_numpy() - eq.equity.to_numpy()))),
                abs(float(nav.iloc[-1]) - statistics['final_equity']),
                abs(float(trades.fee.sum()) - statistics['fees_paid_total']),
                abs(float(trades.slippage.sum()) - statistics['slippage_paid_total'])])
            if len(trades) != statistics['trades_count'] or max(residuals) > 1e-7:
                raise ValueError('Exported ledger does not reproduce reported economics')
            maximum, paths = max(maximum, *residuals), paths + 1
    return {'audited_full_nominal_ledgers': paths, 'maximum_exported_cash_inventory_nav_error': maximum}


def audit_forecasts_and_decisions(out, result):
    f = pd.read_csv(out / 'forecasts.csv', float_precision='round_trip')
    for name in ('available_at', 'last_training_outcome_at', 'last_analogue_outcome_at', 'last_fusion_outcome_at'):
        f[name] = pd.to_datetime(f[name], utc=True)
    valid = f.loc[f.ready]
    for name in ('last_training_outcome_at', 'last_analogue_outcome_at', 'last_fusion_outcome_at'):
        if not valid[name].dropna().le(valid.loc[valid[name].notna(), 'available_at']).all():
            raise ValueError('Exported forecast used a future outcome')
    if (valid.training_count < 252).any() or not np.allclose(valid.bayes_df, 6 + valid.bayes_observations):
        raise ValueError('Invalid training count or posterior degrees of freedom')
    if not np.allclose(valid.bayes_variance, valid.bayes_scale**2 * valid.bayes_df / (valid.bayes_df - 2)):
        raise ValueError('Predictive variance does not match the Student-t posterior')
    if (not np.allclose(valid.fused_mean, valid.fused_weight_bayes * valid.bayes_mean
                        + valid.fused_weight_mc * valid.mc_mean)
            or not np.allclose(valid.fused_weight_bayes + valid.fused_weight_mc, 1.)
            or not valid.fused_weight_bayes.between(0, 1).all()
            or not (valid.fused_variance >= valid[['bayes_variance', 'mc_variance']].min(axis=1)).all()
            or not np.allclose(valid.mc_numerical_mean_se, np.sqrt(valid.mc_variance / valid.mc_draws))):
        raise ValueError('Exported mixture or MC uncertainty invariant failed')
    maximum = 0.
    for name, hurdle in result['log_cost_hurdles'].items():
        bayes = student_t.sf((hurdle - valid.bayes_mean) / valid.bayes_scale, valid.bayes_df)
        fused = norm.sf((hurdle - valid.fused_mean) / np.sqrt(valid.fused_variance))
        maximum = max(maximum, float(np.max(np.abs(bayes - valid[f'bayes_prob_{name}']))),
                      float(np.max(np.abs(fused - valid[f'fused_prob_{name}']))))
        for expert in ('bayes', 'mc', 'fused', 'unconditional'):
            if not valid[f'{expert}_prob_{name}'].between(0, 1).all():
                raise ValueError('Exported probability outside [0,1]')
        if (not np.allclose(valid[f'mc_prob_{name}'] * 4096, np.rint(valid[f'mc_prob_{name}'] * 4096))
                or not np.allclose(valid[f'mc_exact_prob_{name}'] * 64, np.rint(valid[f'mc_exact_prob_{name}'] * 64))):
            raise ValueError('Monte Carlo and analogue counts do not match probabilities')
    decisions = pd.read_csv(out / 'signal_decisions.csv', float_precision='round_trip')
    expected_gate = decisions.ready & decisions.mean_log_return.gt(decisions.log_cost_hurdle) & decisions.gain_probability.ge(.55)
    conviction = ((decisions.gain_probability - .5) / .1).clip(0., 1.).where(
        decisions.mean_log_return.gt(decisions.log_cost_hurdle), 0.).fillna(0.)
    expected_target = decisions.original_target * (.5 + .5 * conviction) * decisions.ready
    if (not expected_gate.equals(decisions.buy_allowed)
            or not np.allclose(conviction, decisions.conviction)
            or not np.allclose(expected_target, decisions.blended_target)
            or not (decisions.blended_target <= decisions.original_target + 1e-12).all()):
        raise ValueError('Exported decision does not match frozen gate/blend rules')
    if maximum > 1e-12:
        raise ValueError('Exported forecast probabilities do not reproduce')
    return {'daily_forecast_rows': len(f), 'ready_primary_forecasts': len(valid),
            'audited_source_decisions': len(decisions), 'maximum_probability_residual': maximum,
            'first_ready_available_at': str(valid.available_at.min())}


def pct(value):
    return f'{100 * value:+.2f}%'


def full_paths(account, policy):
    paths = [account['scenarios'][s]['full_continuous'][policy] for s in SCENARIOS]
    paths.extend(seed['full_continuous'][policy] for seed in account['seed_sensitivity'].values()
                 if policy in seed['full_continuous'])
    return paths


def report_lines(result, archive_sha, compressed_sha):
    lines = ['# ThetaBot: bayesovské a Monte Carlo signály — 12. 10. 2026', '',
        '**V tomto zmrazeném experimentu žádný nový model nezvýšil čistý zisk.**',
        'Na plném i pozdějším kontinuálním období měly Bayes, podmíněná historická',
        'simulace i jejich kombinace vyšší chybu předpovědi než jednoduchý průměr',
        'ze stejného tréninkového okna.',
        'Menší propady doprovází nižší expozice a nižší zisk. Výzkumný režim je volitelný',
        'a výchozím stavem vypnutý; jeho modelované pravděpodobnosti nejsou ověřené.', '',
        '## Hlavní srovnání', '',
        'Účet 10 000 USDT, leden 2022–září 2026, kontinuální financovaný spotový účet.',
        'Zisky jsou po poplatcích, spreadu a slippage. DD je konzervativní 4h high/low',
        'mez přes všechny čtyři exekuční scénáře a dvě další nominal MC inicializace.', '',
        '| Varianta | Čistý zisk | Náklady +4h zpoždění | Nejhorší DD | CAGR |',
        '|---|---:|---:|---:|---:|']
    account = result['results']['10000']
    for name in account['scenarios']['nominal']['full_continuous']:
        nom = account['scenarios']['nominal']['full_continuous'][name]
        joint = account['scenarios']['joint_stress']['full_continuous'][name]
        worst = min(x['max_intraday_drawdown_bound'] for x in full_paths(account, name))
        lines.append(f"| `{name}` | {pct(nom['total_return'])} | {pct(joint['total_return'])} | {pct(worst)} | {pct(nom['cagr'])} |")
    lines += ['', 'Kontrola `ready_control` pouze odloží začátek nákupů do dostupnosti stejné',
        'historie. Její malé zlepšení proti původnímu spotu proto není přínosem Bayese',
        'ani Monte Carlo. Pozdější kontinuální výsledky obou kontrol jsou stejné.', '',
        '## Co bylo implementováno', '',
        '- Bayesovská regrese dává odhad pětidenního log výnosu, Student-t prediktivní',
        '  rozdělení a oddělenou nejistotu očekávané hodnoty. Nepřekrývající se',
        '  tréninkové cíle zmírňují zdvojování informace; neprokazují nezávislost trhu.',
        '- Monte Carlo vzorkuje 4 096 dokončených pětidenních úseků ze 64 nejbližších',
        '  minulých tržních stavů. Nový signál pochází z modelu podobných stavů, ne ze',
        '  samotného náhodného vzorkování. Numerická chyba a nejistota modelu se liší.',
        '- Kombinace používá pouze již dokončené společné chyby obou expertů.',
        '  Váhy jsou nezáporné, jejich součet je jedna a prediktivní rozptyl má',
        '  konzervativní dolní mez. Shodné modely nevytvářejí umělou přesnost.',
        '- `_buy_gate` blokuje nákupy/navýšení, pokud odhad log výnosu nepřevyšuje',
        '  nákladovou překážku nebo P(zisku po překážce) <55 %. Zachovává původní',
        '  výstupy z neaktivních pozic a vynucené snížení capu.',
        '- `_blend` ponechává alespoň polovinu původního cíle a druhou polovinu',
        '  upravuje přes odhadovanou pravděpodobnost. Nezvyšuje původní cílovou',
        '  expozici a neuvolněnou hotovost nepřerozděluje.', '',
        'Všechny modely používají stejná dokončená cenová, objemová a theta data.',
        'Změna modelu není nový externí zdroj informací. Hurdle je first-order',
        'round-trip náklad z jádra, nikoli příslib držení přesně pět dnů.', '',
        '## Zmrazený výběr a pozdější kontroly', '',
        f"[Protokol](BAYES_MONTE_CARLO_RESEARCH_PLAN.md), SHA-256 `{result['protocol_sha256']}`.",
        'Byl uložen před novými zisky; výběr podle let 2022–2024 byl zapsán před',
        'pozdějšími ekonomickými a predikčními srovnáními. Data už byla použita v',
        'předchozích výzkumech, nejde o nový unseen holdout.', '']
    for capital, account in result['results'].items():
        frozen = account['selection']['frozen_candidate']
        gates = result['historical_gates'][capital]
        lines += [f'### Účet {capital} USDT', '', f'Zmrazená volba: `{frozen}`.',
            f"Zlepšení podle protokolu: **{gates['historical_economic_improvement']}**; "
            f"plný/seed rizikový screen: **{gates['full_and_seed_risk_passed']}**.", '',
            '| Varianta | 2025 | Leden–září 2026 | Pozdější kontinuální účet | Plný účet | Joint stres plného účtu |',
            '|---|---:|---:|---:|---:|---:|']
        for policy in account['scenarios']['nominal']['full_continuous']:
            values = [account['scenarios']['nominal'][p][policy]['total_return']
                      for p in ('2025', '2026', 'later_continuous', 'full_continuous')]
            values.append(account['scenarios']['joint_stress']['full_continuous'][policy]['total_return'])
            lines.append(f"| `{policy}` | " + ' | '.join(pct(v) for v in values) + ' |')
        lines += ['', 'Roční účty začínají samostatně; procenta se nesčítají.',
            'Dlouhý účet zachovává minulou hotovost, ztráty, držby i složený výnos.', '',
            '| Varianta | Náklady | +4h | Joint | Fills nominal | Poplatky USDT | Impact USDT | Průměrná expozice |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
        for policy, nom in account['scenarios']['nominal']['full_continuous'].items():
            stress = [pct(account['scenarios'][s]['full_continuous'][policy]['total_return']) for s in SCENARIOS[1:]]
            lines.append(f"| `{policy}` | " + ' | '.join(stress)
                + f" | {nom['trades_count']} | {nom['fees_paid_total']:.2f} | {nom['slippage_paid_total']:.2f} | {nom['mean_exposure']:.2%} |")
        lines += ['', '| Varianta | 7d maximum / zdvojení | 30d maximum / zdvojení | 90d maximum / zdvojení |',
                  '|---|---|---|---|']
        for policy, nom in account['scenarios']['nominal']['full_continuous'].items():
            windows = nom['rolling_gains']
            lines.append(f"| `{policy}` | " + ' | '.join(f"{pct(windows[d]['maximum_return'])} / {windows[d]['doublings']}"
                                                         for d in ('7', '30', '90')) + ' |')
        lines += ['', 'Žádné nové pravidlo nedokázalo opakované zdvojnásobování během týdnů nebo měsíců.', '']
    lines += ['## Přesnost předpovědi', '',
        'Pětidenní cíle 2025–září 2026, identické dokončené řádky. Poměr >1 znamená',
        'horší výsledek proti nepodmíněnému průměru/probabilitě z téhož okna.',
        'Hodnoty jsou popisné; překryv vyhodnocovaných cílů nedává nezávislé p-hodnoty.', '',
        '| Aktivum | Model | Pozorování | Poměr MSE | Poměr Brier loss |',
        '|---|---|---:|---:|---:|']
    for symbol, skill in result['forecast_skill']['later_continuous'].items():
        for expert in ('bayes', 'mc', 'fused'):
            mse = skill[expert]['log_return_mse'] / skill['unconditional']['log_return_mse']
            brier = skill[expert]['gain_brier_loss']['nominal'] / skill['unconditional']['gain_brier_loss']['nominal']
            lines.append(f"| {symbol} | {expert} | {skill['observations']} | {mse:.3f} | {brier:.3f} |")
    lines += ['', '## Citlivost Monte Carlo', '',
        'Stejný model, data i počty vzorků; mění se pouze seed. Volba se znovu neprovádí.', '',
        '| Varianta, účet 10 000 | Nejnižší plný zisk ze 3 seedů | Nejvyšší |',
        '|---|---:|---:|']
    account = result['results']['10000']
    for policy in ('bayes_buy_gate', 'mc_buy_gate', 'fused_buy_gate', 'bayes_blend', 'mc_blend', 'fused_blend'):
        values = [account['scenarios']['nominal']['full_continuous'][policy]['total_return']]
        values += [x['full_continuous'][policy]['total_return'] for x in account['seed_sensitivity'].values()]
        lines.append(f'| `{policy}` | {pct(min(values))} | {pct(max(values))} |')
    lines += ['', 'Samotný MC nákupní filtr je citlivý na numerické překročení prahu: malé',
        'změny odhadu mohou změnit vstup a další cash path. Více simulací řeší jen',
        'numerickou část; nevytváří tržní predikční výhodu.', '',
        '## Audit a reprodukce', '',
        f"- {result['reconciled_paths']} výpočetních cest, účetnictví: {result['accounting_passed']}.",
        f"- {result['archived_control_checks']} shodných archivovaných spotových kontrol.",
        f"- Nejvyšší cash/inventory/NAV residual výzkumu: {result['maximum_accounting_error']:.3g} USDT."]
    diagnostic = result['report_diagnostics']
    lines += [f"- Samostatně přepočteno {diagnostic['audited_full_nominal_ledgers']} CSV ledgerů; residual {diagnostic['maximum_exported_cash_inventory_nav_error']:.3g}.",
        f"- {diagnostic['daily_forecast_rows']} denních řádků hlavního seedu, {diagnostic['ready_primary_forecasts']} připravených předpovědí.",
        f"- {diagnostic['audited_source_decisions']} ověřených rozhodnutí před exekučním lagem.",
        f"- Přepočet pravděpodobností: residual {diagnostic['maximum_probability_residual']:.3g}.",
        f"- První dostupné kompletní předpovědi: {diagnostic['first_ready_available_at']}.", '',
        f'[Kompletní výsledky JSON.gz]({STEM}.json.gz), SHA-256 `{compressed_sha}`.',
        f'[Datový audit ZIP]({STEM}_EVIDENCE.zip), SHA-256 `{archive_sha}`.',
        'ZIP obsahuje hlavní denní předpovědi, zdrojová rozhodnutí, všechny plné',
        'nominal equity/trade CSV obou účtů, zmrazené volby, protokol, zdrojové',
        'soubory a manifest hashů. Úplné statistiky všech scénářů a seedů jsou v JSON.', '',
        '```bash', 'python -m scripts.research_bayes_monte_carlo --probabilistic-enabled --jobs 4',
        'python -m scripts.write_bayes_monte_carlo_report',
        'python -m pytest -q tests/test_probabilistic_signals.py', '```', '',
        'CLI v novém pracovním adresáři vytvoří frozen_protocol.json před prvním',
        'výpočtem. Při opakování kontroluje jeho SHA-256 proti protokolu. Původní',
        'důkaz a konfigurace jsou v ZIP; nemažte je pro post-result úpravu pravidel.', '',
        '## Meze interpretace', '',
        '20% historický rizikový screen není budoucí limit ztráty. Cílové alokační',
        'capy nejsou garancí nepřekročení expozice mezi rozhodnutími při pohybu cen.',
        'Tři dnešní přeživší kryptoměny a již opakovaně použitá historie nedávají',
        'nezávislý důkaz. Poplatky, impact a delay jsou fixní scénáře; skutečné',
        'budoucí fee tiers, hloubka trhu, riziko burzy a hodnoty USDT nebyly ověřeny.',
        'Snížení rizika samo o sobě není zvýšení ziskovosti ani návod na živé obchodování.', '',
        'Metodika: [konjugované bayesovské lineární modely](https://arxiv.org/abs/2406.01819),',
        '[Gaussovská bayesovská analýza](https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf),',
        '[původní autor o blokovém resamplingu](https://mathweb.ucsd.edu/~politis/impactBOOT.pdf).',
        'Tyto zdroje vysvětlují výpočetní nástroje; nepotvrzují ziskovost tohoto modelu.', '']
    return lines


def write(out, spot_folder):
    result = json.loads((out / 'summary.json').read_text())
    if hashlib.sha256(PLAN.read_bytes()).hexdigest() != result['protocol_sha256']:
        raise ValueError('Frozen protocol changed')
    markets, manifest = load_spot_flow(spot_folder)
    if result['spot_dataset_hashes'] != {s: entry['sha256'] for s, entry in manifest['datasets'].items()}:
        raise ValueError('Evidence archive uses a different market source')
    diagnostics = {**audit_ledgers(out, result, markets), **audit_forecasts_and_decisions(out, result)}
    csv_paths = sorted(out.glob('*.csv'))
    diagnostics['csv_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in csv_paths}
    source_paths = [PLAN, REPO / 'spot_bot/portfolio/probabilistic.py',
                    REPO / 'scripts/research_bayes_monte_carlo.py', Path(__file__),
                    REPO / 'tests/test_probabilistic_signals.py']
    diagnostics['source_sha256'] = {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    result['report_diagnostics'] = diagnostics
    payload = (json.dumps(result, separators=(',', ':'), allow_nan=False) + '\n').encode()
    (out / 'summary.json').write_bytes(payload)
    compressed = DESTINATION / (STEM + '.json.gz')
    compressed.write_bytes(gzip.compress(payload, mtime=0))
    archive = DESTINATION / (STEM + '_EVIDENCE.zip')
    files = {p.name: p.read_bytes() for p in [*csv_paths, out / 'summary.json',
                                            out / 'frozen_protocol.json', out / 'frozen_selection.json']}
    files.update({'source/' + str(p.relative_to(REPO)): p.read_bytes() for p in source_paths})
    file_manifest = {name: {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)} for name, data in files.items()}
    files['MANIFEST.json'] = (json.dumps(file_manifest, indent=2) + '\n').encode()
    with zipfile.ZipFile(archive, 'w') as bundle:
        for name, data in sorted(files.items()):
            entry = zipfile.ZipInfo(name, date_time=(2026, 10, 12, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o644 << 16
            bundle.writestr(entry, data, compresslevel=9)
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    compressed_sha = hashlib.sha256(compressed.read_bytes()).hexdigest()
    report = DESTINATION / (STEM + '.md')
    report.write_text('\n'.join(report_lines(result, archive_sha, compressed_sha)))
    print(json.dumps({'report': str(report), 'archive_bytes': archive.stat().st_size,
        'json_gz_bytes': compressed.stat().st_size, 'audit': {k: v for k, v in diagnostics.items() if not k.endswith('sha256')}}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research', type=Path, default=Path('artifacts/bayes_monte_carlo_20261012'))
    parser.add_argument('--spot-data', type=Path, default=Path('data/raw/spot_flow'))
    args = parser.parse_args()
    write(args.research, args.spot_data)


if __name__ == '__main__':
    main()
