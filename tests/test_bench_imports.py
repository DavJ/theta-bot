import importlib


def test_bench_modules_importable():
    importlib.import_module("bench.benchmark_pairs")
    importlib.import_module("bench.benchmark_matrix")
    importlib.import_module("bench.render_research_note")
