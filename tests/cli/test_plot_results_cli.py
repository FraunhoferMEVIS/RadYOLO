from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from radyolo import main


def test_plot_results_cli_calls_plotter(monkeypatch) -> None:
    runner = CliRunner()
    captured: dict[str, Path] = {}
    expected_path: Path | None = None

    def fake_plot_results(path: Path) -> None:
        captured["path"] = path

    monkeypatch.setattr("radyolo.plot_results_", fake_plot_results)

    with runner.isolated_filesystem():
        result_dir = Path("results_dir")
        result_dir.mkdir()
        result_path = result_dir / "results.csv"
        result_path.write_text("epoch,metrics/fitness\n0,0.0\n", encoding="utf-8")
        expected_path = result_path.resolve()

        result = runner.invoke(main, ["plot-results", str(result_path)])

    assert result.exit_code == 0
    assert expected_path is not None
    assert captured["path"] == expected_path


def test_plot_results_cli_returns_error_on_failure(
    monkeypatch,
    caplog,
) -> None:
    runner = CliRunner()

    def fake_plot_results(_path: Path) -> None:
        msg = "plot failure"
        raise RuntimeError(msg)

    monkeypatch.setattr("radyolo.plot_results_", fake_plot_results)
    caplog.set_level("ERROR", logger="radyolo")

    with runner.isolated_filesystem():
        result_dir = Path("results_dir")
        result_dir.mkdir()
        result_path = result_dir / "results.csv"
        result_path.write_text("epoch,metrics/fitness\n0,0.0\n", encoding="utf-8")

        result = runner.invoke(main, ["plot-results", str(result_path)])

    assert result.exit_code == 1
    assert "Traceback (most recent call last):" in result.output
    assert "Error: Plotting results failed. See logs for details." in result.output
    assert "An unexpected error occurred" in caplog.text
