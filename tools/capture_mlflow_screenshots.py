"""Capture screenshots of the real MLflow UI for the report.

Needs tools/requirements.txt and `playwright install chromium`.
Run from the repository root: python tools/capture_mlflow_screenshots.py
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from playwright.sync_api import Page, sync_playwright
from playwright.sync_api import TimeoutError as PlaywrightTimeout

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports" / "figures"
WIDTH, HEIGHT = 1500, 950
# Runs tables need extra width, or MLflow cuts off the metric columns.
RUNS_WIDTH = 2000


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def api(base: str, path: str, body: dict) -> dict:
    request = urllib.request.Request(
        f"{base}/api/2.0/mlflow/{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def wait_for_server(base: str) -> None:
    for _ in range(90):
        try:
            urllib.request.urlopen(f"{base}/health", timeout=2)
            return
        except OSError:
            time.sleep(2)
    raise TimeoutError("MLflow did not start.")


def stop_server(server: subprocess.Popen) -> None:
    """Stop MLflow and its workers. On Windows, stopping only the parent leaves them running."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(server.pid)], capture_output=True)
    else:
        server.terminate()
    server.wait(timeout=30)


def open_page(page: Page, url: str, wait_for: str, height: int, width: int = WIDTH) -> None:
    page.set_viewport_size({"width": width, "height": height})
    page.goto(url)
    page.wait_for_selector(f"text={wait_for}", timeout=90_000)
    page.wait_for_timeout(3000)


def close_assistant(page: Page) -> None:
    """Close MLflow's assistant panel, which otherwise covers a quarter of the page."""
    page.get_by_text("MLflow Assistant").wait_for(timeout=30_000)
    box = page.get_by_text("MLflow Assistant").bounding_box()
    page.mouse.click(WIDTH - 36, box["y"] + box["height"] / 2)
    page.wait_for_timeout(1000)


def show_columns(page: Page, names: list[str]) -> None:
    """Add parameter and metric columns to a runs table."""
    menu = page.get_by_role("dialog")
    # The list of columns can load slowly on a busy machine, so try opening it again if needed.
    for attempt in range(4):
        page.get_by_role("button", name="Columns").first.click()
        try:
            menu.get_by_text(names[0], exact=True).first.wait_for(timeout=15_000)
            break
        except PlaywrightTimeout:
            if attempt == 3:
                raise
            page.keyboard.press("Escape")
            page.wait_for_timeout(3000)
    for name in names:
        # A name can be both a parameter and a tag. Parameters are listed first.
        menu.get_by_text(name, exact=True).first.click()
        page.wait_for_timeout(300)
    page.keyboard.press("Escape")
    page.wait_for_timeout(1500)


def save(page: Page, name: str, clip: dict | None = None) -> None:
    page.screenshot(path=FIGURES / f"{name}.png", clip=clip)
    print(f"saved {name}.png")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mlflow",
            "ui",
            "--backend-store-uri",
            "sqlite:///mlflow.db",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_for_server(base)
        experiments = api(base, "experiments/search", {"max_results": 100})["experiments"]
        ids = {item["name"]: item["experiment_id"] for item in experiments}
        final_id = ids["final-held-back-test"]
        prompt_id = ids["stage-3-ttt-rnn"]
        runs = api(base, "runs/search", {"experiment_ids": [prompt_id], "max_results": 50})["runs"]
        winner = next(
            run
            for run in runs
            if any(t["key"] == "winner" for t in run["data"].get("tags", []))
        )
        run_ids = json.dumps([run["info"]["run_id"] for run in runs])

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel="chromium")
            page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})

            open_page(page, f"{base}/#/experiments", "stage-1-nested-learning", 440)
            close_assistant(page)
            save(page, "mlflow_experiments")

            columns = ["faithfulness", "relevancy_when_answered", "refusal_rate"]
            # The table is cropped to its useful columns so it stays readable when printed.
            clip = {"x": 230, "y": 150, "width": 1420, "height": 195}
            open_page(page, f"{base}/#/experiments/{final_id}/runs", "final-", 480, RUNS_WIDTH)
            show_columns(page, columns + ["setup"])
            save(page, "mlflow_final_runs", clip)

            compare = urllib.parse.quote(run_ids)
            experiment = urllib.parse.quote(json.dumps([prompt_id]))
            url = f"{base}/#/compare-runs?runs={compare}&experiments={experiment}"
            open_page(page, url, "faithfulness", 600)
            save(page, "mlflow_compare", {"x": 200, "y": 0, "width": 1300, "height": 600})

            run_id = winner["info"]["run_id"]
            url = f"{base}/#/experiments/{prompt_id}/runs/{run_id}"
            open_page(page, url, "faithfulness", 960)
            save(page, "mlflow_run_detail", {"x": 205, "y": 235, "width": 790, "height": 485})
            browser.close()
    finally:
        stop_server(server)


if __name__ == "__main__":
    main()
