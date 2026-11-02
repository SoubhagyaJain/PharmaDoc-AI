"""Headless browser acceptance check against running local API/UI services."""

import argparse
import json
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8501")
    args = parser.parse_args()
    output = ROOT / "tmp/browser"
    output.mkdir(parents=True, exist_ok=True)
    screenshots = ROOT / "docs/screenshots"
    screenshots.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 1100}, accept_downloads=True)
        exceptions = []
        page.on("pageerror", lambda error: exceptions.append(str(error)))
        page.goto(args.url)
        page.wait_for_load_state("networkidle")
        page.locator("input[type=file]").wait_for(state="attached", timeout=45000)
        expect(page.get_by_text("Local API ready", exact=True)).to_be_visible()
        page.locator("input[type=file]").set_input_files(
            ROOT / "data/examples/prescription_scan.pdf"
        )
        process = page.get_by_role("button", name="Process document", exact=True)
        expect(process).to_be_visible(timeout=20000)
        process.click()
        page.get_by_role("tab", name="Fields & review", exact=True).wait_for(timeout=120000)
        page.get_by_role("tab", name="JSON export", exact=True).click()
        with page.expect_download() as downloaded:
            page.get_by_role("button", name="Download original JSON", exact=True).click()
        downloaded.value.save_as(output / "original.json")
        original = json.loads((output / "original.json").read_text(encoding="utf-8"))
        assert original["document_category"] == "prescription"
        assert original["medications"][0]["strength"]["value"] == "500 mg"
        assert original["medications"][0]["administered_dose"]["value"] == "1 tablet"
        assert original["pages"][0]["route"] == "ocr"
        page.get_by_role("tab", name="Fields & review", exact=True).click()
        page.get_by_role("combobox", name="Field to review", exact=True).click()
        page.get_by_role("option", name="patient_name", exact=True).click()
        page.get_by_role("textbox", name="Corrected value", exact=True).fill(
            "Synthetic Reviewed Person"
        )
        page.get_by_role("textbox", name="Review note", exact=True).fill(
            "Browser acceptance test correction"
        )
        page.get_by_role("button", name="Save correction", exact=True).click()
        expect(
            page.get_by_text("Correction saved. The original extraction is preserved.", exact=True)
        ).to_be_visible(timeout=20000)
        page.get_by_role("tab", name="JSON export", exact=True).click()
        with page.expect_download() as downloaded:
            page.get_by_role("button", name="Download reviewed JSON", exact=True).click()
        downloaded.value.save_as(output / "reviewed.json")
        reviewed = json.loads((output / "reviewed.json").read_text(encoding="utf-8"))
        assert reviewed["reviewed_fields"]["patient_name"]["value"] == "Synthetic Reviewed Person"
        assert reviewed["fields"] == original["fields"]
        assert reviewed["processing_duration_seconds"] == original["processing_duration_seconds"]
        page.get_by_role("tab", name="OCR & evidence", exact=True).click()
        expect(page.get_by_text("Page 1 · ocr", exact=True)).to_be_visible()
        page.screenshot(path=str(screenshots / "ocr-evidence.png"), full_page=True)
        page.get_by_role("tab", name="Fields & review", exact=True).click()
        page.screenshot(path=str(screenshots / "reviewer.png"), full_page=True)
        assert not page.get_by_test_id("stException").count(), page.locator("body").inner_text()
        assert not exceptions, exceptions
        browser.close()
    print(
        "Browser acceptance passed: scanned upload -> API -> OCR -> evidence -> correction -> unchanged original + reviewed export."
    )


if __name__ == "__main__":
    main()
