"""Regression checks for links exported from notebooks into the browser site."""

import importlib.util
import json
from pathlib import Path
import unittest

SITE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("export_content", SITE / "scripts" / "export_content.py")
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)
REPOSITORY = "https://github.com/HeinrichWirth/gaussian-splatting-course"


class ExportLinkTests(unittest.TestCase):
    def test_readme_sections_keep_filename_and_anchor_in_both_languages(self):
        cases = [
            ("ru", "README_ru.md", "блок-2-как-гауссианы-учатся"),
            ("en", "README.md", "block-2-how-gaussians-learn"),
        ]
        for language, filename, anchor in cases:
            with self.subTest(language=language):
                source = f"The next [block](../../{filename}#{anchor})."
                expected = f"The next [block]({REPOSITORY}/blob/main/{filename}#{anchor})."
                self.assertEqual(exporter.adapt(source, language, 5), expected)

    def test_plain_course_and_lesson_navigation_remain_local(self):
        for language, filename in [("ru", "README_ru.md"), ("en", "README.md")]:
            source = f"[Overview](../../{filename}) and [next](lesson-04.ipynb)."
            self.assertEqual(exporter.rewrite_course_links(source, language),
                             f"[Overview](#/{language}) and [next](#/{language}/4).")

    def test_existing_external_and_local_section_links_are_unchanged(self):
        source = ("[External](https://example.com/README.md#section) "
                  "and [this section](#section-3).")
        self.assertEqual(exporter.rewrite_course_links(source, "en"), source)

    def test_generated_transition_to_block_two_stays_in_the_site(self):
        for language, filename, anchor in [
            ("ru", "README_ru.md", "блок-2-как-гауссианы-учатся"),
            ("en", "README.md", "block-2-how-gaussians-learn"),
        ]:
            with self.subTest(language=language):
                lessons = json.loads((SITE / "dist" / "content" / f"{language}.json").read_text(encoding="utf-8"))
                text = "\n".join(cell["source"] for lesson in lessons
                                 for cell in lesson["cells"] if cell["type"] == "text")
                self.assertIn(f"](#/{language}/6)", text)
                self.assertEqual([lesson["number"] for lesson in lessons], list(range(1, 10)))
                self.assertEqual([lesson["block"] for lesson in lessons], [1]*5+[2]*4)
                self.assertNotRegex(text, r"\]\(\.\./\.\./README(?:_ru)?\.md(?:#|\))")

    def test_exporter_uses_notebooks_inside_this_repository(self):
        self.assertEqual(exporter.COURSE, SITE.parent)
        for language in ("ru", "en"):
            self.assertTrue((exporter.COURSE / "notebooks" / language / "lesson-05.ipynb").is_file())


if __name__ == "__main__":
    unittest.main()
