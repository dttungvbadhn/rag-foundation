from __future__ import annotations

import csv
import hashlib
import re
import subprocess
import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
WIKI = ROOT / "wiki"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def digest_tree(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class WikiRiskGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        for script in ("inspect_data.py", "build_entities.py", "build_wiki.py", "validate_wiki.py"):
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / script)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                encoding="utf-8",
            )
            if result.returncode:
                raise AssertionError(f"{script} failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
        cls.entities = read_csv(OUTPUTS / "entities.csv")
        cls.relations = read_csv(OUTPUTS / "relations.csv")

    def test_expected_entity_counts_and_unique_ids(self) -> None:
        self.assertEqual(
            Counter(row["type"] for row in self.entities),
            Counter({"RuiRo": 12, "KiemSoat": 10, "SuKienRuiRo": 12}),
        )
        ids = [row["id"] for row in self.entities]
        self.assertEqual(len(ids), len(set(ids)))

    def test_entity_schema_and_provenance_are_preserved(self) -> None:
        required = {"id", "type", "name", "description", "source_file", "data_origin", "verification_status"}
        self.assertTrue(required.issubset(self.entities[0]))
        self.assertTrue(all(row["source_file"] for row in self.entities))
        self.assertTrue(all(row["data_origin"] for row in self.entities))
        self.assertTrue(all(row["verification_status"] for row in self.entities))

    def test_expected_relation_counts_types_and_endpoints(self) -> None:
        self.assertEqual(
            Counter(row["relationship_type"] for row in self.relations),
            Counter({"MITIGATES": 10, "OBSERVED_AS": 12}),
        )
        ids = {row["id"] for row in self.entities}
        for relation in self.relations:
            self.assertIn(relation["source_id"], ids)
            self.assertIn(relation["target_id"], ids)
            self.assertTrue(relation["evidence_quote"])
            self.assertTrue(relation["verification_status"])

    def test_relationship_direction_and_entity_types(self) -> None:
        types = {row["id"]: row["type"] for row in self.entities}
        expected = {"MITIGATES": ("KiemSoat", "RuiRo"), "OBSERVED_AS": ("RuiRo", "SuKienRuiRo")}
        for relation in self.relations:
            source_type, target_type = expected[relation["relationship_type"]]
            self.assertEqual(types[relation["source_id"]], source_type)
            self.assertEqual(types[relation["target_id"]], target_type)

    def test_wiki_page_count_frontmatter_and_links(self) -> None:
        pages = list(WIKI.rglob("*.md"))
        self.assertEqual(len(pages), 35)
        entity_pages = [page for page in pages if page.name != "Home.md"]
        for page in entity_pages:
            text = page.read_text(encoding="utf-8")
            self.assertRegex(text, r"(?m)^---$")
            self.assertRegex(text, r'(?m)^id: "[^"\n]+"$')
            self.assertRegex(text, r'(?m)^type: "(RuiRo|KiemSoat|SuKienRuiRo)"$')
            self.assertIn("[[", text)

    def test_all_wikilinks_resolve(self) -> None:
        pages = list(WIKI.rglob("*.md"))
        targets = {page.relative_to(WIKI).with_suffix("").as_posix() for page in pages}
        basenames = {Path(target).name for target in targets}
        link_re = re.compile(r"\[\[([^\]|#]+)")
        broken: list[str] = []
        for page in pages:
            for target in link_re.findall(page.read_text(encoding="utf-8")):
                normalized = target.removesuffix(".md")
                if normalized not in targets and Path(normalized).name not in basenames:
                    broken.append(f"{page}: {target}")
        self.assertEqual(broken, [])

    def test_validation_report_discloses_data_gaps(self) -> None:
        report = (OUTPUTS / "wiki_validation_report.md").read_text(encoding="utf-8")
        self.assertIn("Lỗi chương trình: 0", report)
        self.assertIn("Trạng thái: ĐẠT", report)
        self.assertIn("`RR-011`", report)
        self.assertIn("`RR-012`", report)

    def test_pipeline_is_idempotent(self) -> None:
        generated = [OUTPUTS / "entities.csv", OUTPUTS / "relations.csv"] + list(WIKI.rglob("*.md"))
        before = digest_tree(generated)
        for script in ("build_entities.py", "build_wiki.py", "validate_wiki.py"):
            subprocess.run([sys.executable, str(ROOT / "scripts" / script)], cwd=ROOT, check=True, capture_output=True)
        after = digest_tree(generated)
        self.assertEqual(before, after)

    def test_neo4j_assets_meet_safety_requirements(self) -> None:
        schema = (ROOT / "cypher" / "schema.cypher").read_text(encoding="utf-8")
        loader = (ROOT / "scripts" / "load_neo4j.py").read_text(encoding="utf-8")
        self.assertEqual(schema.count("IS UNIQUE"), 3)
        self.assertIn("MERGE", loader)
        self.assertIn("$properties", loader)
        for variable in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "NEO4J_DATABASE"):
            self.assertIn(variable, loader)
        self.assertNotRegex(loader, r"(?i)password\s*=\s*['\"][^'\"]+['\"]")


if __name__ == "__main__":
    unittest.main(verbosity=2)
