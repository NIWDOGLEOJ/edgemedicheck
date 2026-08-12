#!/usr/bin/env python3
"""
Test suite for EdgeMediCheck.

Focused on the logic that decides whether a medicine is safe to dispense:
date parsing across Indian label formats, database verification precedence,
and decision fusion. These are the parts where a bug is a patient-safety bug
rather than a cosmetic one.

Run with:
    python tests/test_pipeline.py
    python -m pytest tests/test_pipeline.py -v      (if pytest is installed)
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from edgemedicheck import database as db
from edgemedicheck.cnn import VisualResult
from edgemedicheck.dateparse import (
    ParsedDate, dates_agree, days_until_expiry, is_expired, parse_date_string,
)
from edgemedicheck.fusion import (
    GREEN, INFO, RED, YELLOW, R_EXPIRED_DB, R_EXPIRED_LABEL, R_NO_MODEL,
    R_OCR_UNCERTAIN, R_RECALLED, R_UNKNOWN_BATCH, fuse,
)
from edgemedicheck.ocr import OCRField, OCRResult


# ==========================================================================
# Date parsing
# ==========================================================================


class TestDateParsing(unittest.TestCase):

    def test_month_year_formats(self):
        cases = [
            ("08/2027", 2027, 8), ("08-2027", 2027, 8), ("08.2027", 2027, 8),
            ("08/27", 2027, 8), ("AUG 2027", 2027, 8), ("AUG-27", 2027, 8),
            ("August 2027", 2027, 8), ("aug/2027", 2027, 8),
        ]
        for text, year, month in cases:
            with self.subTest(text=text):
                p = parse_date_string(text, "exp")
                self.assertIsNotNone(p, f"failed to parse {text!r}")
                self.assertEqual((p.year, p.month), (year, month))

    def test_day_month_year_indian_convention(self):
        """DD/MM/YYYY is the Indian convention and must win when ambiguous."""
        p = parse_date_string("05/09/2028", "exp")
        self.assertEqual((p.year, p.month, p.day), (2028, 9, 5))

    def test_unambiguous_us_order_still_parses(self):
        """First field > 12 cannot be a day-of-month ordering issue."""
        p = parse_date_string("13/08/2027", "exp")
        self.assertIsNotNone(p)
        self.assertEqual((p.year, p.month, p.day), (2027, 8, 13))

    def test_leftmost_date_wins(self):
        """Regression: label windows contain more than one date.

        On `MFG MAY-26 EXP 05/09/2028`, scanning pattern-by-pattern matched the
        numeric expiry first and assigned it as the manufacturing date.
        """
        p = parse_date_string("MAY-26 EXP 05/09/2028", "mfg")
        self.assertEqual((p.year, p.month), (2026, 5))

    def test_month_only_expiry_valid_to_end_of_month(self):
        """A pack marked 08/2027 is dispensable through 31 August 2027."""
        p = parse_date_string("08/2027", "exp")
        self.assertEqual(p.effective, date(2027, 8, 31))
        self.assertFalse(is_expired(p, date(2027, 8, 31)))
        self.assertTrue(is_expired(p, date(2027, 9, 1)))

    def test_month_only_mfg_starts_at_month_start(self):
        p = parse_date_string("03/2025", "mfg")
        self.assertEqual(p.effective, date(2025, 3, 1))

    def test_rejects_impossible_dates(self):
        for text in ("13/2027", "00/2027", "32/08/2027", "08/1850"):
            with self.subTest(text=text):
                p = parse_date_string(text, "exp")
                if p is not None:
                    self.assertTrue(1 <= p.month <= 12)
                    self.assertTrue(2000 <= p.year <= 2099)

    def test_ocr_digit_confusion_repair(self):
        """O->0 and l->1 are the common thermal-print misreads."""
        p = parse_date_string("O8/2O27", "exp")
        self.assertIsNotNone(p)
        self.assertEqual((p.year, p.month), (2027, 8))

    def test_dates_agree_at_month_granularity(self):
        label = parse_date_string("08/2027", "exp")
        record = ParsedDate(2027, 8, 31, "2027-08-31", "exp")
        self.assertTrue(dates_agree(label, record))

    def test_dates_disagree_across_months(self):
        a = parse_date_string("08/2027", "exp")
        b = parse_date_string("11/2027", "exp")
        self.assertFalse(dates_agree(a, b))

    def test_days_until_expiry(self):
        p = parse_date_string("08/2027", "exp")
        self.assertEqual(days_until_expiry(p, date(2027, 8, 1)), 30)

    def test_empty_and_garbage_input(self):
        for text in ("", "   ", "no date here", "!!!"):
            self.assertIsNone(parse_date_string(text, "exp"))


# ==========================================================================
# Database verification
# ==========================================================================


class TestDatabaseVerification(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
        self.tmp.close()
        self.db = self.tmp.name
        db.init_db(self.db)
        db.upsert_product(
            "PARACIP", "Cipla Pharmaceuticals Ltd", "PC24101",
            exp_date="2027-01-31", mfg_date="2024-01-01", db_path=self.db,
        )
        db.upsert_product(
            "PANTOCID", "Sun Pharma Laboratories Ltd", "PT24777",
            exp_date="2027-07-31", status=db.RECALLED, db_path=self.db,
        )
        db.upsert_product(
            "CROCIN", "Sun Pharma Laboratories Ltd", "CR22045",
            exp_date="2024-04-30", status=db.EXPIRED, db_path=self.db,
        )

    def tearDown(self):
        Path(self.db).unlink(missing_ok=True)

    def test_valid_batch(self):
        r = db.verify("PC24101", parse_date_string("01/2027", "exp"),
                      today=date(2026, 6, 1), db_path=self.db)
        self.assertEqual(r.status, db.MATCH_VALID)

    def test_unknown_batch_is_not_counterfeit(self):
        """A missing record means missing data, not a fake pack."""
        r = db.verify("ZZ99999", today=date(2026, 6, 1), db_path=self.db)
        self.assertEqual(r.status, db.MATCH_UNKNOWN)

    def test_recalled_batch_is_unsafe(self):
        r = db.verify("PT24777", today=date(2026, 6, 1), db_path=self.db)
        self.assertEqual(r.status, db.MATCH_UNSAFE)

    def test_recall_outranks_valid_dates(self):
        """A recalled batch is unsafe even when its expiry is still in future."""
        r = db.verify("PT24777", parse_date_string("07/2027", "exp"),
                      today=date(2026, 6, 1), db_path=self.db)
        self.assertEqual(r.status, db.MATCH_UNSAFE)

    def test_expired_status(self):
        r = db.verify("CR22045", today=date(2026, 6, 1), db_path=self.db)
        self.assertEqual(r.status, db.MATCH_EXPIRED)

    def test_expiry_mismatch_detects_tampered_label(self):
        """Label says 2029, record says 2027 -- the relabelling case."""
        r = db.verify("PC24101", parse_date_string("01/2029", "exp"),
                      today=date(2026, 6, 1), db_path=self.db)
        self.assertEqual(r.status, db.MATCH_MISMATCH)

    def test_manufacturer_variation_is_tolerated(self):
        """OCR mangles logo type; near-matches must not raise a hard flag."""
        r = db.verify("PC24101", parse_date_string("01/2027", "exp"),
                      manufacturer="CIPLA PHARMA LTD.",
                      today=date(2026, 6, 1), db_path=self.db)
        self.assertEqual(r.status, db.MATCH_VALID)

    def test_merged_ocr_line_does_not_condemn_pack(self):
        """Regression: neighbouring label lines merge into the manufacturer.

        This previously produced a hard mismatch and a RED verdict on a
        genuine pack.
        """
        r = db.verify(
            "PC24101", parse_date_string("01/2027", "exp"),
            manufacturer="Schedule H Prescription Drug - Caution Cipla "
                         "Pharmaceuticals Ltd",
            today=date(2026, 6, 1), db_path=self.db,
        )
        self.assertEqual(r.status, db.MATCH_VALID)

    def test_wrong_manufacturer_is_soft_not_hard(self):
        r = db.verify("PC24101", parse_date_string("01/2027", "exp"),
                      manufacturer="Totally Different Remedies",
                      today=date(2026, 6, 1), db_path=self.db)
        self.assertEqual(r.status, db.MATCH_NAME_MISMATCH)

    def test_no_batch_number(self):
        r = db.verify(None, today=date(2026, 6, 1), db_path=self.db)
        self.assertEqual(r.status, db.MATCH_UNKNOWN)

    def test_scan_log_roundtrip(self):
        db.log_scan({
            "timestamp": "2026-06-01T10:00:00", "verdict": "green",
            "reason_code": "OK", "reason_text": "No issue detected.",
            "batch_number": "PC24101", "latency_ms": 812.0,
        }, db_path=self.db)
        rows = db.recent_scans(5, self.db)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["batch_number"], "PC24101")
        self.assertEqual(db.scan_stats(self.db)["total_scans"], 1)


# ==========================================================================
# Decision fusion
# ==========================================================================


def make_ocr(batch="PC24101", exp="01/2027", status="complete") -> OCRResult:
    r = OCRResult(status=status, mean_confidence=85.0)
    r.fields["batch_number"] = OCRField("batch_number", batch, 90.0)
    parsed = parse_date_string(exp, "exp") if exp else None
    r.fields["exp_date"] = OCRField("exp_date", exp, 90.0, parsed_date=parsed)
    r.fields["mfg_date"] = OCRField("mfg_date")
    r.fields["manufacturer"] = OCRField("manufacturer")
    r.fields["product_name"] = OCRField("product_name", "PARACIP", 80.0)
    return r


def visual(score=0.1, usable=True, backend="heuristic") -> VisualResult:
    return VisualResult(
        suspicion_score=score, backend=backend,
        label="suspicious" if score >= 0.65 else "genuine", usable=usable,
    )


GOOD_LOOKUP = db.LookupResult(db.MATCH_VALID, reason="ok")
TODAY = date(2026, 6, 1)


class TestFusion(unittest.TestCase):

    def test_all_clear_is_green(self):
        v = fuse(make_ocr(), GOOD_LOOKUP, visual(0.1),
                 sharpness=200, brightness=130, today=TODAY)
        self.assertEqual(v.verdict, GREEN)

    def test_expired_label_is_red(self):
        v = fuse(make_ocr(exp="01/2024"), GOOD_LOOKUP, visual(0.1),
                 sharpness=200, brightness=130, today=TODAY)
        self.assertEqual(v.verdict, RED)
        self.assertEqual(v.reason_code, R_EXPIRED_LABEL)

    def test_expired_beats_clean_visual(self):
        """A genuine but expired pack is still unsafe to dispense."""
        v = fuse(make_ocr(exp="01/2024"), GOOD_LOOKUP, visual(0.01),
                 sharpness=300, brightness=130, today=TODAY)
        self.assertEqual(v.verdict, RED)

    def test_recalled_is_red(self):
        lookup = db.LookupResult(db.MATCH_UNSAFE, reason="recalled")
        v = fuse(make_ocr(), lookup, visual(0.1),
                 sharpness=200, brightness=130, today=TODAY)
        self.assertEqual(v.verdict, RED)
        self.assertEqual(v.reason_code, R_RECALLED)

    def test_unknown_batch_is_yellow_not_red(self):
        """The local database is not a national registry."""
        lookup = db.LookupResult(db.MATCH_UNKNOWN, reason="not found")
        v = fuse(make_ocr(), lookup, visual(0.1),
                 sharpness=200, brightness=130, today=TODAY)
        self.assertEqual(v.verdict, YELLOW)
        self.assertEqual(v.reason_code, R_UNKNOWN_BATCH)

    def test_uncertain_ocr_is_never_green(self):
        """Absence of evidence must not read as evidence of safety."""
        v = fuse(make_ocr(status="uncertain"), GOOD_LOOKUP, visual(0.1),
                 sharpness=200, brightness=130, today=TODAY)
        self.assertEqual(v.verdict, YELLOW)
        self.assertEqual(v.reason_code, R_OCR_UNCERTAIN)

    def test_abstaining_visual_does_not_block_green(self):
        """An unusable visual stream is informational, not a downgrade."""
        v = fuse(make_ocr(), GOOD_LOOKUP, visual(0.5, usable=False),
                 sharpness=200, brightness=130, today=TODAY)
        self.assertEqual(v.verdict, GREEN)
        codes = [f.code for f in v.findings]
        self.assertIn(R_NO_MODEL, codes)
        self.assertTrue(
            all(f.verdict == INFO for f in v.findings if f.code == R_NO_MODEL)
        )

    def test_heuristic_high_score_is_yellow_not_red(self):
        """Without a trained model we may prompt review, never accuse."""
        v = fuse(make_ocr(), GOOD_LOOKUP, visual(0.95, backend="heuristic"),
                 sharpness=200, brightness=130, today=TODAY)
        self.assertEqual(v.verdict, YELLOW)

    def test_model_backed_high_score_is_red(self):
        v = fuse(make_ocr(), GOOD_LOOKUP, visual(0.95, backend="tflite"),
                 sharpness=200, brightness=130, today=TODAY)
        self.assertEqual(v.verdict, RED)

    def test_blurred_capture_is_yellow(self):
        v = fuse(make_ocr(), GOOD_LOOKUP, visual(0.1),
                 sharpness=5, brightness=130, today=TODAY)
        self.assertEqual(v.verdict, YELLOW)

    def test_package_outside_scan_region_is_yellow(self):
        v = fuse(make_ocr(), GOOD_LOOKUP, visual(0.1), sharpness=200,
                 brightness=130, region_ok=False, today=TODAY)
        self.assertEqual(v.verdict, YELLOW)

    def test_near_expiry_is_yellow(self):
        v = fuse(make_ocr(exp="06/2026"), GOOD_LOOKUP, visual(0.1),
                 sharpness=200, brightness=130, today=date(2026, 6, 20))
        self.assertEqual(v.verdict, YELLOW)

    def test_red_outranks_yellow(self):
        lookup = db.LookupResult(db.MATCH_UNSAFE, reason="recalled")
        v = fuse(make_ocr(status="uncertain"), lookup, visual(0.5),
                 sharpness=5, brightness=130, today=TODAY)
        self.assertEqual(v.verdict, RED)

    def test_verdict_serialises(self):
        v = fuse(make_ocr(), GOOD_LOOKUP, visual(0.1),
                 sharpness=200, brightness=130, today=TODAY)
        d = v.to_dict()
        for key in ("verdict", "headline", "reason_code", "advice", "findings"):
            self.assertIn(key, d)


# ==========================================================================
# Preprocessing
# ==========================================================================


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        from edgemedicheck.preprocess import preprocess

        self.preprocess = preprocess
        rng = np.random.default_rng(0)
        img = np.full((400, 700, 3), 235, dtype=np.uint8)
        img[80:320, 60:640] = 250
        # Text-like dark bars give deskew and sharpness something to measure.
        for y in range(120, 300, 40):
            img[y:y + 12, 90:600] = 25
        img = np.clip(
            img.astype(np.int16) + rng.integers(-4, 5, img.shape), 0, 255
        ).astype(np.uint8)
        self.image = img

    def test_produces_both_streams(self):
        p = self.preprocess(self.image)
        self.assertEqual(p.ocr_image.ndim, 2, "OCR stream must be grayscale")
        self.assertEqual(p.cnn_image.shape, (224, 224, 3))
        self.assertEqual(p.cnn_image.dtype, np.float32)
        self.assertTrue(0.0 <= p.cnn_image.min() and p.cnn_image.max() <= 1.0)

    def test_quality_metrics_present(self):
        p = self.preprocess(self.image)
        self.assertGreater(p.sharpness, 0)
        self.assertGreater(p.brightness, 0)

    def test_handles_tiny_image(self):
        tiny = np.full((40, 60, 3), 200, dtype=np.uint8)
        p = self.preprocess(tiny)
        self.assertEqual(p.cnn_image.shape, (224, 224, 3))


# ==========================================================================
# Visual calibration
# ==========================================================================


class TestCalibration(unittest.TestCase):

    def test_pooled_global_does_not_score_unseen_product(self):
        """Regression: judging an unseen product against a pooled reference
        set flagged genuine stock as anomalous."""
        from edgemedicheck.cnn import Calibration, CalibrationSet

        c = Calibration({"edge_energy": 100.0}, {"edge_energy": 10.0}, 20)
        cs = CalibrationSet(
            groups={"PARACIP": c}, global_calibration=c, global_is_pooled=True
        )
        _, scope = cs.get("SOMETHING_ELSE")
        self.assertEqual(scope, "none")
        _, scope = cs.get("PARACIP")
        self.assertEqual(scope, "group")

    def test_flat_global_is_used_when_not_pooled(self):
        from edgemedicheck.cnn import Calibration, CalibrationSet

        c = Calibration({"edge_energy": 100.0}, {"edge_energy": 10.0}, 20)
        cs = CalibrationSet(groups={}, global_calibration=c,
                            global_is_pooled=False)
        _, scope = cs.get("ANY")
        self.assertEqual(scope, "global")

    def test_untrustworthy_calibration_abstains(self):
        from edgemedicheck.cnn import Calibration, CalibrationSet

        c = Calibration({"edge_energy": 100.0}, {"edge_energy": 10.0}, 3)
        cs = CalibrationSet(groups={}, global_calibration=c)
        _, scope = cs.get("ANY")
        self.assertEqual(scope, "none")

    def test_shrinkage_floors_tiny_scales(self):
        """A homogeneous group can collapse its MAD towards zero."""
        from edgemedicheck.cnn import Calibration, shrink_towards

        group = Calibration({"a": 1.0}, {"a": 0.0001}, 20)
        pooled = Calibration({"a": 1.0}, {"a": 0.5}, 100)
        out = shrink_towards(group, pooled, weight=0.5)
        self.assertAlmostEqual(out.scales["a"], 0.25)


# ==========================================================================
# Barcode / GS1
# ==========================================================================


class TestGS1Parsing(unittest.TestCase):

    def setUp(self):
        from edgemedicheck import barcode

        self.bc = barcode
        self.GS = barcode.GS

    def test_unseparated_concatenation(self):
        """Fixed-length AIs make an unseparated payload parseable."""
        c = self.bc.interpret("01034531200000031725093010ABC123", "QR", "t")
        self.assertTrue(c.is_gs1)
        self.assertEqual(c.gtin, "03453120000003")
        self.assertEqual(c.batch, "ABC123")
        self.assertEqual(c.exp_date.iso, "2025-09-30")

    def test_fnc1_separated(self):
        payload = ("010345312000000317250930" + self.GS + "10ABC123"
                   + self.GS + "21SER0001")
        c = self.bc.interpret(payload, "DATAMATRIX", "t")
        self.assertEqual(c.batch, "ABC123")
        self.assertEqual(c.serial, "SER0001")

    def test_bracketed_human_readable_form(self):
        c = self.bc.interpret(
            "(01)03453120000003(17)250930(10)ABC123", "QR", "t"
        )
        self.assertEqual(c.batch, "ABC123")
        self.assertEqual(c.exp_date.iso, "2025-09-30")

    def test_day_zero_means_end_of_month(self):
        """GS1 permits DD=00; it means the last day of that month."""
        c = self.bc.interpret("010345312000000317250900" + self.GS + "10L1",
                              "QR", "t")
        self.assertEqual(c.exp_date.iso, "2025-09")
        self.assertEqual(c.exp_date.effective, date(2025, 9, 30))

    def test_production_and_expiry_dates(self):
        payload = ("011234567890123111240115" + self.GS + "17260731"
                   + self.GS + "10B7")
        c = self.bc.interpret(payload, "QR", "t")
        self.assertEqual(c.mfg_date.iso, "2024-01-15")
        self.assertEqual(c.exp_date.iso, "2026-07-31")

    def test_url_payload_is_not_forced_into_gs1(self):
        """Not every pack QR is GS1; some encode a verification URL."""
        c = self.bc.interpret("https://verify.example.in/x?c=1", "QR", "t")
        self.assertFalse(c.is_gs1)
        self.assertIsNone(c.batch)
        self.assertTrue(c.parse_notes)

    def test_plain_barcode_is_not_gs1(self):
        c = self.bc.interpret("8901234567890", "EAN13", "t")
        self.assertFalse(c.is_gs1)

    def test_rejects_impossible_encoded_date(self):
        c = self.bc.interpret("010345312000000317259931" + self.GS + "10L",
                              "QR", "t")
        self.assertIsNone(c.exp_date)


class TestBarcodeCrossCheck(unittest.TestCase):
    """The cross-check is the tamper detector, so its failure modes matter."""

    def setUp(self):
        from edgemedicheck import barcode

        self.bc = barcode
        payload = "010345312000000317270831" + barcode.GS + "10RF0159"
        code = barcode.interpret(payload, "QR", "t")
        self.result = barcode.BarcodeResult(codes=[code])

    def check(self, batch, exp):
        return self.bc.cross_check(
            self.result, batch, parse_date_string(exp, "exp") if exp else None
        )

    def test_matching_text_agrees(self):
        self.assertEqual(self.check("RF0159", "08/2027").status, "agree")

    def test_ocr_character_confusion_is_not_a_conflict(self):
        """`RF0159` misread as `RFO159` is an OCR artefact, not tampering."""
        self.assertEqual(self.check("RFO159", "08/2027").status, "agree")

    def test_altered_expiry_is_a_conflict(self):
        cc = self.check("RF0159", "08/2029")
        self.assertTrue(cc.is_conflict)
        self.assertTrue(any("expiry" in c.lower() for c in cc.conflicts))

    def test_wrong_batch_is_a_conflict(self):
        self.assertTrue(self.check("XX9999", "08/2027").is_conflict)

    def test_one_field_only_is_partial(self):
        self.assertEqual(self.check(None, "08/2027").status, "partial")

    def test_nothing_to_compare(self):
        self.assertEqual(self.check(None, None).status, "not_comparable")

    def test_no_code_at_all(self):
        empty = self.bc.BarcodeResult()
        cc = self.bc.cross_check(empty, "RF0159", None)
        self.assertEqual(cc.status, "no_code")

    def test_gs1_code_outranks_plain_barcode_as_primary(self):
        plain = self.bc.interpret("8901234567890", "EAN13", "t")
        mixed = self.bc.BarcodeResult(codes=[plain] + self.result.codes)
        self.assertTrue(mixed.primary.is_gs1)
        self.assertEqual(mixed.batch, "RF0159")


class TestBarcodeFusion(unittest.TestCase):

    def setUp(self):
        from edgemedicheck import barcode
        from edgemedicheck.fusion import R_CODE_CONFLICT, R_CODE_EXPIRED

        self.bc = barcode
        self.R_CODE_CONFLICT = R_CODE_CONFLICT
        self.R_CODE_EXPIRED = R_CODE_EXPIRED

    def test_conflict_produces_red(self):
        cc = self.bc.CrossCheck("conflict", ["printed expiry differs"])
        v = fuse(make_ocr(), GOOD_LOOKUP, visual(0.1), sharpness=200,
                 brightness=130, today=TODAY, crosscheck=cc)
        self.assertEqual(v.verdict, RED)
        self.assertEqual(v.reason_code, self.R_CODE_CONFLICT)

    def test_conflict_outranks_plain_expiry(self):
        """Tampering is a more serious finding than stock that aged out."""
        cc = self.bc.CrossCheck("conflict", ["printed expiry differs"])
        v = fuse(make_ocr(exp="01/2024"), GOOD_LOOKUP, visual(0.1),
                 sharpness=200, brightness=130, today=TODAY, crosscheck=cc)
        self.assertEqual(v.reason_code, self.R_CODE_CONFLICT)

    def test_expired_barcode_is_red(self):
        code = self.bc.interpret(
            "010345312000000317240131" + self.bc.GS + "10L1", "QR", "t"
        )
        result = self.bc.BarcodeResult(codes=[code])
        v = fuse(make_ocr(exp="01/2027"), GOOD_LOOKUP, visual(0.1),
                 sharpness=200, brightness=130, today=TODAY, barcode=result)
        self.assertEqual(v.verdict, RED)

    def test_agreement_is_informational_only(self):
        cc = self.bc.CrossCheck("agree", [], ["batch matches"])
        v = fuse(make_ocr(), GOOD_LOOKUP, visual(0.1), sharpness=200,
                 brightness=130, today=TODAY, crosscheck=cc)
        self.assertEqual(v.verdict, GREEN)

    def test_missing_code_does_not_penalise_the_pack(self):
        """Plenty of legitimate packaging carries no 2D code."""
        v = fuse(make_ocr(), GOOD_LOOKUP, visual(0.1), sharpness=200,
                 brightness=130, today=TODAY,
                 barcode=self.bc.BarcodeResult(),
                 crosscheck=self.bc.CrossCheck("no_code"))
        self.assertEqual(v.verdict, GREEN)


# ==========================================================================
# Pharmacist feedback
# ==========================================================================


class TestFeedback(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
        self.tmp.close()
        self.db = self.tmp.name
        db.init_db(self.db)
        self.scan_id = db.log_scan({
            "timestamp": "2026-06-01T10:00:00", "verdict": "green",
            "reason_code": "OK", "batch_number": "PC24101",
            "image_path": "/tmp/x.jpg", "latency_ms": 700.0,
        }, db_path=self.db)

    def tearDown(self):
        Path(self.db).unlink(missing_ok=True)

    def test_record_and_report(self):
        db.record_feedback(
            scan_id=self.scan_id, system_verdict="green",
            correct_verdict="red", correct_label=db.LABEL_SUSPICIOUS,
            batch_number="PC24101", image_path="/tmp/x.jpg", db_path=self.db,
        )
        stats = db.feedback_stats(self.db)
        self.assertEqual(stats["total"], 1)
        self.assertEqual(stats["pending_export"], 1)
        self.assertEqual(stats["disagreements"]["green->red"], 1)

    def test_export_marks_records_consumed(self):
        fid = db.record_feedback(
            scan_id=self.scan_id, system_verdict="green",
            correct_verdict="red", correct_label=db.LABEL_SUSPICIOUS,
            db_path=self.db,
        )
        self.assertEqual(len(db.pending_feedback(self.db)), 1)
        db.mark_feedback_exported([fid], self.db)
        self.assertEqual(len(db.pending_feedback(self.db)), 0)
        self.assertEqual(len(db.pending_feedback(self.db, include_exported=True)), 1)

    def test_latest_scan_is_the_correction_target(self):
        latest = db.latest_scan(self.db)
        self.assertEqual(latest["scan_id"], self.scan_id)

    def test_migration_adds_columns_to_an_old_database(self):
        """A deployed scanner must keep its audit history across upgrades."""
        import sqlite3

        tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
        tmp.close()
        conn = sqlite3.connect(tmp.name)
        # A pre-barcode scan_log table.
        conn.execute(
            "CREATE TABLE scan_log (scan_id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " timestamp TEXT NOT NULL, verdict TEXT NOT NULL,"
            " reason_code TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO scan_log (timestamp, verdict, reason_code) "
            "VALUES ('2026-01-01T00:00:00','green','OK')"
        )
        conn.commit()
        conn.close()

        db.init_db(tmp.name)

        conn = sqlite3.connect(tmp.name)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(scan_log)")}
        rows = conn.execute("SELECT COUNT(*) FROM scan_log").fetchone()[0]
        conn.close()
        Path(tmp.name).unlink(missing_ok=True)

        self.assertIn("code_batch", cols)
        self.assertIn("crosscheck", cols)
        self.assertEqual(rows, 1, "existing scan history must survive migration")


# ==========================================================================
# Dataset splitting
# ==========================================================================


class SplitTestBase(unittest.TestCase):
    """Builds a small on-disk dataset with real pack structure."""

    def setUp(self):
        import shutil

        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def make(self, products=6, genuine_packs=3, surrogate_packs=2, shots=4,
             manifest=True):
        import json as _json

        entries = []
        for i in range(products):
            product = f"PROD{i:02d}"
            batch = f"B{i:02d}"
            for label, packs in (("genuine", range(1, genuine_packs + 1)),
                                 ("suspicious",
                                  range(genuine_packs + 1,
                                        genuine_packs + surrogate_packs + 1))):
                for pack in packs:
                    for shot in range(shots):
                        name = f"{product}__{batch}__p{pack:02d}__{shot:02d}.jpg"
                        d = self.root / label
                        d.mkdir(parents=True, exist_ok=True)
                        (d / name).write_bytes(b"\xff\xd8\xff\xd9")  # stub JPEG
                        entries.append({
                            "file": f"{label}/{name}",
                            "product_name": product, "batch_number": batch,
                            "pack_id": pack, "label": label,
                        })
        if manifest:
            (self.root / "manifest.json").write_text(_json.dumps(entries))
        return entries


class TestSplitLoading(SplitTestBase):

    def setUp(self):
        super().setUp()
        from edgemedicheck import splits

        self.splits = splits

    def test_loads_labels_and_pack_identity(self):
        self.make()
        samples = self.splits.load_samples(self.root)
        self.assertEqual(len(samples), 6 * 5 * 4)
        self.assertEqual({s.label for s in samples}, {"genuine", "suspicious"})
        self.assertEqual(len({s.product for s in samples}), 6)
        self.assertEqual(len({s.pack_id for s in samples}), 6 * 5)

    def test_colliding_basenames_do_not_overwrite_labels(self):
        """Regression: `genuine/X.jpg` and `suspicious/X.jpg` share a basename.

        A manifest keyed only by basename let one class silently overwrite the
        other, so every image loaded with a single label.
        """
        import json as _json

        for label in ("genuine", "suspicious"):
            d = self.root / label
            d.mkdir(parents=True, exist_ok=True)
            (d / "SAME__B1__p01__00.jpg").write_bytes(b"\xff\xd8\xff\xd9")

        # Manifest deliberately stores bare basenames, as an older or
        # hand-written one would.
        (self.root / "manifest.json").write_text(_json.dumps([
            {"file": "SAME__B1__p01__00.jpg", "product_name": "SAME",
             "batch_number": "B1", "pack_id": 1, "label": "genuine"},
            {"file": "SAME__B1__p01__00.jpg", "product_name": "SAME",
             "batch_number": "B1", "pack_id": 1, "label": "suspicious"},
        ]))

        samples = self.splits.load_samples(self.root)
        self.assertEqual(len(samples), 2)
        self.assertEqual({s.label for s in samples}, {"genuine", "suspicious"})

    def test_directory_label_wins_over_manifest(self):
        """Directory placement is unambiguous; the manifest is not."""
        import json as _json

        d = self.root / "suspicious"
        d.mkdir(parents=True, exist_ok=True)
        (d / "A__B__p01__00.jpg").write_bytes(b"\xff\xd8\xff\xd9")
        (self.root / "manifest.json").write_text(_json.dumps([
            {"file": "A__B__p01__00.jpg", "product_name": "A",
             "batch_number": "B", "pack_id": 1, "label": "genuine"},
        ]))
        samples = self.splits.load_samples(self.root)
        self.assertEqual(samples[0].label, "suspicious")

    def test_collect_layout_is_not_double_counted(self):
        """`collect` writes each image to dataset/ and by_product/."""
        ds = self.root / "dataset" / "genuine"
        bp = self.root / "by_product" / "PROD00"
        ds.mkdir(parents=True)
        bp.mkdir(parents=True)
        name = "PROD00__B00__p01__00.jpg"
        (ds / name).write_bytes(b"\xff\xd8\xff\xd9")
        (bp / name).write_bytes(b"\xff\xd8\xff\xd9")

        samples = self.splits.load_samples(self.root)
        self.assertEqual(len(samples), 1)

    def test_pack_identity_recovered_from_filename(self):
        self.make(manifest=False)
        samples = self.splits.load_samples(self.root)
        self.assertTrue(all(s.source == "filename" for s in samples))
        self.assertEqual(len({s.pack_id for s in samples}), 6 * 5)

    def test_missing_identity_falls_back_coarsely(self):
        """The fallback must merge groups, never split them.

        Merging is conservative; treating each image as its own pack would
        silently reintroduce the leakage this module prevents.
        """
        d = self.root / "genuine"
        d.mkdir(parents=True)
        for i in range(6):
            (d / f"random_photo_{i}.png").write_bytes(b"\x89PNG\r\n\x1a\n")

        samples = self.splits.load_samples(self.root)
        self.assertEqual(len(samples), 6)
        self.assertEqual(len({s.pack_id for s in samples}), 1,
                         "unidentified images must group together, not apart")


class TestPackSplit(SplitTestBase):

    def setUp(self):
        super().setUp()
        from edgemedicheck import splits

        self.splits = splits
        self.make()
        self.samples = splits.load_samples(self.root)

    def test_no_pack_spans_folds(self):
        split = self.splits.split_by_pack(self.samples, 0.2, 0.2, seed=1)
        train = {s.pack_id for s in split.train}
        val = {s.pack_id for s in split.val}
        test = {s.pack_id for s in split.test}
        self.assertFalse(train & val)
        self.assertFalse(train & test)
        self.assertFalse(val & test)

    def test_every_image_is_used_exactly_once(self):
        split = self.splits.split_by_pack(self.samples, 0.2, 0.2, seed=1)
        placed = [s.path for f in split.folds.values() for s in f]
        self.assertEqual(len(placed), len(self.samples))
        self.assertEqual(len(set(placed)), len(self.samples))

    def test_both_classes_present_in_each_fold(self):
        split = self.splits.split_by_pack(self.samples, 0.2, 0.2, seed=1)
        for name, fold in split.folds.items():
            self.assertEqual({s.label for s in fold},
                             {"genuine", "suspicious"}, f"fold {name}")

    def test_deterministic_for_a_given_seed(self):
        a = self.splits.split_by_pack(self.samples, 0.2, 0.2, seed=7)
        b = self.splits.split_by_pack(self.samples, 0.2, 0.2, seed=7)
        self.assertEqual(a.paths("val"), b.paths("val"))

    def test_different_seeds_give_different_splits(self):
        a = self.splits.split_by_pack(self.samples, 0.2, 0.0, seed=1)
        b = self.splits.split_by_pack(self.samples, 0.2, 0.0, seed=99)
        self.assertNotEqual(a.paths("val"), b.paths("val"))

    def test_leakage_detector_catches_a_bad_split(self):
        split = self.splits.split_by_pack(self.samples, 0.2, 0.0, seed=1)
        split.val.append(split.train[0])  # force a leak
        with self.assertRaises(self.splits.LeakageError):
            self.splits.assert_no_leakage(split)

    def test_split_round_trips_through_json(self):
        split = self.splits.split_by_pack(self.samples, 0.2, 0.2, seed=3)
        path = self.root / "split.json"
        split.save(path)
        reloaded = self.splits.Split.load(path)
        self.assertEqual(split.paths("train"), reloaded.paths("train"))
        self.assertEqual(split.regime, reloaded.regime)
        self.splits.assert_no_leakage(reloaded)


class TestProductSplit(SplitTestBase):

    def setUp(self):
        super().setUp()
        from edgemedicheck import splits

        self.splits = splits
        self.make(products=8)
        self.samples = splits.load_samples(self.root)

    def test_held_out_products_never_appear_in_training(self):
        split = self.splits.split_by_product(self.samples, n_holdout=3, seed=1)
        train = {s.product for s in split.train}
        test = {s.product for s in split.test}
        self.assertFalse(train & test)
        self.assertEqual(len(test), 3)

    def test_validation_is_still_pack_disjoint(self):
        split = self.splits.split_by_product(self.samples, n_holdout=2, seed=1)
        self.assertFalse({s.pack_id for s in split.train}
                         & {s.pack_id for s in split.val})

    def test_explicit_holdout_is_honoured(self):
        split = self.splits.split_by_product(
            self.samples, holdout_products=["PROD00", "PROD01"], seed=1
        )
        self.assertEqual({s.product for s in split.test}, {"PROD00", "PROD01"})

    def test_unknown_holdout_product_raises(self):
        with self.assertRaises(ValueError):
            self.splits.split_by_product(
                self.samples, holdout_products=["NOSUCHPRODUCT"]
            )

    def test_cannot_hold_out_everything(self):
        products = sorted({s.product for s in self.samples})
        split = self.splits.split_by_product(
            self.samples, holdout_products=products[:-1], seed=1
        )
        self.assertTrue(split.train, "at least one product must remain")

    def test_prefers_holding_out_products_carrying_both_classes(self):
        split = self.splits.split_by_product(self.samples, n_holdout=2, seed=1)
        self.assertEqual({s.label for s in split.test},
                         {"genuine", "suspicious"})


class TestSplitAdequacy(SplitTestBase):

    def setUp(self):
        super().setUp()
        from edgemedicheck import splits

        self.splits = splits

    def test_warns_when_too_few_training_packs(self):
        self.make(products=2, genuine_packs=1, surrogate_packs=1, shots=10)
        samples = self.splits.load_samples(self.root)
        split = self.splits.split_by_pack(samples, 0.2, 0.0, seed=1)
        warnings = self.splits.check_adequacy(split)
        self.assertTrue(any("training pack" in w for w in warnings))

    def test_quiet_when_the_dataset_is_adequate(self):
        self.make(products=14, genuine_packs=3, surrogate_packs=3, shots=3)
        samples = self.splits.load_samples(self.root)
        split = self.splits.split_by_pack(samples, 0.2, 0.15, seed=1)
        self.assertEqual(self.splits.check_adequacy(split), [])

    def test_describe_reports_packs_not_just_images(self):
        self.make(products=6)
        samples = self.splits.load_samples(self.root)
        split = self.splits.split_by_pack(samples, 0.2, 0.0, seed=1)
        text = self.splits.describe(split)
        self.assertIn("Effective sample size", text)
        self.assertIn("pack", text.lower())


# ==========================================================================
# Network address detection
# ==========================================================================


class TestLanAddress(unittest.TestCase):
    """The scanner is served to phones and tablets over the pharmacy LAN, so
    printing a wrong address is a real usability failure."""

    def setUp(self):
        import run

        self.run = run

    def test_rejects_unreachable_addresses(self):
        for ip in ("127.0.0.1", "127.0.1.1", "0.0.0.0", "", "169.254.10.2"):
            with self.subTest(ip=ip):
                self.assertFalse(self.run._is_usable_lan_ip(ip))

    def test_accepts_private_ranges(self):
        for ip in ("192.168.1.42", "10.0.0.5", "172.16.3.9"):
            with self.subTest(ip=ip):
                self.assertTrue(self.run._is_usable_lan_ip(ip))

    def test_rejects_apipa(self):
        """169.254.x.x means DHCP was never reached; nothing can route to it."""
        self.assertFalse(self.run._is_usable_lan_ip("169.254.1.1"))

    def test_falls_back_to_interfaces_without_a_gateway(self):
        """An offline pharmacy LAN has no default route.

        The UDP route probe fails there, so interface enumeration has to be
        what finds the address.
        """
        import subprocess

        real_run = subprocess.run

        class FakeCompleted:
            stdout = (
                "1: lo    inet 127.0.0.1/8 scope host lo\n"
                "2: eth0    inet 192.168.7.31/24 scope global eth0\n"
            )

        subprocess.run = lambda *a, **kw: FakeCompleted()
        try:
            self.assertEqual(self.run.lan_ip(), "192.168.7.31")
        finally:
            subprocess.run = real_run


# ==========================================================================
# Hands-free presence detection (live screen)
# ==========================================================================


class TestPresenceDetector(unittest.TestCase):
    """The live screen waits for a pack instead of scanning on a timer.

    A full pipeline pass costs 0.6-1.3 s on a workstation and several seconds
    on a Pi 4, so these decisions determine both what the pharmacist sees and
    how much CPU is burned at an empty counter. Two failures matter: firing a
    scan when nothing is there (which reports amber -- "verify manually" --
    about a pack that does not exist), and dropping a verdict that is still
    being read.
    """

    def setUp(self):
        import cv2

        from edgemedicheck.capture import PresenceDetector

        self.cv2 = cv2
        self.det = PresenceDetector()

    def _empty(self):
        return np.full((720, 1280, 3), 30, np.uint8)

    def _pack(self):
        frame = self._empty()
        self.cv2.rectangle(frame, (300, 150), (980, 560), (240, 240, 235), -1)
        return frame

    def _feed(self, frame, times=1):
        state = None
        for _ in range(times):
            state = self.det.update(frame)
        return state

    def test_empty_scene_is_never_ready(self):
        state = self._feed(self._empty(), 6)
        self.assertFalse(state["present"])
        self.assertFalse(state["ready"])

    def test_pack_becomes_ready_once_it_settles(self):
        self._feed(self._empty(), 3)
        # The frame in which the pack arrives is, by definition, a changed
        # frame: present may latch, but it must not be ready yet.
        first = self._feed(self._pack(), 1)
        self.assertFalse(first["ready"])
        settled = self._feed(self._pack(), 4)
        self.assertTrue(settled["present"])
        self.assertTrue(settled["steady"])
        self.assertTrue(settled["ready"])

    def test_not_ready_while_the_scene_keeps_changing(self):
        """A pack still being moved into place must not be scanned.

        The displacement here is deliberately large. A pack nudged by a few
        pixels genuinely is settled as far as motion blur is concerned, so
        the threshold is meant to ignore it.
        """
        self._feed(self._empty(), 3)
        state = None
        for i in range(6):
            frame = self._empty()
            x = 120 + i * 130
            self.cv2.rectangle(frame, (x, 150), (x + 620, 560),
                               (240, 240, 235), -1)
            state = self.det.update(frame)
            self.assertFalse(state["ready"], f"ready while moving at step {i}")
        self.assertFalse(state["ready"])

    def test_removing_the_pack_returns_to_idle(self):
        self._feed(self._empty(), 3)
        self._feed(self._pack(), 5)
        state = self._feed(self._empty(), 6)
        self.assertFalse(state["present"])
        self.assertFalse(state["ready"])

    def test_ready_drops_immediately_when_the_pack_leaves(self):
        """`present` deliberately lags, to debounce. `ready` must not.

        Otherwise a scan can fire at an empty counter in the few frames
        between the pack being lifted and `present` catching up.
        """
        self._feed(self._empty(), 3)
        self._feed(self._pack(), 5)
        after = self.det.update(self._empty())
        self.assertTrue(after["present"])      # still debouncing
        self.assertFalse(after["ready"])       # but not scannable

    def test_single_stray_frame_does_not_flip_presence(self):
        self._feed(self._empty(), 6)
        state = self._feed(self._pack(), 1)
        self.assertFalse(state["present"])

    def test_area_ratio_is_scale_invariant(self):
        """The detector downscales for speed; the presence decision must not
        change because of it."""
        from edgemedicheck.capture import find_package_region

        full = self._pack()
        small = self.cv2.resize(full, (480, 270), interpolation=self.cv2.INTER_AREA)
        _, _, ratio_full = find_package_region(full)
        _, _, ratio_small = find_package_region(small)
        self.assertAlmostEqual(ratio_full, ratio_small, places=2)


class TestLiveStateEndpoint(unittest.TestCase):
    """The live screen polls this several times a second, so it must stay
    cheap and must never run the pipeline."""

    def _client(self, folder):
        import app as flask_app

        application = flask_app.create_app(
            db_path=self.db_path, folder=folder, backend="folder"
        )
        application.config["TESTING"] = True
        return application.test_client()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "t.sqlite3")
        db.init_db(self.db_path)

        import cv2

        self.empty_dir = Path(self.tmp.name) / "empty"
        self.pack_dir = Path(self.tmp.name) / "pack"
        for d in (self.empty_dir, self.pack_dir):
            d.mkdir()
        blank = np.full((720, 1280, 3), 30, np.uint8)
        pack = blank.copy()
        cv2.rectangle(pack, (300, 150), (980, 560), (240, 240, 235), -1)
        for i in range(4):
            cv2.imwrite(str(self.empty_dir / f"{i}.jpg"), blank)
            cv2.imwrite(str(self.pack_dir / f"{i}.jpg"), pack)

    def tearDown(self):
        self.tmp.cleanup()

    def test_state_reports_the_documented_keys(self):
        client = self._client(str(self.pack_dir))
        body = client.get("/api/live/state").get_json()
        for key in ("present", "steady", "ready", "area_ratio", "camera_ok"):
            self.assertIn(key, body)

    def test_scan_is_refused_at_an_empty_counter(self):
        """Without this gate an empty frame is put through OCR and comes back
        amber, which on the live screen reads as a warning about a pack that
        is not there."""
        client = self._client(str(self.empty_dir))
        client.get("/api/live/state")          # let the detector see a frame
        body = client.post("/api/scan/frame").get_json()
        self.assertTrue(body.get("idle"))
        self.assertEqual(body.get("reason"), "no_package_in_frame")

    def test_force_overrides_the_gate(self):
        """The operator can always demand a reading of whatever is in view."""
        client = self._client(str(self.empty_dir))
        body = client.post("/api/scan/frame?force=1").get_json()
        self.assertNotIn("idle", body)
        self.assertIn("verdict", body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
