"""
Unit tests for BridgeTwin bridge inspection and resource scheduling modules.

Covers:
  - AASHTO database initialisation and lookups   (bridge_twin)
  - Defect detection prompt / response parsing   (bridge_twin)
  - InspectionRecord management                  (bridge_twin)
  - InspectionSession persistence and sync       (bridge_twin)
  - Gemma 4 vision message builder               (bridge_twin)
  - SchedulerEnv step / reward logic             (scheduler)
  - TabularQScheduler training and output        (scheduler)
  - Schedule overlap detection                   (scheduler)
  - explain() human-readable output              (scheduler)

Run:
    pytest tests/test_bridge_twin.py -v -m bridge_unit

All tests run offline with no network access and no Gemma 4 model required.
"""

import json
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Ensure the scripts/ directory is importable
# ---------------------------------------------------------------------------

_SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)


# ===================================================================
# SECTION 1: AASHTO database
# ===================================================================

@pytest.mark.bridge_unit
class TestAashtoDatabase:
    """Tests for the local AASHTO SQLite catalogue."""

    def test_init_creates_tables(self, tmp_path):
        from bridge_twin import _init_aashto_db

        db = tmp_path / "test.db"
        conn = _init_aashto_db(db)
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "aashto_conditions" in tables
        assert "inspection_records" in tables
        conn.close()

    def test_seed_data_present(self, tmp_path):
        from bridge_twin import _init_aashto_db

        db = tmp_path / "test.db"
        conn = _init_aashto_db(db)
        count = conn.execute("SELECT COUNT(*) FROM aashto_conditions").fetchone()[0]
        conn.close()
        assert count > 0, "AASHTO condition table must be seeded"

    def test_lookup_condition_found(self, tmp_path):
        from bridge_twin import lookup_condition

        result = lookup_condition("RC-CON", 3, db_path=tmp_path / "test.db")
        assert result is not None
        assert result["element_code"] == "RC-CON"
        assert result["condition_state"] == 3
        assert result["severity_label"] == "Poor"

    def test_lookup_condition_not_found(self, tmp_path):
        from bridge_twin import lookup_condition

        result = lookup_condition("XX-ZZZ", 9, db_path=tmp_path / "test.db")
        assert result is None

    def test_list_elements_returns_distinct(self, tmp_path):
        from bridge_twin import list_elements

        elements = list_elements(db_path=tmp_path / "test.db")
        assert len(elements) > 0
        codes = [e["element_code"] for e in elements]
        assert "RC-CON" in codes
        assert "ST-BM" in codes
        # Uniqueness
        assert len(codes) == len(set(codes))

    def test_all_four_condition_states_seeded(self, tmp_path):
        from bridge_twin import _init_aashto_db

        db = tmp_path / "test.db"
        conn = _init_aashto_db(db)
        for cs in (1, 2, 3, 4):
            count = conn.execute(
                "SELECT COUNT(*) FROM aashto_conditions WHERE condition_state=?", (cs,)
            ).fetchone()[0]
            assert count > 0, f"No entries for condition state {cs}"
        conn.close()


# ===================================================================
# SECTION 2: Detection helpers
# ===================================================================

@pytest.mark.bridge_unit
class TestDetectionHelpers:
    """Tests for prompt building and response parsing."""

    def test_build_detection_prompt_contains_schema(self):
        from bridge_twin import build_detection_prompt

        prompt = build_detection_prompt()
        assert "condition_state" in prompt
        assert "element_type" in prompt
        assert "CS 3" in prompt

    def test_build_detection_prompt_with_context(self):
        from bridge_twin import build_detection_prompt

        prompt = build_detection_prompt("Concrete pier near midspan")
        assert "Concrete pier near midspan" in prompt

    def test_parse_detection_response_clean_json(self):
        from bridge_twin import parse_detection_response

        raw = json.dumps({
            "element_type": "column",
            "condition_state": 3,
            "severity": "Poor",
            "defects": ["spalling > 1 in.", "exposed rebar"],
            "notes": "Found on south face",
        })
        result = parse_detection_response(raw)
        assert result is not None
        assert result["condition_state"] == 3
        assert result["severity"] == "Poor"

    def test_parse_detection_response_markdown_fence(self):
        from bridge_twin import parse_detection_response

        wrapped = (
            "```json\n"
            '{"element_type": "deck", "condition_state": 2, "severity": "Fair", '
            '"defects": ["map cracking"], "notes": ""}\n'
            "```"
        )
        result = parse_detection_response(wrapped)
        assert result is not None
        assert result["condition_state"] == 2

    def test_parse_detection_response_invalid_returns_none(self):
        from bridge_twin import parse_detection_response

        result = parse_detection_response("This is just plain text with no JSON.")
        assert result is None

    def test_infer_condition_state_from_text_spall(self):
        from bridge_twin import infer_condition_state_from_text

        code, cs = infer_condition_state_from_text(
            "I see a large spall on the concrete pylon with exposed rebar."
        )
        assert cs == 3
        # Should map to a concrete element
        assert code in ("RC-CON", "RC-DEC", "RC-BM")

    def test_infer_condition_state_good(self):
        from bridge_twin import infer_condition_state_from_text

        _, cs = infer_condition_state_from_text("The paint coating looks good and intact.")
        assert cs == 1

    def test_infer_condition_state_severe(self):
        from bridge_twin import infer_condition_state_from_text

        _, cs = infer_condition_state_from_text("Severe section loss on the steel girder.")
        assert cs == 4

    def test_infer_condition_state_default_element(self):
        from bridge_twin import infer_condition_state_from_text

        code, _ = infer_condition_state_from_text("Something unrecognisable.")
        assert code == "RC-CON"  # default


# ===================================================================
# SECTION 3: InspectionRecord
# ===================================================================

@pytest.mark.bridge_unit
class TestInspectionRecord:
    """Tests for the InspectionRecord dataclass."""

    def test_record_has_unique_id(self):
        from bridge_twin import InspectionRecord

        r1 = InspectionRecord()
        r2 = InspectionRecord()
        assert r1.id != r2.id

    def test_set_image_bytes_encodes_base64(self):
        from bridge_twin import InspectionRecord

        rec = InspectionRecord()
        rec.set_image_bytes(b"\xff\xd8\xff\xe0")
        assert rec.image_b64 is not None
        import base64
        decoded = base64.b64decode(rec.image_b64)
        assert decoded == b"\xff\xd8\xff\xe0"

    def test_to_dict_round_trip(self):
        from bridge_twin import InspectionRecord

        rec = InspectionRecord(bridge_id="IL-0047", inspector_id="eng-001")
        d = rec.to_dict()
        rec2 = InspectionRecord.from_dict(d)
        assert rec2.bridge_id == rec.bridge_id
        assert rec2.id == rec.id

    def test_timestamp_is_utc_iso(self):
        from bridge_twin import InspectionRecord
        from datetime import datetime, timezone

        rec = InspectionRecord()
        # Should parse as a valid ISO timestamp
        dt = datetime.fromisoformat(rec.timestamp_utc)
        assert dt.tzinfo is not None


# ===================================================================
# SECTION 4: InspectionSession
# ===================================================================

@pytest.mark.bridge_unit
class TestInspectionSession:
    """Tests for InspectionSession — new_record, save, sync, export."""

    def test_new_record_tracks_in_session(self, tmp_path):
        from bridge_twin import InspectionSession

        with InspectionSession("IL-0047", "eng-001", db_path=tmp_path / "t.db") as sess:
            r = sess.new_record(lat=40.5, lon=-88.9)
            assert r.bridge_id == "IL-0047"
            assert r.inspector_id == "eng-001"
            assert r.latitude == pytest.approx(40.5)

    def test_apply_detection_from_json(self, tmp_path):
        from bridge_twin import InspectionSession

        with InspectionSession("IL-0047", "eng-001", db_path=tmp_path / "t.db") as sess:
            rec = sess.new_record()
            gemma_json = json.dumps({
                "element_type": "column",
                "condition_state": 3,
                "severity": "Poor",
                "defects": ["spalling"],
                "notes": "South face",
            })
            sess.apply_detection(rec, gemma_json)
            assert rec.condition_state == 3
            assert rec.severity_label == "Poor"
            assert "spalling" in rec.notes

    def test_apply_detection_fallback(self, tmp_path):
        from bridge_twin import InspectionSession

        with InspectionSession("IL-0047", "eng-001", db_path=tmp_path / "t.db") as sess:
            rec = sess.new_record()
            sess.apply_detection(rec, "There is a severe crack and section loss on the steel beam.")
            assert rec.condition_state == 4  # "severe" keyword
            assert rec.element_code is not None

    def test_save_and_load_unsynced(self, tmp_path):
        from bridge_twin import InspectionSession

        db = tmp_path / "t.db"
        with InspectionSession("IL-0047", "eng-001", db_path=db) as sess:
            r = sess.new_record(lat=40.5, lon=-88.9)
            r.condition_state = 2
            r.element_code = "RC-DEC"
            r.severity_label = "Fair"
            sess.save()

        # Re-open and load
        with InspectionSession("IL-0047", "eng-001", db_path=db) as sess2:
            unsynced = sess2.load_unsynced()
            assert len(unsynced) == 1
            assert unsynced[0].condition_state == 2

    def test_mark_synced(self, tmp_path):
        from bridge_twin import InspectionSession

        db = tmp_path / "t.db"
        with InspectionSession("IL-0047", "eng-001", db_path=db) as sess:
            r = sess.new_record()
            r.condition_state = 1
            r.element_code = "RC-CON"
            r.severity_label = "Good"
            sess.save()
            sess.mark_synced([r.id])
            unsynced = sess.load_unsynced()
            assert len(unsynced) == 0

    def test_export_sync_payload_structure(self, tmp_path):
        from bridge_twin import InspectionSession

        db = tmp_path / "t.db"
        with InspectionSession("IL-0047", "eng-001", db_path=db) as sess:
            r = sess.new_record()
            r.condition_state = 2
            r.element_code = "RC-DEC"
            r.severity_label = "Fair"
            sess.save()
            payload = sess.export_sync_payload()

        assert payload["bridge_id"] == "IL-0047"
        assert payload["inspector_id"] == "eng-001"
        assert payload["record_count"] == 1
        assert isinstance(payload["records"], list)
        assert "exported_at" in payload

    def test_multiple_bridges_isolated(self, tmp_path):
        from bridge_twin import InspectionSession

        db = tmp_path / "t.db"
        with InspectionSession("IL-0047", "eng-001", db_path=db) as s1:
            r = s1.new_record()
            r.condition_state = 1
            r.element_code = "RC-CON"
            r.severity_label = "Good"
            s1.save()

        with InspectionSession("IL-0099", "eng-002", db_path=db) as s2:
            r2 = s2.new_record()
            r2.condition_state = 3
            r2.element_code = "ST-BM"
            r2.severity_label = "Poor"
            s2.save()
            unsynced = s2.load_unsynced()
            assert all(r.bridge_id == "IL-0099" for r in unsynced)


# ===================================================================
# SECTION 5: Vision message builder
# ===================================================================

@pytest.mark.bridge_unit
class TestVisionMessageBuilder:
    """Tests for build_vision_message (Gemma 4 multimodal prompt)."""

    def test_message_has_system_and_user(self):
        from bridge_twin import build_vision_message

        msgs = build_vision_message("AAAA")  # minimal b64
        roles = [m["role"] for m in msgs]
        assert "system" in roles
        assert "user" in roles

    def test_image_url_data_uri(self):
        from bridge_twin import build_vision_message
        import base64

        raw = b"\xff\xd8\xff\xe0"
        b64 = base64.b64encode(raw).decode()
        msgs = build_vision_message(b64, image_mime="image/jpeg")
        user_content = msgs[-1]["content"]
        image_parts = [p for p in user_content if p.get("type") == "image_url"]
        assert len(image_parts) == 1
        url = image_parts[0]["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,")

    def test_context_injected(self):
        from bridge_twin import build_vision_message

        msgs = build_vision_message("AAAA", context="Concrete pier, south face")
        user_content = msgs[-1]["content"]
        text_parts = [p for p in user_content if p.get("type") == "text"]
        assert any("Concrete pier" in p.get("text", "") for p in text_parts)


# ===================================================================
# SECTION 6: SchedulerEnv
# ===================================================================

@pytest.mark.bridge_unit
class TestSchedulerEnv:
    """Tests for the SchedulerEnv step / reward logic."""

    def _simple_env(self):
        from scheduler import SchedulerEnv

        return SchedulerEnv(
            n_slots=24,
            resources=["crew-A", "road-closure"],
            tasks=[
                {"id": "T1", "resource": "crew-A",      "duration": 4, "earliest": 0},
                {"id": "T2", "resource": "road-closure", "duration": 6, "earliest": 0},
            ],
        )

    def test_reset_all_unscheduled(self):
        env = self._simple_env()
        state = env.reset()
        from scheduler import SchedulerEnv
        assert all(v == SchedulerEnv.UNSCHEDULED for v in state.values())

    def test_step_schedules_task(self):
        env = self._simple_env()
        env.reset()
        new_state, reward, done = env.step(0, 2)
        assert new_state["T1"] == 2
        assert reward > 0

    def test_step_out_of_range_penalised(self):
        env = self._simple_env()
        env.reset()
        # slot 22 + duration 4 = 26 > n_slots=24 → invalid
        _, reward, _ = env.step(0, 22)
        assert reward < 0

    def test_done_when_all_scheduled(self):
        env = self._simple_env()
        env.reset()
        env.step(0, 0)
        _, _, done = env.step(1, 8)
        assert done

    def test_overlap_penalised(self):
        from scheduler import SchedulerEnv

        env = SchedulerEnv(
            n_slots=24,
            resources=["crew-A"],
            tasks=[
                {"id": "T1", "resource": "crew-A", "duration": 4, "earliest": 0},
                {"id": "T2", "resource": "crew-A", "duration": 4, "earliest": 0},
            ],
        )
        env.reset()
        env.step(0, 0)              # T1 → slots 0-3
        _, reward, _ = env.step(1, 0)  # T2 → slots 0-3 (same resource, overlap)
        assert reward < 0, "Overlapping assignment must yield negative reward"

    def test_count_overlaps(self):
        from scheduler import SchedulerEnv

        env = SchedulerEnv(
            n_slots=24,
            resources=["crew-A"],
            tasks=[
                {"id": "T1", "resource": "crew-A", "duration": 4, "earliest": 0},
                {"id": "T2", "resource": "crew-A", "duration": 4, "earliest": 0},
            ],
        )
        env.reset()
        env.step(0, 0)
        env.step(1, 0)
        assert env.count_overlaps() == 1

    def test_no_overlap_different_resources(self):
        from scheduler import SchedulerEnv

        env = SchedulerEnv(
            n_slots=24,
            resources=["crew-A", "crew-B"],
            tasks=[
                {"id": "T1", "resource": "crew-A", "duration": 4, "earliest": 0},
                {"id": "T2", "resource": "crew-B", "duration": 4, "earliest": 0},
            ],
        )
        env.reset()
        env.step(0, 0)
        env.step(1, 0)
        assert env.count_overlaps() == 0


# ===================================================================
# SECTION 7: TabularQScheduler
# ===================================================================

@pytest.mark.bridge_unit
class TestTabularQScheduler:
    """Tests for the Q-learning resource scheduler."""

    def _make_env(self):
        from scheduler import SchedulerEnv

        return SchedulerEnv(
            n_slots=48,
            resources=["crew-A", "road-closure", "equipment-crane"],
            tasks=[
                {"id": "T1", "resource": "crew-A",        "duration": 8, "earliest": 0,  "priority": 2},
                {"id": "T2", "resource": "road-closure",   "duration": 4, "earliest": 8,  "priority": 3},
                {"id": "T3", "resource": "equipment-crane","duration": 6, "earliest": 0,  "priority": 1},
            ],
        )

    def test_train_completes_without_error(self):
        from scheduler import TabularQScheduler

        env = self._make_env()
        sched = TabularQScheduler(env, seed=42)
        sched.train(episodes=50)  # quick — unit test

    def test_best_schedule_all_tasks_present(self):
        from scheduler import TabularQScheduler

        env = self._make_env()
        sched = TabularQScheduler(env, seed=42)
        sched.train(episodes=100)
        schedule = sched.best_schedule()
        for task in env.tasks():
            assert task.id in schedule, f"Task {task.id} missing from schedule"

    def test_trained_schedule_reduces_overlaps(self):
        from scheduler import TabularQScheduler, SchedulerEnv

        # Two tasks on the same resource — a decent scheduler should find
        # a non-overlapping assignment (T1: 0-4, T2: 4-8)
        env = SchedulerEnv(
            n_slots=24,
            resources=["crew-A"],
            tasks=[
                {"id": "T1", "resource": "crew-A", "duration": 4, "earliest": 0},
                {"id": "T2", "resource": "crew-A", "duration": 4, "earliest": 0},
            ],
        )
        sched = TabularQScheduler(env, seed=0)
        sched.train(episodes=300)
        schedule = sched.best_schedule()
        overlaps = sched._count_overlaps_in(schedule)
        assert overlaps == 0, f"Expected 0 overlaps after training, got {overlaps}"

    def test_explain_returns_string(self):
        from scheduler import TabularQScheduler

        env = self._make_env()
        sched = TabularQScheduler(env, seed=42)
        sched.train(episodes=50)
        text = sched.explain()
        assert isinstance(text, str)
        assert "BridgeTwin Dispatch Schedule" in text

    def test_explain_with_epoch(self):
        from scheduler import TabularQScheduler

        env = self._make_env()
        sched = TabularQScheduler(env, seed=42)
        sched.train(episodes=50)
        text = sched.explain(epoch="2026-04-14T08:00:00")
        # Should contain a date string
        assert "2026-04-14" in text

    def test_task_invalid_duration_raises(self):
        from scheduler import Task

        with pytest.raises(ValueError, match="duration"):
            Task(id="X", resource="r", duration=0)

    def test_schedule_entry_overlap_detection(self):
        from scheduler import ScheduleEntry

        a = ScheduleEntry("T1", "crew-A", 0, 4)
        b = ScheduleEntry("T2", "crew-A", 2, 6)
        c = ScheduleEntry("T3", "crew-A", 4, 8)
        d = ScheduleEntry("T4", "crew-B", 0, 4)

        assert a.overlaps(b)      # same resource, overlapping slots
        assert not a.overlaps(c)  # adjacent — no overlap
        assert not a.overlaps(d)  # different resource
