# MindScape MVP – Data Realism & Scale Expansion Plan

**Date**: 2026-05-10  
**Status**: Planning → Implementation  
**Goal**: Transform the current high-quality demo fixture into a production-grade MVP with believable scale, depth, and longitudinal realism while preserving the existing Mesop + backend_api architecture.

---

## 1. Current State Assessment (Why This Matters)

- **Organizations**: 2 (Aurora Mind Institute, North Star Neurohealth)
- **Doctors**: 5 total (3 with patients, 2 with empty panels despite claiming 96–208 patient panels)
- **Patients**: 5 total (3 under Amira, 2 under Li Wei)
- **Data Model Split**:
  - Static base data: `product_data.py` (dicts)
  - Runtime mutations: `clinic_state.py` (JSON file with patients_by_doctor, checkins, sessions, Nancy tasks, SOS, alerts, etc.)
  - Merging layer: `backend_api/facade.py`
  - Hospital directory: `shenzhen_directory.py` (real fetched data, but only used for Nancy/SOS, not as core backbone)

The app UI (workspace, patient profiles, timelines, alerts, Nancy handoffs) already expects rich data. Without expansion, it remains a convincing demo rather than a credible clinical OS.

---

## 2. Exact MVP Targets (Production-Grade but Achievable)

| Layer                  | Target Count | Rationale |
|------------------------|--------------|-----------|
| Organizations / Networks | 6            | 4–8 realistic for multi-region mental health platform |
| Hospitals / Clinics    | 18           | 3 per organization on average; mix of academic, community, district |
| Doctors                | 60           | Avg 3–4 per hospital; every doctor has a seeded panel |
| Patients               | 720          | Avg 12 patients per doctor (psychiatric panels are smaller than primary care) |
| Longitudinal Events / Patient | 18–36     | 6–12 months of real chronology (intake → consults → check-ins → sessions → alerts → SOS) |
| Total Timeline Events  | ~20,000      | Believable for a 60-doctor network over 9 months |

**Diagnosis Distribution (realistic psychiatric mix)**:
- Mood disorders (MDD, Bipolar II, PDD, Adjustment): 35%
- Anxiety / Trauma (GAD, Panic, Social, PTSD, ASD): 25%
- Personality / Dissociative: 12%
- Psychotic spectrum: 8%
- Neurodevelopmental (ADHD, ASD): 8%
- Somatic / Sleep / Substance: 12%

---

## 3. Recommended Implementation Order

1. **Data Architecture Shift** (this week)
   - Move from monolithic `product_data.py` dicts to structured seed JSON files under `data/seed/`.
   - Create `scripts/generate_seed.py` (synthetic but clinically coherent generator).
   - Keep `clinic_state.py` as the mutable runtime layer (already well-designed).

2. **Hospitals + Doctors First** (foundational)
   - Leverage + extend `shenzhen_directory.py` output as real hospital backbone.
   - Create 6 organizations with 18 hospitals.
   - Seed 60 doctors with specialties, departments, realistic panel sizes, research interests, weekly schedules, performance metrics.

3. **Patient Panels + Longitudinal Depth** (biggest value)
   - Generate 720 patients with full chart depth.
   - Each patient gets 6–12 months of events: intake, consult notes, daily/weekly check-ins, Nancy directives, outreach, alerts, SOS/escalations, session uploads + clinician notes.

4. **UI Scale Hardening**
   - Add search, risk/diagnosis filters, pagination, sorting to workspace and doctor pages.
   - Lazy-load patient cards and timelines.

5. **Auth, Roles, Audit (MVP minimum)**
   - Lightweight role separation (Doctor vs Org Admin vs Nancy Agent).
   - Every clinical action carries `author_id`, `timestamp`, `source` (manual / Nancy / system).

6. **Polish & Validation**
   - Ensure retrieval corpus, MindScape engine, and Nancy still work with new volume.
   - Add seed validation tests.

---

## 4. Normalized Seed Schema (JSON Files under `data/seed/`)

### 4.1 hospitals.json
```json
[
  {
    "id": "shenzhen-mental-health-center",
    "name": "Shenzhen Mental Health Center",
    "organization_id": "aurora-mind-institute",
    "type": "Tertiary Psychiatric Hospital",
    "district": "Futian",
    "address": "...",
    "beds": 420,
    "specialties": ["Mood Disorders", "Psychosis", "Trauma", "Child & Adolescent"],
    "research_focus": ["Digital phenotyping", "Early psychosis detection"]
  }
]
```

### 4.2 doctors.json (60 records)
```json
{
  "id": "amira-khan",
  "name": "Dr. Amira Khan",
  "title": "Consultant Psychiatrist",
  "hospital_id": "shenzhen-mental-health-center",
  "organization_id": "aurora-mind-institute",
  "specialty": "Trauma, Complex PTSD, Dissociative Disorders",
  "department": "Trauma & Dissociation Unit",
  "panel_size": 186,
  "research_interests": ["Voice biomarkers in trauma", "Longitudinal resilience"],
  "weekly_schedule": {"Mon": "Clinic + Research", "Tue": "Full Clinic", ...},
  "performance": {"triage_speed": "18% faster", "followup_rate": "93%"},
  "inbox_unread": 7
}
```

### 4.3 patients_base.json (720 records – base demographics + current state)
Each patient includes:
- id, doctor_id, hospital_id
- demographics (age, gender, language, emergency_contact, referral_source)
- current_diagnosis, comorbidities (list), risk_level, care_plan, next_appointment
- status, last_update

### 4.4 patient_longitudinal.json (or per-patient timeline files)
Normalized event stream per patient:
```json
{
  "patient_id": "nadia-s",
  "events": [
    {
      "date": "2025-11-03",
      "type": "intake",
      "title": "Initial Intake",
      "summary": "...",
      "author": "intake_coordinator",
      "source": "manual"
    },
    {
      "date": "2025-11-12",
      "type": "session",
      "title": "Live Diagnostic Session",
      "mindscape_result": { "hypothesis": "PTSD with panic", "bsv": {...} },
      "clinician_note": "..."
    },
    {
      "date": "2026-01-15",
      "type": "checkin",
      "channel": "Nancy",
      "mood": 4,
      "sleep_hours": 5.5,
      "triggers": ["bus commute"],
      "alert_generated": true
    },
    {
      "date": "2026-02-20",
      "type": "sos",
      "severity": "High",
      "resolved_by": "on_call_team"
    }
  ]
}
```

### 4.5 Additional Supporting Seeds
- `diagnosis_distributions.json` (for generator)
- `event_templates.json` (clinical narrative templates)
- `nancy_directives_templates.json`

---

## 5. Seed Generator (`scripts/generate_seed.py`)

**Core Logic**:
1. Load hospital + doctor seeds.
2. For each doctor, sample N patients from realistic diagnosis distribution.
3. For each patient, generate 6–12 months of chronologically consistent events using templates + rules (e.g., after high-risk session → increased check-in frequency → possible SOS if sleep collapses).
4. Ensure internal consistency (medication adherence issues appear in both logs and alerts; trauma history appears in multiple event types).
5. Output normalized JSON files + update `clinic_state.json` initial state.

**Clinical Coherence Rules** (examples):
- Bipolar II patient: sleep compression events precede hypomanic flags.
- PTSD patient: anniversary reactions trigger SOS spikes.
- Somatic patient: reassurance-seeking check-ins decrease after CBT notes.

---

## 6. Integration Points to Update

- `product_data.py` → becomes thin wrapper that loads from `data/seed/`.
- `backend_api/facade.py` → add search/filter/pagination helpers.
- `clinic_state.py` → ensure it can initialize from large seed.
- `app.py` (Mesop) → add loading states and virtualized lists for 700+ patients.
- New: `tests/test_seed_generator.py` + `tests/test_data_consistency.py`.

---

## 7. Next Immediate Actions (I Can Execute Now)

1. Create the directory structure + base schema JSON files.
2. Implement the heading-aware chunker improvement (already done) is orthogonal but helpful.
3. Build the first version of `generate_seed.py` that produces 6 orgs / 18 hospitals / 60 doctors with empty-but-plausible patient slots.
4. Then run the generator for the first 200 patients with full longitudinal depth.
5. Wire the new seeds into facade.py so the existing UI immediately shows realistic scale.

---

**This plan directly addresses every gap you listed while respecting the existing architecture (runtime JSON state + Mesop frontend + Nancy integration).**

Ready to proceed with Phase 1 implementation? I can start by generating the seed directory and the generator script right now.