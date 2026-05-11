#!/usr/bin/env python3
"""
MindScape Robust Synthetic Seed Generator

Generates production-grade, clinically coherent seed data for:
- Organizations, Hospitals, Doctors
- Patients with realistic diagnosis distributions
- Longitudinal event histories (6-12 months)

Usage:
    python scripts/generate_seed.py --doctors 60 --patients 720 --seed 42 --output-dir data/seed

The generator produces:
- doctors_expanded.json (full 60 doctors)
- patients_base.json
- patient_longitudinal/ (one JSON per patient with timeline)
- clinic_state_initial.json (ready to merge into runtime state)
"""

import argparse
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# -----------------------------
# Clinical Distributions (Realistic)
# -----------------------------
DIAGNOSIS_DISTRIBUTION = {
    "Major Depressive Disorder": 0.18,
    "Generalized Anxiety Disorder": 0.12,
    "PTSD": 0.09,
    "Bipolar II Disorder": 0.07,
    "Social Anxiety Disorder": 0.06,
    "Panic Disorder": 0.05,
    "Adjustment Disorder": 0.08,
    "ADHD": 0.07,
    "Borderline Personality Disorder": 0.05,
    "Insomnia Disorder": 0.06,
    "Alcohol Use Disorder": 0.04,
    "Somatic Symptom Disorder": 0.04,
    "Schizophrenia": 0.03,
    "Dissociative Identity Disorder": 0.02,
    "Persistent Depressive Disorder": 0.04,
}

RISK_LEVELS = ["Low", "Medium", "High"]
AGE_BANDS = [(18, 29), (30, 44), (45, 59), (60, 75)]

LANGUAGES = ["English", "Mandarin", "Cantonese", "Hindi", "Swahili", "Kiswahili", "French", "Spanish"]

# Event templates for longitudinal generation
EVENT_TEMPLATES = {
    "intake": "Initial intake assessment completed. Presenting concern: {concern}. Risk level assigned: {risk}.",
    "consult": "Follow-up consultation. {note}.",
    "checkin": "Daily/weekly check-in via Nancy. Mood: {mood}/10, Sleep: {sleep}h. Triggers noted: {triggers}.",
    "session": "Live MindScape diagnostic session. Hypothesis: {hypothesis}. BSV: V={valence}, A={arousal}, D={dominance}.",
    "sos": "SOS escalation triggered. Severity: {severity}. Resolved by: {resolved_by}.",
    "medication": "Medication review. {med_name} {action}. Adherence: {adherence}. Side effects: {side_effects}.",
    "alert": "Clinical alert generated: {alert_type}. Action taken: {action}.",
}

# -----------------------------
# Core Generator Functions
# -----------------------------

def load_json(path: Path) -> list[dict]:
    return json.loads(path.read_text())

def save_json(data: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

def generate_doctor_id(name: str) -> str:
    return name.lower().replace("dr. ", "").replace(" ", "-").replace(".", "")

def expand_doctors(base_doctors: list[dict], hospitals: list[dict], target: int = 60) -> list[dict]:
    """Expand base doctors to target count while maintaining realism."""
    expanded = list(base_doctors)
    specialties = [
        "Anxiety & Mood Disorders", "Trauma & PTSD", "Bipolar Spectrum", "Psychosis & Early Intervention",
        "Neurodevelopmental (ADHD/ASD)", "Community & Perinatal", "Geriatric Psychiatry", "Addiction Psychiatry",
        "Eating Disorders", "Dissociative Disorders", "Computational Psychiatry", "Youth Mental Health"
    ]
    departments = [
        "Mood & Anxiety Unit", "Trauma Recovery", "Early Intervention in Psychosis", "Community Mental Health",
        "Neurodevelopmental Clinic", "Computational Psychiatry Lab", "Perinatal & Women's Mental Health"
    ]

    hospital_ids = [h["id"] for h in hospitals]
    org_map = {h["id"]: h["organization_id"] for h in hospitals}

    existing_ids = {d["id"] for d in expanded}
    counter = 1

    while len(expanded) < target:
        hosp = random.choice(hospitals)
        specialty = random.choice(specialties)
        dept = random.choice(departments)

        base_name = f"Dr. {random.choice(['Aarav', 'Priya', 'Jia', 'Min', 'Omar', 'Fatima', 'Lucas', 'Sofia', 'Kenji', 'Mei'])} {random.choice(['Patel', 'Zhang', 'Kim', 'Nguyen', 'Okonkwo', 'Al-Mansouri', 'Silva', 'Yamamoto'])}"
        doc_id = generate_doctor_id(base_name)
        while doc_id in existing_ids:
            doc_id = f"{doc_id}-{counter}"
            counter += 1

        years = random.randint(5, 22)
        panel = random.randint(65, 240)

        new_doc = {
            "id": doc_id,
            "name": base_name,
            "title": random.choice([
                "Consultant Psychiatrist", "Senior Psychiatrist", "Clinical Psychologist",
                "Consultant & Research Lead", "Associate Professor of Psychiatry"
            ]),
            "hospital_id": hosp["id"],
            "organization_id": org_map[hosp["id"]],
            "specialty": specialty,
            "department": dept,
            "panel_size": panel,
            "research_interests": random.sample([
                "Voice biomarkers", "Digital phenotyping", "Longitudinal mood modeling",
                "Trauma recovery outcomes", "Task-shifting models", "Computational psychiatry",
                "Cross-cultural interventions", "Early intervention", "Wearable integration"
            ], k=3),
            "weekly_schedule": {
                "Monday": random.choice(["Clinic", "Research", "Clinic + Research"]),
                "Tuesday": random.choice(["Full Clinic", "Supervision", "Clinic + Teaching"]),
                "Wednesday": random.choice(["Research Day", "Full Clinic", "Admin + Research"]),
                "Thursday": "Full Clinic",
                "Friday": random.choice(["Research & Admin", "Clinic", "Teaching"])
            },
            "performance": {
                "followup_completion": f"{random.randint(82, 97)}%",
                "patient_satisfaction": f"{round(random.uniform(4.2, 4.9), 1)}/5"
            },
            "inbox_unread": random.randint(2, 18),
            "years_experience": years
        }
        expanded.append(new_doc)
        existing_ids.add(doc_id)

    return expanded

def generate_patient(doctor: dict, hospitals: list[dict], idx: int) -> dict:
    """Generate a single realistic patient record."""
    hosp = next(h for h in hospitals if h["id"] == doctor["hospital_id"])
    age_band = random.choice(AGE_BANDS)
    age = random.randint(*age_band)

    diagnosis = random.choices(
        list(DIAGNOSIS_DISTRIBUTION.keys()),
        weights=list(DIAGNOSIS_DISTRIBUTION.values())
    )[0]

    risk = random.choices(RISK_LEVELS, weights=[0.35, 0.45, 0.20])[0]
    language = random.choice(LANGUAGES)

    first_name = random.choice(["Aarav", "Priya", "Jia", "Min", "Omar", "Fatima", "Lucas", "Sofia", "Kenji", "Mei", "Nadia", "Leon", "Mei", "Aarav"])
    last_initial = random.choice(["S", "T", "L", "P", "K", "R", "M", "W", "C", "H"])

    patient_id = f"{first_name.lower()}-{last_initial.lower()}-{idx:04d}"

    return {
        "id": patient_id,
        "name": f"{first_name} {last_initial}.",
        "age": age,
        "language": language,
        "doctor_id": doctor["id"],
        "hospital_id": doctor["hospital_id"],
        "organization_id": doctor["organization_id"],
        "status": random.choice(["Stabilizing", "In treatment", "Needs review", "In remission", "Watch closely"]),
        "risk": risk,
        "diagnosis": diagnosis,
        "comorbidities": random.sample([
            "GAD", "MDD", "Insomnia", "PTSD", "Alcohol Use", "ADHD", "Somatic Symptoms"
        ], k=random.randint(0, 2)),
        "next_appointment": (datetime.now() + timedelta(days=random.randint(3, 45))).strftime("%b %d, %Y"),
        "care_plan": f"Standard {diagnosis.split()[0]} pathway with {random.choice(['CBT', 'medication review', 'trauma-focused therapy', 'digital check-ins'])}.",
        "last_update": (datetime.now() - timedelta(days=random.randint(1, 21))).strftime("%b %d, %Y"),
        "emergency_contact": f"{random.choice(['Spouse', 'Parent', 'Sibling', 'Partner'])} - {random.randint(1000000000, 9999999999)}",
        "referral_source": random.choice(["GP", "Self-referral", "Emergency Department", "School", "Employer EAP"]),
    }

def generate_longitudinal_events(patient: dict, months: int = 9) -> list[dict]:
    """Generate 18-36 clinically coherent events over N months."""
    events = []
    start_date = datetime.now() - timedelta(days=months * 30)
    current = start_date

    # Intake
    events.append({
        "date": current.strftime("%Y-%m-%d"),
        "type": "intake",
        "title": "Initial Intake Assessment",
        "summary": EVENT_TEMPLATES["intake"].format(
            concern=patient["diagnosis"], risk=patient["risk"]
        ),
        "author": "intake_team",
        "source": "manual"
    })
    current += timedelta(days=random.randint(7, 18))

    # 2-4 follow-up consults
    for i in range(random.randint(2, 4)):
        events.append({
            "date": current.strftime("%Y-%m-%d"),
            "type": "consult",
            "title": f"Follow-up Consultation #{i+1}",
            "summary": EVENT_TEMPLATES["consult"].format(
                note=random.choice([
                    "Good response to CBT", "Medication adjustment discussed",
                    "Sleep hygiene review", "Anxiety management strategies reviewed"
                ])
            ),
            "author": patient["doctor_id"],
            "source": "manual"
        })
        current += timedelta(days=random.randint(14, 35))

    # Check-ins (higher frequency for higher risk)
    checkin_count = random.randint(12, 28) if patient["risk"] == "High" else random.randint(6, 15)
    for _ in range(checkin_count):
        current += timedelta(days=random.randint(2, 7))
        if current > datetime.now():
            break
        mood = random.randint(3, 8)
        sleep = round(random.uniform(4.5, 8.5), 1)
        triggers = random.choice(["work stress", "relationship", "trauma reminder", "sleep debt", "none reported"])
        events.append({
            "date": current.strftime("%Y-%m-%d"),
            "type": "checkin",
            "title": "Nancy Check-in",
            "summary": EVENT_TEMPLATES["checkin"].format(mood=mood, sleep=sleep, triggers=triggers),
            "author": "nancy_agent",
            "source": "nancy"
        })

    # 1-3 MindScape sessions
    for i in range(random.randint(1, 3)):
        current += timedelta(days=random.randint(20, 50))
        if current > datetime.now():
            break
        events.append({
            "date": current.strftime("%Y-%m-%d"),
            "type": "session",
            "title": f"Live Diagnostic Session #{i+1}",
            "summary": EVENT_TEMPLATES["session"].format(
                hypothesis=patient["diagnosis"],
                valence=round(random.uniform(-0.6, 0.4), 2),
                arousal=round(random.uniform(0.3, 0.9), 2),
                dominance=round(random.uniform(0.2, 0.8), 2)
            ),
            "author": patient["doctor_id"],
            "source": "mindscape"
        })

    # Occasional SOS / alerts for higher risk patients
    if patient["risk"] in ["Medium", "High"] and random.random() < 0.45:
        current += timedelta(days=random.randint(10, 40))
        if current <= datetime.now():
            events.append({
                "date": current.strftime("%Y-%m-%d"),
                "type": "sos",
                "title": "SOS Escalation",
                "summary": EVENT_TEMPLATES["sos"].format(
                    severity=patient["risk"],
                    resolved_by=random.choice(["on_call_team", "crisis_line", "family_support", "emergency_services"])
                ),
                "author": "system",
                "source": "nancy"
            })

    # Sort chronologically
    events.sort(key=lambda e: e["date"])
    return events[:36]  # Cap at 36 events

# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="MindScape Robust Seed Generator")
    parser.add_argument("--doctors", type=int, default=60, help="Target number of doctors")
    parser.add_argument("--patients", type=int, default=720, help="Target number of patients")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output-dir", type=Path, default=Path("data/seed"), help="Output directory")
    args = parser.parse_args()

    random.seed(args.seed)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading base seed files...")
    base_doctors = load_json(Path("data/seed/doctors.json"))
    hospitals = load_json(Path("data/seed/hospitals.json"))
    organizations = load_json(Path("data/seed/organizations.json"))

    print(f"Expanding doctors to {args.doctors}...")
    doctors = expand_doctors(base_doctors, hospitals, target=args.doctors)
    save_json(doctors, output_dir / "doctors_expanded.json")

    print(f"Generating {args.patients} patients with longitudinal timelines...")
    patients = []
    longitudinal_dir = output_dir / "patient_longitudinal"
    longitudinal_dir.mkdir(exist_ok=True)

    for i in range(args.patients):
        doctor = random.choice(doctors)
        patient = generate_patient(doctor, hospitals, i)
        patients.append(patient)

        # Generate timeline
        events = generate_longitudinal_events(patient, months=random.randint(6, 12))
        save_json({"patient_id": patient["id"], "events": events},
                  longitudinal_dir / f"{patient['id']}.json")

        if (i + 1) % 100 == 0:
            print(f"  Generated {i+1} patients...")

    save_json(patients, output_dir / "patients_base.json")

    # Generate initial clinic_state snapshot
    print("Building initial clinic_state snapshot...")
    clinic_state = {
        "patients_by_doctor": {},
        "version": "1.0-mvp-seed",
        "generated_at": datetime.now().isoformat()
    }
    for p in patients:
        clinic_state["patients_by_doctor"].setdefault(p["doctor_id"], []).append(p)

    save_json(clinic_state, output_dir / "clinic_state_initial.json")

    print("\n=== Seed Generation Complete ===")
    print(f"Organizations: {len(organizations)}")
    print(f"Hospitals: {len(hospitals)}")
    print(f"Doctors: {len(doctors)}")
    print(f"Patients: {len(patients)}")
    print(f"Longitudinal timelines: {len(list(longitudinal_dir.glob('*.json')))}")
    print(f"Output written to: {output_dir.resolve()}")

if __name__ == "__main__":
    main()