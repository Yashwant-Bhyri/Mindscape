from clinic_state import get_runtime_patients

import json
from pathlib import Path
from copy import deepcopy

# Robust seed loading (prefers expanded data/seed/ files)
_SEED_DIR = Path(__file__).resolve().parent / "data" / "seed"

def _load_seed(name: str) -> list[dict]:
    path = _SEED_DIR / name
    if path.exists():
        return json.loads(path.read_text())
    return []

# Load expanded organizations if available, else fall back to original
ORGANIZATIONS = _load_seed("organizations.json") or [{
    "id": "aurora-mind-institute",
    "name": "Aurora Mind Institute",
    "region": "Shenzhen + Singapore + Bengaluru",
    "focus": "Precision psychiatry, trauma recovery, multilingual behavioral intelligence.",
    "signature_program": "Longitudinal resilience tracking for high-risk urban communities.",
    "stats": {"patients": "14.2k annual journeys", "research": "82 studies", "response": "96% SLA"},
    "researchers": ["amira-khan", "li-wei", "elena-rossi"]
}]

DEFAULT_DOCTOR_ID = "amira-khan"

DOCTORS = {
    "amira-khan": {
        "id": "amira-khan",
        "name": "Dr. Amira Khan",
        "title": "Consultant Psychiatrist and Director of Adaptive Care Systems",
        "organization_id": "aurora-mind-institute",
        "specialty": "Trauma, dissociation, complex affective disorders",
        "tagline": "Building precision mental-health systems that feel as human as the clinicians who use them.",
        "mission": "Dr. Khan leads a care model that combines live multimodal diagnosis, longitudinal patient tracking, and rapid research translation for high-complexity psychiatric care.",
        "location": "Singapore",
        "years": "14 years in acute and community psychiatry",
        "hospital_name": "深圳市康宁医院",
        "hospital_district": "罗湖区",
        "department_name": "Trauma, Dissociation and Complex Care Unit",
        "stats": {
            "panel": "186 active patients",
            "watchlist": "12 high-risk care escalations",
            "research": "9 live research protocols",
            "followup": "93% follow-up completion",
        },
        "focus_areas": [
            "Complex PTSD and dissociative states",
            "High-frequency relapse detection",
            "Voice-first diagnostic support in multilingual settings",
        ],
        "patients": [
            {
                "id": "nadia-s",
                "name": "Nadia S.",
                "age": 29,
                "status": "Stabilizing",
                "risk": "Medium",
                "next_appointment": "May 12, 2026 at 09:30",
                "diagnosis": "PTSD with panic symptoms",
                "history": "Three-month recovery arc with improving sleep regularity and reduced avoidance behaviors.",
                "last_update": "Trauma triggers reduced after journaling adherence increased to 5 days per week.",
                "care_plan": "Trauma-focused CBT, grounding practice, sleep stabilization, and weekly check-in review.",
                "health_history": [
                    "Emergency presentation after recurrent flashback episodes 4 months ago.",
                    "Sleep fragmentation and hypervigilance improved after routine stabilization plan.",
                    "No recent self-harm intent disclosed; panic spikes still linked to sensory reminders.",
                ],
                "diagnosis_reports": [
                    {"title": "PTSD follow-up synthesis", "date": "May 8, 2026", "summary": "Reduced avoidance, fewer nocturnal panic episodes, and stronger grounding response."},
                    {"title": "Functional recovery review", "date": "April 28, 2026", "summary": "Work re-entry tolerated in limited schedule with moderate anticipatory anxiety."},
                ],
                "personal_logs": [
                    {"date": "May 9", "entry": "Used grounding audio during a bus-trigger episode and recovered faster than last month."},
                    {"date": "May 7", "entry": "Slept 6.5 hours and completed trauma journal without dissociative drift."},
                ],
                "care_team_notes": [
                    {"source": "Therapist", "note": "Patient is more able to narrate triggering events without immediate shutdown."},
                    {"source": "Care coordinator", "note": "Confirmed strong attendance and active family support."},
                ],
                "goals": [
                    "Sustain sleep regularity above 6 hours nightly.",
                    "Reduce panic-trigger recovery time during public-transit exposure.",
                    "Build confidence for gradual work-capacity expansion.",
                ],
                "alerts": [
                    "Monitor sudden sensory-trigger spikes before workplace transitions.",
                    "Flag any renewed dissociation during trauma processing sessions.",
                ],
            },
            {
                "id": "leon-t",
                "name": "Leon T.",
                "age": 35,
                "status": "Needs review",
                "risk": "High",
                "next_appointment": "May 10, 2026 at 16:00",
                "diagnosis": "Bipolar II disorder",
                "history": "Energy volatility and sleep compression observed across the last two check-ins.",
                "last_update": "Medication compliance dip flagged by caregiver note and mood-entry drift.",
                "care_plan": "Urgent medication review, sleep restoration, caregiver coordination, and relapse-risk tracking.",
                "health_history": [
                    "Documented bipolar II diagnosis with prior hypomanic acceleration after sleep compression.",
                    "Recent reduction in medication adherence coincided with increased goal-directed activity.",
                    "Family reports renewed irritability and narrowing sleep window over the last week.",
                ],
                "diagnosis_reports": [
                    {"title": "Mood instability escalation note", "date": "May 9, 2026", "summary": "Sleep compression, adherence concerns, and elevated activation suggest closer relapse monitoring."},
                    {"title": "Quarterly medication response review", "date": "April 11, 2026", "summary": "Functioning improved while adherence remained stable and sleep was protected."},
                ],
                "personal_logs": [
                    {"date": "May 9", "entry": "Slept 3 hours, felt extremely productive, skipped breakfast and morning meds."},
                    {"date": "May 8", "entry": "Noticed racing thoughts late at night but felt too energized to stop working."},
                ],
                "care_team_notes": [
                    {"source": "Caregiver", "note": "Speech is faster and patience lower than baseline."},
                    {"source": "Pharmacist", "note": "Prescription refill delayed by two days this cycle."},
                ],
                "goals": [
                    "Restore sleep consistency before hypomanic escalation strengthens.",
                    "Reinstate medication adherence with caregiver visibility.",
                    "Reduce stimulation load during evening hours.",
                ],
                "alerts": [
                    "High relapse risk if sleep remains under 4 hours.",
                    "Escalate urgently if impulsive spending or pressured speech accelerates.",
                ],
            },
            {
                "id": "mei-l",
                "name": "Mei L.",
                "age": 41,
                "status": "In remission",
                "risk": "Low",
                "next_appointment": "May 18, 2026 at 11:00",
                "diagnosis": "Somatic symptom disorder",
                "history": "Somatic distress episodes now intermittent after CBT reinforcement and family support.",
                "last_update": "Appointment can be converted to telehealth if symptom load remains low.",
                "care_plan": "Maintenance CBT, symptom journaling, and lower-intensity telehealth review.",
                "health_history": [
                    "Prior cycle of repeated urgent consultations for unexplained pain episodes.",
                    "Functional distress reduced after reframing symptom-attention loops in therapy.",
                    "Family support now consistently reinforces coping routines instead of emergency-seeking behaviors.",
                ],
                "diagnosis_reports": [
                    {"title": "Maintenance remission review", "date": "May 6, 2026", "summary": "Stable functioning with intermittent somatic spikes and no recent unnecessary emergency use."},
                ],
                "personal_logs": [
                    {"date": "May 8", "entry": "Noticed chest tightness after stress but used breathing exercises before seeking reassurance."},
                ],
                "care_team_notes": [
                    {"source": "Therapist", "note": "Patient is catching catastrophic interpretation patterns earlier."},
                ],
                "goals": [
                    "Sustain remission through lower-intensity follow-up.",
                    "Keep reassurance-seeking behavior below prior baseline.",
                ],
                "alerts": [
                    "Watch for renewed health-anxiety spirals during family conflict or illness scares.",
                ],
            },
        ],
        "appointments": [
            {"time": "09:30", "patient": "Nadia S.", "type": "Trauma follow-up", "mode": "In clinic"},
            {"time": "13:00", "patient": "Case conference", "type": "Interdisciplinary review", "mode": "Hybrid"},
            {"time": "16:00", "patient": "Leon T.", "type": "Mood instability review", "mode": "Telehealth"},
        ],
        "reports": [
            {"title": "Weekly relapse risk digest", "status": "Ready", "updated": "2 hours ago"},
            {"title": "Voice-derived dissociation markers", "status": "In review", "updated": "Yesterday"},
            {"title": "Community intervention outcomes", "status": "Published internally", "updated": "3 days ago"},
        ],
        "chat_threads": [
            {"patient": "Leon T.", "channel": "Care team", "summary": "Family requested earlier medication review after sleep collapse pattern returned.", "time": "11 min ago"},
            {"patient": "Nadia S.", "channel": "Patient check-in", "summary": "Patient shared improved grounding response after flashback episode yesterday.", "time": "42 min ago"},
            {"patient": "Mei L.", "channel": "Therapy coordination", "summary": "Therapist requested symptom trend export for next joint session.", "time": "Today"},
        ],
        "weekly_brief": [
            "Dissociative symptom forums reported a rise in sensory-trigger narratives tied to abrupt routine disruption.",
            "Two fresh meta-analyses suggest stronger short-interval follow-up improves retention after acute panic episodes.",
            "Clinician agents flagged new multilingual speech-biomarker papers relevant to trauma-state misclassification.",
        ],
        "performance": [
            {"label": "Triage response speed", "value": "18% faster than last quarter"},
            {"label": "Missed-appointment recovery", "value": "71% re-engaged within 72 hours"},
            {"label": "Research-to-practice adoption", "value": "6 new protocol changes this month"},
        ],
        "forum_posts": [
            {
                "title": "Unexpected overlap between freeze states and sleep-debt signatures",
                "community": "Trauma Signal Exchange",
                "activity": "24 replies",
                "summary": "Peers are comparing whether exhausted trauma patients are being over-clustered with low-affect depressive presentations.",
            },
            {
                "title": "Structured prompts that improved dissociation follow-up quality",
                "community": "Doctor's Corner",
                "activity": "11 replies",
                "summary": "A shared template is emerging for sessions where recall fragmentation blocks conventional history taking.",
            },
        ],
    },
    "li-wei": {
        "id": "li-wei",
        "name": "Dr. Li Wei",
        "title": "Senior Research Psychiatrist, Computational Mood Lab",
        "organization_id": "aurora-mind-institute",
        "specialty": "Mood disorders, digital phenotyping, recurrence prediction",
        "tagline": "Turning weak early signals into confident intervention windows for mood instability.",
        "mission": "Dr. Wei combines longitudinal patient modeling with clinician workflows that catch recurrence before daily life collapses.",
        "location": "Singapore",
        "years": "11 years in affective disorder research",
        "hospital_name": "香港大学深圳医院",
        "hospital_district": "福田区",
        "department_name": "Mood Disorders and Digital Phenotyping Lab",
        "stats": {
            "panel": "141 active patients",
            "watchlist": "8 active escalations",
            "research": "12 predictive modeling projects",
            "followup": "91% follow-up completion",
        },
        "focus_areas": [
            "Bipolar spectrum recurrence forecasting",
            "Sleep disruption as an early-warning feature",
            "Translating passive signals into clinician-safe summaries",
        ],
        "patients": [
            {
                "id": "aarav-p",
                "name": "Aarav P.",
                "age": 27,
                "status": "Watch closely",
                "risk": "Medium",
                "next_appointment": "May 11, 2026 at 10:15",
                "diagnosis": "Cyclothymic disorder",
                "history": "Mood amplitude has widened across two weeks despite preserved functioning.",
                "last_update": "Self-reported sleep debt reached 4 nights in a row.",
                "care_plan": "Track routine disruption, protect sleep, and intervene early if mood amplitude widens further.",
                "health_history": [
                    "Cyclothymic mood variability historically intensified during travel or work-rhythm changes.",
                    "Current functioning preserved, but sleep debt trend is concerning for recurrence acceleration.",
                ],
                "diagnosis_reports": [
                    {"title": "Rhythm instability review", "date": "May 9, 2026", "summary": "Routine disruption is the clearest early warning feature this week."},
                ],
                "personal_logs": [
                    {"date": "May 9", "entry": "Energy up late at night again after two travel-heavy days."},
                ],
                "care_team_notes": [
                    {"source": "Coach", "note": "Patient remains highly functional but is minimizing mounting sleep loss."},
                ],
                "goals": [
                    "Interrupt sleep debt accumulation before mood instability escalates.",
                ],
                "alerts": [
                    "Watch for transition from functional variability to clinically relevant acceleration.",
                ],
            },
            {
                "id": "june-k",
                "name": "June K.",
                "age": 33,
                "status": "Stable",
                "risk": "Low",
                "next_appointment": "May 15, 2026 at 15:00",
                "diagnosis": "Persistent depressive disorder",
                "history": "Gradual social re-engagement with steady therapy attendance.",
                "last_update": "Reflective journaling showed improved self-efficacy statements.",
                "care_plan": "Continue therapy cadence, reinforce social re-engagement, and track cognition-energy balance.",
                "health_history": [
                    "Longstanding low mood with better response once routines and self-efficacy tracking were added.",
                ],
                "diagnosis_reports": [
                    {"title": "Low-mood maintenance note", "date": "May 7, 2026", "summary": "Language in self-logs suggests modest but real gains in agency and future orientation."},
                ],
                "personal_logs": [
                    {"date": "May 7", "entry": "Reached out to two friends this week without spiraling about being a burden."},
                ],
                "care_team_notes": [
                    {"source": "Therapist", "note": "Affect remains constricted but engagement is noticeably stronger."},
                ],
                "goals": [
                    "Maintain slow upward trajectory without overloading the schedule.",
                ],
                "alerts": [
                    "Monitor for quiet relapse masked by apparent compliance.",
                ],
            },
        ],
        "appointments": [
            {"time": "10:15", "patient": "Aarav P.", "type": "Recurrence review", "mode": "Telehealth"},
            {"time": "14:00", "patient": "Data roundtable", "type": "Research sync", "mode": "Virtual"},
        ],
        "reports": [
            {"title": "Sleep compression alert digest", "status": "Ready", "updated": "5 hours ago"},
            {"title": "Bipolar trajectory board review", "status": "Queued", "updated": "Today"},
        ],
        "chat_threads": [
            {"patient": "Aarav P.", "channel": "Patient check-in", "summary": "Requested coping plan for travel-related routine disruption.", "time": "28 min ago"},
        ],
        "weekly_brief": [
            "New recurrence studies are reinforcing the predictive value of sleep compression over self-rated mood alone.",
            "Multi-site clinics are shifting toward earlier micro-interventions after rhythm irregularity alerts.",
        ],
        "performance": [
            {"label": "Prediction precision", "value": "78% top-tier triage agreement"},
            {"label": "Protocol adoption", "value": "4 new clinics piloting the model"},
        ],
        "forum_posts": [
            {
                "title": "Are we underweighting travel disruption in mood forecasting?",
                "community": "Mood Systems Board",
                "activity": "9 replies",
                "summary": "Clinicians are comparing whether travel-driven routine changes deserve their own escalation class.",
            }
        ],
    },
    "elena-rossi": {
        "id": "elena-rossi",
        "name": "Dr. Elena Rossi",
        "title": "Clinical Psychologist and Recovery Design Lead",
        "organization_id": "aurora-mind-institute",
        "specialty": "Eating disorders, family systems, adolescent relapse prevention",
        "tagline": "Recovery programs should adapt as quickly as the person behind the chart.",
        "mission": "Dr. Rossi focuses on highly relational care models that link families, therapists, and psychiatry teams without losing therapeutic nuance.",
        "location": "Bengaluru",
        "years": "9 years in adolescent behavioral health",
        "hospital_name": "深圳市儿童医院",
        "hospital_district": "福田区",
        "department_name": "Adolescent Recovery and Family Systems Program",
        "stats": {
            "panel": "96 active patients",
            "watchlist": "5 elevated relapse risks",
            "research": "3 family-systems pilots",
            "followup": "95% care-plan adherence",
        },
        "focus_areas": [
            "Adolescent relapse prevention",
            "Family-mediated treatment loops",
            "Behavior change tracking beyond symptom scores",
        ],
        "patients": [],
        "appointments": [],
        "reports": [],
        "chat_threads": [],
        "weekly_brief": [
            "Caregiver-engaged interventions continue to outperform isolated follow-up for early relapse warning states.",
        ],
        "performance": [
            {"label": "Family engagement", "value": "88% weekly participation rate"},
        ],
        "forum_posts": [],
    },
    "samir-okafor": {
        "id": "samir-okafor",
        "name": "Dr. Samir Okafor",
        "title": "Director of Community Psychiatry Innovation",
        "organization_id": "north-star-neurohealth",
        "specialty": "Psychosis early detection, youth outreach, community mental-health systems",
        "tagline": "High-trust neighborhood care can move as fast as modern clinical intelligence.",
        "mission": "Dr. Okafor builds distributed care systems that keep community teams and specialists synchronized around early-risk patterns.",
        "location": "Nairobi",
        "years": "15 years in public mental-health leadership",
        "hospital_name": "深圳市第三人民医院",
        "hospital_district": "龙岗区",
        "department_name": "Community Psychiatry and Early Intervention Unit",
        "stats": {
            "panel": "208 active patients",
            "watchlist": "17 active escalations",
            "research": "5 implementation studies",
            "followup": "89% outreach completion",
        },
        "focus_areas": [
            "First-episode psychosis outreach",
            "Community-based early warning pipelines",
            "Care continuity across fragmented systems",
        ],
        "patients": [],
        "appointments": [],
        "reports": [],
        "chat_threads": [],
        "weekly_brief": [
            "Outreach teams are seeing higher engagement when pre-visit summaries are phrased as practical daily-life shifts rather than symptom labels.",
        ],
        "performance": [
            {"label": "Community re-engagement", "value": "64% after first missed contact"},
        ],
        "forum_posts": [],
    },
    "maya-patel": {
        "id": "maya-patel",
        "name": "Dr. Maya Patel",
        "title": "Neuropsychiatry Research Fellow",
        "organization_id": "north-star-neurohealth",
        "specialty": "ADHD, ASD, learning disorders, executive-function interventions",
        "tagline": "Better neurodevelopmental care starts with systems that respect real-world complexity.",
        "mission": "Dr. Patel connects assessment quality, patient history, and intervention design into clearer decision pathways for neurodivergent patients.",
        "location": "Boston",
        "years": "7 years in neurodevelopmental assessment",
        "hospital_name": "北京大学深圳医院",
        "hospital_district": "福田区",
        "department_name": "Neurodevelopment and Executive Function Clinic",
        "stats": {
            "panel": "123 active patients",
            "watchlist": "6 transition-risk cases",
            "research": "7 applied cognition projects",
            "followup": "92% care-plan completion",
        },
        "focus_areas": [
            "Executive function and school transitions",
            "Adult ADHD diagnostic refinement",
            "Structured follow-up design for neurodivergent patients",
        ],
        "patients": [],
        "appointments": [],
        "reports": [],
        "chat_threads": [],
        "weekly_brief": [
            "Teams are reporting better handoff quality when cognitive-load summaries are attached to every appointment packet.",
        ],
        "performance": [
            {"label": "Assessment turnaround", "value": "31% faster than service baseline"},
        ],
        "forum_posts": [],
    },
}


def get_organization(organization_id: str):
    for organization in ORGANIZATIONS:
        if organization["id"] == organization_id:
            return organization
    return ORGANIZATIONS[0]


def get_doctor(doctor_id: str):
    return DOCTORS.get(doctor_id, DOCTORS[DEFAULT_DOCTOR_ID])


def get_doctors_for_organization(organization_id: str):
    return [DOCTORS[doctor_id] for doctor_id in get_organization(organization_id)["researchers"]]


def get_patients_for_doctor(doctor_id: str):
    base_patients = get_doctor(doctor_id).get("patients", [])
    return get_runtime_patients(doctor_id) + base_patients


def get_patient(doctor_id: str, patient_id: str | None = None):
    patients = get_patients_for_doctor(doctor_id)
    if not patients:
        return None
    if patient_id is None:
        return patients[0]
    for patient in patients:
        if patient.get("id") == patient_id:
            return patient
    return None
