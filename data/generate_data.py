"""
Synthetic Complaint Dataset Generator
Generates multilingual complaint data with labels for:
  - Officer assignment (8 officers)
  - Priority (High / Medium / Low)
  - ETA in days
  - Text for embedding-based similarity search
"""

import random
import pandas as pd
import numpy as np

random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────
# Officer registry
# ─────────────────────────────────────────────
OFFICERS = [
    {"id": "OFF001", "name": "Rahul Sharma",  "department": "Infrastructure & Roads"},
    {"id": "OFF002", "name": "Priya Mehta",   "department": "Water & Sanitation"},
    {"id": "OFF003", "name": "Amit Verma",    "department": "Electricity & Utilities"},
    {"id": "OFF004", "name": "Sunita Patel",  "department": "Public Safety & Security"},
    {"id": "OFF005", "name": "Vijay Kumar",   "department": "Health & Environment"},
    {"id": "OFF006", "name": "Anjali Singh",  "department": "Land & Property"},
    {"id": "OFF007", "name": "Ravi Nair",     "department": "Transport & Traffic"},
    {"id": "OFF008", "name": "Meena Reddy",   "department": "Administrative Services"},
]

# ─────────────────────────────────────────────
# Complaint templates per officer category
# Each entry: (text_template, priority_distribution, eta_range_days)
# ─────────────────────────────────────────────
COMPLAINT_TEMPLATES = {
    "OFF001": {
        "priority_weights": {"High": 0.35, "Medium": 0.45, "Low": 0.20},
        "eta": {"High": (1, 3), "Medium": (5, 10), "Low": (15, 25)},
        "templates": [
            "There is a massive pothole on {street} road near {landmark} which has caused {count} accidents in the past week.",
            "The road at {street} has completely collapsed near {landmark}. Heavy vehicles cannot pass and it is dangerous.",
            "Street lights on {street} have not been working for {days} days creating a safety hazard at night.",
            "The bridge on {street} connecting {area1} and {area2} has developed cracks and is unsafe for vehicles.",
            "Severe road flooding near {landmark} on {street} makes it impossible to travel. Drainage is blocked.",
            "Construction debris has been blocking {street} road for {days} days with no clearing action taken.",
            "The footpath on {street} near {landmark} is completely broken and elderly people keep falling.",
            "Huge potholes near {landmark} have damaged many vehicles. The {street} stretch is a nightmare to drive on.",
            "Road accident happened due to broken divider on {street}. Urgent repair needed before more lives are lost.",
            "Storm water drains on {street} are choked causing severe waterlogging every time it rains near {landmark}.",
        ]
    },
    "OFF002": {
        "priority_weights": {"High": 0.40, "Medium": 0.40, "Low": 0.20},
        "eta": {"High": (1, 4), "Medium": (5, 10), "Low": (10, 20)},
        "templates": [
            "No water supply in {area1} for the past {days} days. Residents are in severe distress.",
            "Sewage water is overflowing onto {street} road near {landmark} spreading disease.",
            "The water supplied to {area1} is yellowish and smells bad. It is not safe for drinking.",
            "A major water pipeline has burst near {landmark} and water is being wasted for {days} days.",
            "Drainage system in {area1} has completely failed. Sewage water entering homes.",
            "Water supply is only available for {count} hours a day in {area1} sector. This is insufficient.",
            "Open drainage near {landmark} is releasing foul smell and causing health issues in {area1}.",
            "Manholes near {street} and {landmark} are overflowing, spreading sewage on the road.",
            "The water tanker promised to {area1} has not arrived for {days} days during this severe shortage.",
            "Contaminated water supply in {area1} has caused stomach illness in over {count} families.",
        ]
    },
    "OFF003": {
        "priority_weights": {"High": 0.45, "Medium": 0.35, "Low": 0.20},
        "eta": {"High": (1, 2), "Medium": (3, 7), "Low": (10, 20)},
        "templates": [
            "Complete power outage in {area1} for {days} days. Hospital and essential services affected.",
            "A live electric wire has fallen on {street} near {landmark}. Extremely dangerous situation.",
            "Electricity transformer in {area1} exploded last night. {count} households without power.",
            "Frequent power cuts in {area1} every day for the past {days} days. No prior notice given.",
            "Electricity meter in my building shows unusually high reading of {count} units. Suspected billing error.",
            "Electric pole on {street} near {landmark} is leaning dangerously. Risk of it falling.",
            "Streetlights in {area1} have not been working for {days} days. Theft incidents increasing.",
            "High tension wire is hanging very low near {landmark} on {street}. Children at risk.",
            "Power fluctuations are destroying home appliances in {area1} sector. Voltage stabilizer not helping.",
            "Illegal electricity connections from {area1} are overloading the main feeder causing outages.",
        ]
    },
    "OFF004": {
        "priority_weights": {"High": 0.55, "Medium": 0.30, "Low": 0.15},
        "eta": {"High": (1, 2), "Medium": (3, 7), "Low": (10, 20)},
        "templates": [
            "A theft occurred at my house in {area1} near {landmark}. Police has not responded despite {days} days.",
            "Chain snatching incidents happening daily near {street} and {landmark}. Please increase patrolling.",
            "A group of anti-social elements is regularly harassing residents near {landmark} in {area1}.",
            "My neighbour in {area1} is running an illegal business causing noise and safety issues.",
            "CCTV cameras near {landmark} on {street} have been non-functional for {days} days.",
            "Suspicious activities observed near abandoned building at {street}. No police action taken.",
            "Women are being harassed at the bus stop near {landmark} every evening. Urgent attention needed.",
            "Illegal weapon storage reportedly happening in {area1} area. Locals are scared to report.",
            "Domestic violence case going unreported in {area1}. Victim needs immediate intervention.",
            "Drug peddling openly happening near school on {street} in {area1} area for past {days} days.",
        ]
    },
    "OFF005": {
        "priority_weights": {"High": 0.30, "Medium": 0.45, "Low": 0.25},
        "eta": {"High": (2, 5), "Medium": (7, 14), "Low": (15, 30)},
        "templates": [
            "Garbage has not been collected in {area1} for {days} days. Massive garbage dump near {landmark}.",
            "A factory near {street} is releasing toxic fumes causing breathing problems in {area1}.",
            "Stray dog menace in {area1} near {landmark}. {count} people bitten in the past month.",
            "Illegal garbage dumping near {landmark} on {street} is creating a major health hazard.",
            "Mosquito breeding due to stagnant water near {area1} has caused dengue cases to spike.",
            "Noise pollution from construction site near {landmark} violates permissible limits 24/7.",
            "Chemical waste dumped in the river near {area1} is killing fish and contaminating water.",
            "Open burning of garbage near {street} and {landmark} is causing severe air pollution.",
            "Dead animals are not being cleared near {landmark} for {days} days causing disease risk.",
            "A slaughterhouse near {area1} is operating illegally and discharging waste into open drains.",
        ]
    },
    "OFF006": {
        "priority_weights": {"High": 0.20, "Medium": 0.45, "Low": 0.35},
        "eta": {"High": (5, 10), "Medium": (14, 30), "Low": (30, 60)},
        "templates": [
            "Encroachment on government land near {landmark} in {area1}. Illegal structure being built.",
            "Property boundary dispute with neighbour in {area1}. Official measurement requested.",
            "Illegal construction on residential plot near {street} without proper permits.",
            "My land documents for plot in {area1} near {landmark} show incorrect ownership. Correction needed.",
            "Builder in {area1} is constructing {count} extra floors beyond approved plan.",
            "Heritage building near {street} is being demolished illegally. Urgent preservation action needed.",
            "Forceful eviction attempt by landlord in {area1} without proper legal notice.",
            "Commercial shop running illegally in residential zone near {landmark} in {area1}.",
            "Plot number {count} in {area1} layout has not received proper title deed for {days} days.",
            "Trespassing on my agricultural land near {area1}. Boundary wall has been broken.",
        ]
    },
    "OFF007": {
        "priority_weights": {"High": 0.25, "Medium": 0.50, "Low": 0.25},
        "eta": {"High": (2, 5), "Medium": (7, 15), "Low": (15, 30)},
        "templates": [
            "Bus route {count} from {area1} to {area2} has been suspended for {days} days without notice.",
            "Auto-rickshaws at {landmark} are charging exorbitant fares and refusing to use meters.",
            "Signal at {street} and {landmark} junction is not working causing massive traffic jams daily.",
            "Parking mafia operating near {landmark} charging illegal amounts. Officials ignoring complaints.",
            "Metro station at {area1} is overcrowded. Platform gates closing on commuters during rush hour.",
            "Goods vehicles using residential {street} during prohibited hours causing noise and damage.",
            "No bus service available in {area1} locality since {days} days. Residents walking long distances.",
            "Drunk drivers racing on {street} near {landmark} every night. Police not taking action.",
            "School buses near {landmark} violating traffic rules and endangering children daily.",
            "Taxi aggregators overcharging commuters during peak hours in {area1} area. Surge pricing abuse.",
        ]
    },
    "OFF008": {
        "priority_weights": {"High": 0.10, "Medium": 0.40, "Low": 0.50},
        "eta": {"High": (3, 7), "Medium": (10, 20), "Low": (20, 45)},
        "templates": [
            "My ration card application has been pending for {days} days with no response from office.",
            "Birth certificate application for my child submitted {days} days ago. No update received.",
            "Pension application for senior citizen in {area1} not processed despite {count} visits to office.",
            "Income certificate application rejected without proper reason. No appeal process explained.",
            "Government scheme benefits not reaching eligible families in {area1} for {count} months.",
            "Property tax payment online portal is non-functional. Unable to pay for {days} days.",
            "Death certificate for my family member not issued even after {days} days of application.",
            "Domicile certificate application stuck at {landmark} government office for {days} days.",
            "RTI application filed {days} days ago. No response provided as required by law.",
            "Corruption alleged at {landmark} government office. Officials demanding bribe for basic services.",
        ]
    },
}

# Fill-in values for template slots
STREETS = ["MG Road", "NH-48 Highway", "Anna Salai", "Linking Road", "FC Road",
           "Residency Road", "Brigade Road", "Baner Road", "Whitefield Main Road", "Outer Ring Road"]
LANDMARKS = ["City Hospital", "Central Park", "Railway Station", "Bus Depot", "Town Hall",
              "Police Station", "Government School", "Market Complex", "Metro Station", "Post Office"]
AREAS = ["Sector 14", "Phase 2", "East Colony", "West Block", "North Extension",
         "South Township", "Industrial Area", "New Layout", "Old Town", "Green Valley"]

# ─────────────────────────────────────────────
# Multilingual complaint prefixes / suffixes
# to simulate Hindi-English (Hinglish) and Tamil-English mix
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# Priority signal phrases injected into text
# These simulate realistic urgency cues that a model can learn
# ─────────────────────────────────────────────
PRIORITY_SIGNALS = {
    "High": [
        "This is a life-threatening emergency. ",
        "People are in immediate danger. Urgent action needed! ",
        "URGENT: This cannot wait. Lives are at risk. ",
        "Emergency situation! Immediate intervention required. ",
        "Critical situation — please respond within 24 hours. ",
        "Accidents have already occurred. Cannot be ignored any longer. ",
    ],
    "Medium": [
        "This is a serious problem affecting daily life. ",
        "The issue has persisted and needs prompt attention. ",
        "Significant inconvenience to local residents. Timely action needed. ",
        "This is escalating and requires attention within a week. ",
        "The problem is getting worse and should be addressed soon. ",
        "Multiple families are affected. Requesting action at the earliest. ",
    ],
    "Low": [
        "This is a minor issue but needs to be addressed eventually. ",
        "Not urgent, but the matter requires attention when convenient. ",
        "A small concern that would improve quality of life if fixed. ",
        "Requesting action at the next available opportunity. ",
        "This is a routine matter that can be handled in due course. ",
        "Low priority but wanted to bring this to your notice. ",
    ],
}

MULTILINGUAL_WRAPPERS = [
    # English-only (majority)
    lambda t: t,
    lambda t: t,
    lambda t: t,
    lambda t: t,
    lambda t: t,
    # Hinglish mix
    lambda t: f"Bahut bada problem hai. {t} Kripya jaldi action lo.",
    lambda t: f"Yeh problem bahut serious hai. {t} Please help karo.",
    lambda t: f"{t} Hamara mohalla bahut pareshan hai. Urgent help chahiye.",
    # Tamil-English mix
    lambda t: f"Romba kastam aaguthu. {t} Thayavu seithu help pannunga.",
    lambda t: f"{t} Ithu romba avasaram. Undan help thevai.",
    # Hindi transliteration
    lambda t: f"Mera complaint yeh hai ki {t.lower()} Is par turant dhyan diya jaye.",
    # Spanish-English (light)
    lambda t: f"Es urgente. {t} Por favor actúe rápidamente.",
    lambda t: f"{t} Necesitamos ayuda urgente en esta área.",
]


def fill_template(template: str) -> str:
    """Replace template placeholders with random values."""
    return (template
            .replace("{street}", random.choice(STREETS))
            .replace("{landmark}", random.choice(LANDMARKS))
            .replace("{area1}", random.choice(AREAS))
            .replace("{area2}", random.choice(AREAS))
            .replace("{days}", str(random.randint(2, 30)))
            .replace("{count}", str(random.randint(2, 50))))


def sample_eta(officer_id: str, priority: str) -> int:
    lo, hi = COMPLAINT_TEMPLATES[officer_id]["eta"][priority]
    return int(np.clip(np.random.normal((lo + hi) / 2, (hi - lo) / 4), lo, hi))


def generate_complaints(n_per_officer: int = 100) -> pd.DataFrame:
    records = []
    officer_map = {o["id"]: o for o in OFFICERS}

    for officer_id, config in COMPLAINT_TEMPLATES.items():
        officer_info = officer_map[officer_id]
        p_weights = config["priority_weights"]
        priorities = list(p_weights.keys())
        weights    = list(p_weights.values())
        templates  = config["templates"]

        for i in range(n_per_officer):
            template_text = random.choice(templates)
            filled_text   = fill_template(template_text)
            priority      = random.choices(priorities, weights=weights, k=1)[0]
            eta           = sample_eta(officer_id, priority)
            wrapper       = random.choice(MULTILINGUAL_WRAPPERS)
            # Inject priority signal phrase (50% chance) for realistic text signals
            signal        = random.choice(PRIORITY_SIGNALS[priority])
            if random.random() < 0.70:
                filled_text = signal + filled_text
            final_text    = wrapper(filled_text)

            records.append({
                "complaint_id":     f"CMP{len(records)+1:04d}",
                "text":             final_text,
                "officer_id":       officer_id,
                "officer_name":     officer_info["name"],
                "department":       officer_info["department"],
                "priority":         priority,
                "eta_days":         eta,
                "language_hint":    "multilingual",
            })

    df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


if __name__ == "__main__":
    import os
    df = generate_complaints(n_per_officer=100)   # 800 total
    out_path = os.path.join(os.path.dirname(__file__), "synthetic_complaints.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} complaints -> {out_path}")
    print("\nClass distribution:")
    print(df["officer_name"].value_counts())
    print("\nPriority distribution:")
    print(df["priority"].value_counts())
    print(f"\nETA stats (days): mean={df['eta_days'].mean():.1f}  "
          f"min={df['eta_days'].min()}  max={df['eta_days'].max()}")
