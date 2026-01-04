import os
import asyncio
import json
import queue
import time
from pathlib import Path
import re
import html as htmlmod
from collections import deque
from urllib.parse import quote
import ctypes.util
portaudio_path = "/nix/store/x44kh3nk8qzjyhs0127x8lv761qg3mx3-portaudio-190700_20210406/lib/libportaudio.so.2"
if os.path.exists(portaudio_path):
    _original_find_library = ctypes.util.find_library
    def _patched_find_library(name):
        if name == 'portaudio':
            return portaudio_path
        return _original_find_library(name)
    ctypes.util.find_library = _patched_find_library

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from local_corrector import LocalCorrector

# =============================================================================
# CONFIG
# =============================================================================

# Use environment variable for API key

# =============================================================================
# OPENAI CONFIG - Transcript Enhancement
# =============================================================================
OPENAI_ENABLED = False  # Set to False to disable OpenAI processing
OPENAI_MODEL = "gpt-4o-mini"  # Cost-effective model for text cleanup
OPENAI_TIMEOUT = 3.0  # Max seconds to wait for OpenAI response

# OpenAI removed (local model used instead)
openai_client = None

OPENAI_SYSTEM_PROMPT = """You are a police radio transcript editor. Your job is to clean up and enhance real-time police radio transcriptions.

RULES:
1. Fix obvious transcription errors and misheard words
2. Use proper police/radio jargon and terminology
3. Keep the original meaning - do NOT add information that wasn't said
4. Format codes properly: 10-4, 10-97, 11-44, etc.
5. When you see 10-27, 10-28, or 10-29 codes (license/registration/wants checks), convert phonetic alphabet names to letters:
   - Adam=A, Boy=B, Charles=C, David=D, Edward=E, Frank=F, George=G, Henry=H
   - Ida=I, John=J, King=K, Lincoln=L, Mary=M, Nora=N, Ocean=O, Paul=P
   - Queen=Q, Robert=R, Sam=S, Tom=T, Union=U, Victor=V, William=W
   - X-ray=X, Yellow=Y, Zebra=Z
   Example: "10-28 on Adam Boy Charles 123" → "10-28 on ABC 123"
6. Preserve callsigns (e.g., "Lincoln 3", "Adam 21", "King 5")
7. Do not hallucinate dollar signs ($) or currency amounts unless explicitly stated
8. Do not remove short identifiers like “3” if contextually they may be unit numbers
9. Keep it concise - this is radio traffic, not prose
10. If the input does not clearly sound like police radio traffic, return it unchanged.

OUTPUT FORMAT:
Return ONLY the cleaned transcript text. No explanations, no quotes, no prefixes."""

# Local model corrector (trained on your logs)
LOCAL_MODEL_DIR = os.environ.get('LOCAL_MODEL_DIR', 'model_corrector_focus')
try:
    local_corrector = LocalCorrector(model_dir=LOCAL_MODEL_DIR)
    print(f"[INFO] LocalCorrector loaded: {LOCAL_MODEL_DIR}")
except Exception as e:
    local_corrector = None
    print(f"[WARN] LocalCorrector not available: {e}")


TRAINING_MODE = True
# -------------------------
# Deepgram / audio settings
# -------------------------
SAMPLE_RATE = 16000
CHANNELS = 1
ENDPOINTING_MS = 0
UTTERANCE_END_MS = 2500

DG_BASE_URL = (
    "wss://api.deepgram.com/v1/listen"
    f"?model=nova-3"
    f"&language=en-US"
    f"&encoding=linear16"
    f"&sample_rate={SAMPLE_RATE}"
    f"&channels={CHANNELS}"
    f"&smart_format=true"
    f"&numerals=true"
    f"&punctuate=true"
    f"&profanity_filter=false"
    f"&interim_results=true"
    f"&endpointing={ENDPOINTING_MS}"
    f"&utterance_end_ms={UTTERANCE_END_MS}"
    f"&vad_events=true"
)

# Minimum confidence threshold for word filtering (0.0-1.0)
MIN_WORD_CONFIDENCE = 0.35
KEEP_ALL_FINAL_WORDS = True  # Preserve context; don't drop low-confidence words in finals

# Tight list of keyterms (helps Deepgram bias without making it too noisy)
PHONETIC_UNITS = [
    "Adam", "Boy", "Charles", "David", "Edward", "Frank", "George", "Henry",
    "Ida", "John", "King", "Lincoln", "Mary", "Nora", "Ocean", "Paul", "Queen",
    "Robert", "Sam", "Tom", "Union", "Victor", "William", "X-ray", "Yellow", "Zebra",
    "Echo",
]

# Enhanced keyterms for police radio - includes common dispatch terminology
KEYTERMS = [
    # Phonetic alphabet
    "Adam", "Boy", "Charles", "David", "Edward", "Frank", "George", "Henry", "Ida",
    "John", "King", "Lincoln", "Mary", "Nora", "Ocean", "Paul", "Queen", "Robert",
    "Sam", "Tom", "Union", "Victor", "William", "X-ray", "Xray", "Yellow", "Zebra", "Echo",
    # Common dispatch terms
    "dispatch", "copy", "repeat", "standby", "code", "clear", "roger", "affirmative",
    "negative", "en route", "on scene", "responding", "advise", "be advised",
    "requesting", "available", "unavailable", "disregard", "acknowledge",
    # Status terms
    "in service", "out of service", "on duty", "off duty", "break", "lunch",
    # Location terms
    "northbound", "southbound", "eastbound", "westbound", "intersection",
    # Unit identifiers
    "K9", "K-9", "canine", "unit", "squad", "patrol", "detective", "sergeant",
    # 10-codes - multiple spelling variations for better recognition
    "10-4", "ten four", "ten-four", "10 4",
    "10-7", "ten seven", "ten-seven", "10 7",
    "10-8", "ten eight", "ten-eight", "10 8",
    "10-9", "ten nine", "ten-nine", "10 9",
    "10-20", "ten twenty", "ten-twenty", "10 20",
    "10-22", "ten twenty two", "ten-twenty-two",
    "10-23", "ten twenty three", "ten-twenty-three",
    "10-28", "ten twenty eight", "ten-twenty-eight", "10 28",
    "10-29", "ten twenty nine", "ten-twenty-nine", "10 29",
    "10-33", "ten thirty three", "ten-thirty-three",
    "10-97", "ten ninety seven", "ten-ninety-seven", "10 97",
    "10-98", "ten ninety eight", "ten-ninety-eight", "10 98",
    # 11-codes
    "11-25", "eleven twenty five", "eleven-twenty-five",
    "11-44", "eleven forty four", "eleven-forty-four",
    "11-99", "eleven ninety nine", "eleven-ninety-nine",
    # Code levels
    "code one", "code 1", "code two", "code 2", "code three", "code 3",
    "code four", "code 4", "code five", "code 5", "code six", "code 6",
    # Alert terms
    "officer down", "shots fired", "pursuit", "foot pursuit", "vehicle pursuit",
    "suspect", "victim", "witness", "complainant", "reporting party", "RP",
    # Vehicle related
    "license plate", "registration", "VIN", "vehicle", "sedan", "SUV", "truck", "van",
    "BOL", "BOLO", "be on the lookout", "APB", "all points bulletin",
    # Radio protocol
    "over", "out", "go ahead", "stand by", "switching", "primary", "tactical",
    # Common H&S codes (spoken as numbers)
    "5150", "fifty one fifty", "5170", "11550", "eleven five fifty",
    # Common PC/VC codes
    "459", "four fifty nine", "burglary", "211", "two eleven", "robbery",
    "187", "one eighty seven", "homicide", "murder", "245", "ADW",
    "10851", "auto theft", "23152", "DUI", "drunk driving",
    "415", "four fifteen", "disturbing", "602", "trespassing",
    # Status codes
    "AID", "public safety", "mutual aid",
    # Common descriptors
    "male", "female", "white", "black", "Hispanic", "Asian",
    "juvenile", "adult", "armed", "unarmed",
    # Directions
    "NB", "SB", "EB", "WB", "north", "south", "east", "west",
    # Dispatch/status terms from logs
    "welfare check", "welfare", "navigation center", "staging", "briefing",
    "area check", "patrol check", "cover", "covering", "primary", "secondary",
    "hit and run", "case number", "case", "incident", "property",
    "lobby", "station", "equipment", "tow", "tow truck",
    # Location types
    "Valero", "Petco", "Whole Foods", "Chick-fil-A", "parking lot",
    "apartments", "center", "residence", "intersection",
    # Vehicle terms
    "expired", "valid", "suspended", "revoked", "pending",
    "Honda", "Toyota", "Chevy", "Ford", "Infiniti", "Tesla", "Acura",
    "Mercedes", "Volkswagen", "Dodge", "sedan", "Civic", "Prius", "Accord",
    # People/relationship terms
    "RO", "registered owner", "mother", "father", "son", "daughter",
    "grandfather", "child", "juvenile", "employee", "manager",
    # Report terms
    "report", "filed", "cancel", "disregard", "affirm", "affirmative",
    "negative", "copy", "clear", "return",
]

# Limit keyterms to avoid URL length issues (Deepgram rejects URLs > ~8000 chars)
# Nova-3 uses "keyterm" parameter (not "keywords" which is for Nova-2)
# Set to 0 to disable keyterms entirely for debugging connection issues
MAX_KEYTERMS = 30  # Re-enabled with safe limit


audio_q: "queue.Queue[bytes]" = queue.Queue()

# -------------------------
# Logging fallback when finals never arrive
# -------------------------
FORCE_LOG_AFTER_SECONDS = 8.0
FORCE_LOG_MIN_INTERVAL = 5.0

# =============================================================================
# Audio tuning (trunked radio direct connection - minimal processing needed)
# =============================================================================
TUNE_ENABLED = True
HP_HZ = 80.0          # Lower cutoff - trunked signal is clean
LP_HZ = 7500.0        # Higher cutoff - preserve more audio detail

GATE_ENABLED = False  # Disabled - strong signal doesn't need noise gate
GATE_RMS = 0.003
GATE_ATTENUATION = 0.5

AGC_ENABLED = True    # Keep AGC but with gentler settings
AGC_TARGET_RMS = 0.035
AGC_MAX_GAIN = 3.0    # Reduced from 8.0 - less aggressive
AGC_MIN_GAIN = 0.5    # Raised from 0.25 - less compression

LIMIT_ENABLED = False
LIMIT_THRESHOLD = 0.85

SOFTCLIP_ENABLED = False

PREEMPH_ENABLED = False  # Disabled - clean signal doesn't need emphasis
PREEMPH = 0.85

# =============================================================================
# PATHS (anchored to script directory)
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
OBS_DIR = BASE_DIR / "obs_text"
OBS_DIR.mkdir(parents=True, exist_ok=True)

OBS_LIVE_FILE = OBS_DIR / "live_caption.txt"
OBS_FINAL_FILE = OBS_DIR / "final_caption.txt"
OBS_CAPTION_LOG_FILE = OBS_DIR / "caption_log.txt"
INCOMING_BLOCKS_FILE = Path(os.environ.get("INCOMING_BLOCKS_FILE", r"D:\radio_test2\incoming_blocks.txt"))
LIVE_MAX_CHARS = 300

FULL_LOG_FILE = OBS_DIR / "full_transcript_log.txt"
FULL_LOG_HTML_FILE = OBS_DIR / "full_transcript_log.html"
OBS_LOWER_THIRD_HTML = OBS_DIR / "lower_third.html"

UNRECOGNIZED_TERMS_LOG = OBS_DIR / "unrecognized_terms.log"
OBS_ALERTS_HTML = OBS_DIR / "alerts.html"

SILENCE_GAP_SECONDS = 4.0
TS_FORMAT = "%Y-%m-%d %H:%M:%S"

# =============================================================================
# LOWER THIRD MODE (HTML)
# =============================================================================
LOWER_THIRD_MODE = True

# OBS Browser Source size suggestion:
#   Width: 1920
#   Height: 360
LOWER_THIRD_WIDTH = 1920
LOWER_THIRD_HEIGHT = 360

# keep just the most recent blocks visible in lower-third OBS overlay
LOWER_THIRD_MAX_BLOCKS = 6

# Full HTML log keeps more history
FULL_HTML_MAX_BLOCKS = 100

# =============================================================================
# 10 / 11 CODES (meaning annotations)
# =============================================================================
CODE_MEANINGS: dict[str, str] = {
    # 10-codes
    "10-1": "Receiving poorly",
    "10-2": "Receiving OK",
    "10-3": "Change channels",
    "10-4": "understood",
    "10-5": "Relay to",
    "10-6": "Busy, standby",
    "10-7": "Out of service",
    "10-7A": "Out of service at home",
    "10-7B": "Out of service - personal",
    "10-7CT": "Out of service, court",
    "10-7FU": "Out of service, follow up",
    "10-7OD": "Out of service - off duty",
    "10-7RW": "Out of service, report writing",
    "10-7T": "Out of service, training",
    "10-8": "In service/available for assignment",
    "10-8FU": "Follow up, but available",
    "10-9": "Repeat last transmission",
    "10-10": "Off duty",
    "10-10A": "Off duty at home",
    "10-11": "Identify this frequency",
    "10-12": "Visitors are present (be discrete)",
    "10-13": "Advise weather and road conditions",
    "10-14": "Citizen holding suspect",
    "10-15": "Prisoner in custody",
    "10-16": "Pick up prisoner",
    "10-17": "Request for gasoline",
    "10-18": "Equipment exchange",
    "10-19": "Return/returning to the station",
    "10-20": "Location?",
    "10-21": "Telephone",
    "10-21A": "Advise home of return time",
    "10-21B": "Phone your home",
    "10-21R": "Phone radio dispatch",
    "10-22": "Disregard the last assignment",
    "10-22C": "Leave area if all secure",
    "10-23": "Standby",
    "10-24": "Request car-to-car transmission",
    "10-25": "Do you have contact with?",
    "10-26": "Clear",
    "10-27": "Driver's license check",
    "10-28": "Vehicle registration request",
    "10-29": "Check wants/warrants (vehicle)",
    "10-29A": "Check wants/warrants (subject)",
    "10-29C": "Check complete (subject)",
    "10-29F": "Subject wanted for felony",
    "10-29H": "Caution - severe hazard potential",
    "10-29M": "Subject wanted for misdemeanor",
    "10-29R": "Check wants/record (subject)",
    "10-29V": "Vehicle wanted in connection with crime",
    "10-30": "Does not conform to regulations",
    "10-31": "Status check/valid registration",
    "31 VALID": "Person/vehicle is valid and clear",
    "31 A VALID": "Person/vehicle is valid and clear",
    "31 SUSPENDED": "License is suspended",
    "31 REVOKED": "License is revoked",
    "10-32": "Drowning",
    "10-33": "Alarm sounding",
    "10-33A": "Audible alarm",
    "10-33S": "Silent alarm",
    "10-34": "Assist at office",
    "10-35": "Time check",
    "10-36": "Confidential information",
    "10-37": "Identify the operator",
    "10-39": "Can unit come to the radio?",
    "10-40": "Is unit available for phone call?",
    "10-42": "Check on the welfare of",
    "10-43": "Call a doctor",
    "10-45": "Condition of patient?",
    "10-45A": "Condition of patient is good",
    "10-45B": "Condition of patient is serious",
    "10-45C": "Condition of patient is critical",
    "10-45D": "Patient is deceased",
    "10-46": "Sick person (ambulance enroute)",
    "10-48": "Ambulance transfer call",
    "10-49": "Proceed to/Enroute to",
    "10-50": "Under influence of narcotics/Take a report",
    "10-51": "Subject is drunk",
    "10-52": "Resuscitator is needed",
    "10-53": "Person down",
    "10-54": "Possible dead body",
    "10-55": "Coroner’s case",
    "10-56": "Suicide",
    "10-56A": "Attempted suicide",
    "10-57": "Firearm discharged",
    "10-58": "Garbage complaint",
    "10-59": "Security check/Malicious mischief",
    "10-60": "Lock out",
    "10-61": "Miscellaneous public service",
    "10-62": "Meet a citizen",
    "10-62A": "Take a report from a citizen",
    "10-62B": "Civil standby",
    "10-62FD": "Citizen flag-down",
    "10-63": "Prepare to copy",
    "10-64": "Found property",
    "10-65": "Missing person",
    "10-65F": "Found missing person",
    "10-65J": "Missing juvenile",
    "10-65JX": "Missing female juvenile",
    "10-65MH": "Missing person, mentally handicapped",
    "10-66": "Suspicious person",
    "10-66P": "Suspicious package",
    "10-66W": "Suspicious person with a weapon",
    "10-66X": "Suspicious female",
    "10-67": "Person calling for help",
    "10-68": "Call for police made via telephone",
    "10-70": "Prowler",
    "10-71": "Shooting",
    "10-72": "Knifing",
    "10-73": "How do you receive?",
    "10-79": "Bomb threat",
    "10-80": "Explosion",
    "10-86": "Any traffic?",
    "10-87": "Meet the officer at",
    "10-88": "Fill with the officer/Assume your post",
    "10-91": "Animal",
    "10-91A": "Stray",
    "10-91B": "Noisy animal",
    "10-91C": "Injured animal",
    "10-91D": "Dead animal",
    "10-91E": "Animal bite",
    "10-91G": "Animal pickup",
    "10-91H": "Stray horse",
    "10-91J": "Pickup/collect",
    "10-91L": "Leash law violation",
    "10-91V": "Vicious animal",
    "10-95": "Pedestrian/Requesting ID/Tech unit",
    "10-96": "Out of vehicle - send backup",
    "10-97": "Arrived at the scene",
    "10-98": "Available for assignment",
    "10-99": "Open police garage door",
    "10-100": "Civil disturbance - Mutual aid standby",
    "10-101": "Civil disturbance - Mutual aid request",

    # 11-codes
    "11-10": "Take a report",
    "11-24": "Abandoned automobile",
    "11-25": "Traffic hazard",
    "11-26": "Abandoned bicycle",
    "11-27": "10-27 with driver being held",
    "11-28": "10-28 with driver being held",
    "11-40": "Advise if ambulance is needed",
    "11-41": "Ambulance is needed",
    "11-42": "No ambulance is needed",
    "11-48": "Furnish transportation",
    "11-51": "Escort",
    "11-52": "Funeral detail",
    "11-54": "Suspicious vehicle",
    "11-55": "Officer being followed by automobile",
    "11-56": "Officer being followed by auto with dangerous persons",
    "11-57": "Unidentified auto at scene of assignment",
    "11-58": "Radio traffic monitored - phone non-routine messages",
    "11-59": "Give intensive attention to high hazard areas",
    "11-60": "Attack in a high hazard area",
    "11-65": "Signal light is out",
    "11-66": "Defective traffic light",
    "11-71": "Fire",
    "11-78": "Aircraft accident",
    "11-79": "Accident - ambulance has been sent",
    "11-80": "Accident - major injuries",
    "11-81": "Accident - minor injuries",
    "11-82": "Accident - no injuries",
    "11-83": "Accident - no details",
    "11-84": "Direct traffic",
    "11-85": "Tow truck required",
    "11-86": "Traffic stop/plate check location",
    "11-94": "Pedestrian stop",
    "11-95": "Routine traffic stop",
    "11-96": "Checking a suspicious vehicle",
    "11-97": "Time/security check on patrol vehicles",
    "11-98": "Meet",
    "11-99": "Officer needs help",

    # 900 Series Codes
    "904": "Fire",
    "904A": "Automobile fire",
    "904B": "Building fire",
    "904G": "Grass fire",
    "909": "Traffic problem - police needed",
    "910": "Can handle this detail",
    "911UNK": "Unknown 911 calls",
    "925": "Suspicious vehicle",
    "932": "Turn on mobile relay",
    "933": "Turn off mobile relay",
    "949": "Burning inspection",
    "950": "Control burn in progress/about to begin/ended",
    "951": "Need fire investigator",
    "952": "Report on conditions",
    "953": "Investigate smoke",
    "953A": "Investigate gas",
    "954": "Off the air at scene of fire",
    "955": "Fire is under control",
    "956": "Assignment not finished",
    "957": "Delayed response",
    "980": "Restrict calls to emergency only",
    "981": "Resume normal traffic",
    "1000": "Plane crash",
    "3000": "Road block",

    # Other Codes
    "CODE 1": "Do so at your convenience",
    "CODE 2": "Urgent",
    "CODE 3": "Emergency/lights and siren",
    "CODE 4": "No further assistance is needed",
    "CODE 5": "Stakeout",
    "CODE 6": "Responding from a long distance",
    "CODE 7": "Mealtime",
    "CODE 8": "Request cover/backup",
    "CODE 9": "Set up a roadblock",
    "CODE 10": "Bomb threat",
    "CODE 12": "Notify news media",
    "CODE 20": "Officer needs assistance",
    "CODE 22": "Restricted radio traffic",
    "CODE 30": "Officer needs HELP - EMERGENCY!",
    "CODE 33": "Mobile emergency - clear this radio channel",
    "CODE 43": "TAC forces committed",
}

# =============================================================================
# PC CODES (California Penal Codes)
# =============================================================================
PC_MEANINGS: dict[str, str] = {
    "32": "Accessory to a felony",
    "67": "Offer a bribe to executive officer",
    "69": "Deter/resist executive officer by threat/force/violence",
    "71": "Threaten injury to school officer or employee",
    "102": "Take or destroy property in custody of officer",
    "118": "Perjury",
    "136.1(a)": "Intimidation of witness/victim from attending/testifying",
    "136.1(b)": "Intimidation of witness/victim from reporting crime",
    "136.1(c)": "Intimidation of witness/victim by force/threat of violence",
    "137(a)": "Offer bribe to influence testimony",
    "146a": "Impersonating a peace officer",
    "148": "Interfering with an officer",
    "148.1": "False report of a bomb",
    "148.5": "False report of a crime",
    "150": "Refuse to aid an officer",
    "151(a)": "Advocate killing/injuring officer",
    "187": "Murder",
    "192.1": "Voluntary manslaughter",
    "192.2": "Involuntary manslaughter",
    "192.3": "Vehicular manslaughter",
    "203": "Mayhem",
    "207": "Kidnap",
    "209a": "Kidnaping for ransom/extortion",
    "209b": "Kidnaping for robbery",
    "211": "Robbery",
    "215": "Carjacking",
    "220": "Assault with intent to mayhem/rape/sodomy/oral copulation",
    "236": "False imprisonment",
    "240": "Assault",
    "241": "Assault on peace officer/EMT/firefighter",
    "242": "Battery",
    "243a": "Battery against a citizen",
    "243b": "Battery against a peace officer",
    "243(f)(4)": "Serious bodily injury defined",
    "244": "Throwing acid with intent to disfigure or burn",
    "245": "Assault witMODEL_IDh a deadly weapon",
    "245b": "Assault with deadly weapon against peace officer",
    "246": "Shooting at an inhabited dwelling or vehicle",
    "261": "Rape",
    "261.5": "Rape - under 18 years of age",
    "262": "Rape of spouse",
    "266h": "Pimping",
    "266i": "Pandering",
    "270": "Child neglect/failing to pay support payments",
    "271": "Child abandonment - under 14",
    "272": "Contributing to the delinquency of a minor",
    "273.5a": "Corporal injury to spouse/cohabitant",
    "273d": "Corporal injury upon child",
    "278": "Child abduction from parent or guardian",
    "285": "Incest",
    "286": "Sodomy",
    "288": "Sex crimes against children",
    "288a": "Oral copulation",
    "290": "Sex registration",
    "311.2a": "Possession of obscene matter",
    "311.2b": "Possessing obscene matter depicting a minor",
    "314": "Indecent exposure",
    "330": "Gambling",
    "373": "Public nuisance misdemeanors",
    "374b": "Garbage dumping",
    "402b": "Abandoned refrigerator",
    "415": "Disturbing the peace",
    "417": "Brandishing a weapon",
    "451": "Arson",
    "459": "Burglary",
    "470": "Forgery",
    "476a": "Insufficient funds (checks)",
    "484": "Theft",
    "484e": "Theft of a credit card",
    "484f": "Forged credit card",
    "484g": "Illegal use of a credit card",
    "487": "Grand Theft ($400+)",
    "487(a)": "Grand Theft",
    "488": "Petty theft (<$400)",
    "496": "Receiving stolen property",
    "499b": "Joyriding",
    "503": "Embezzlement",
    "537": "Nonpayment of a bill (Restaurants, etc.)",
    "537e": "Article with serial number removed",
    "555": "Posted trespass",
    "594": "Vandalism",
    "597": "Killing or abusing animals",
    "602": "Trespass",
    "602L": "Trespass",
    "603": "Trespass with damage",
    "647a": "Annoy/molest child",
    "647b": "Prostitution",
    "647f": "Drunk in public",
    "647g": "Disorderly conduct - loitering on private property at night",
    "647h": "Disorderly conduct - peeking into an inhabited building",
    "653m": "Harassment by phone (obscene call)",
    "835": "Method of Arrest",
    "835a": "Effecting Arrest; Resistance",
    "4532": "Escape",
    "12020": "Possession of a deadly weapon",
    "12025": "Possession of a concealed firearm",
    "12031": "Possession of a loaded firearm",
}

# =============================================================================
# VEHICLE CODES (California Vehicle Codes)
# =============================================================================
VC_MEANINGS: dict[str, str] = {
    "10851": "Auto theft",
    "10852": "Malicious mischief to a vehicle",
    "14601": "Suspended or revoked license",
    "20001": "Hit and run - injury or death",
    "20002": "Hit and run - property damage",
    "21111": "Throwing article",
    "22348": "Maximum speed law - 55 MPH",
    "22350": "Basic speed law - unsafe speed",
    "22500e": "Vehicle blocking a driveway",
    "23109": "Exhibition of speed",
    "23110": "Throwing articles at a vehicle",
    "23112": "Throwing garbage on highway",
    "23152": "Drunk driving",
}

# =============================================================================
# HEALTH AND SAFETY CODES (California H&S Codes)
# =============================================================================
HS_MEANINGS: dict[str, str] = {
    "5150": "Mental/emotional",
    "5170": "Unable to care for self",
    "11350": "Possession (heroin, cocaine, etc.)",
    "11351": "Possession for sales (heroin, cocaine, etc.)",
    "11352": "Sale/transportation (heroin, cocaine, etc.)",
    "11357a": "Possession of hashish",
    "11357b": "Possession of less than 1 ounce marijuana",
    "11357c": "Possession of more than 1 ounce marijuana",
    "11358": "Cultivation of marijuana",
    "11359": "Possession for sales (marijuana)",
    "11360a": "Sale/transportation (marijuana)",
    "11364": "Paraphernalia",
    "11368": "Forged prescription",
    "11377": "Possession (barbiturates, amphetamines, LSD)",
    "11378": "Possession for sale (barbiturates, amphetamines, LSD)",
    "11550": "Under influence of a controlled substance",
    "12677": "Fireworks",
}

# =============================================================================
# KEYWORD / HIGHLIGHT CONFIG
# =============================================================================
ALERT_KEYWORDS = [
    # High-priority CODE series
    "Code 3", "Code 20", "Code 30", "Code 33",
    "Code 6A", "Code 6D", "Code 6F", "Code 6H", "Code 6M",
    # Critical 10-codes
    "10-71", "10-72", "10-53", "10-54", "10-55", "10-56", "10-57",
    "10-79", "10-80", "10-45C", "10-45D", "10-100", "10-101",
    "10-29F", "10-29H",  # Felony warrant, severe hazard
    # Critical 11-codes
    "11-55", "11-56", "11-60", "11-78", "11-80", "11-99",
    # Major crimes (PC)
    "187", "207", "211", "215", "245", "246", "261", "459",
    # 900 series emergencies
    "904", "1000",
    # Mental health emergencies
    "5150",
    # Pursuit/emergency terms
    "pursuit", "shots fired", "officer down", "foot pursuit",
    "armed", "weapon", "hostage", "barricade",
]
LOCATIONS = [
    "Valero", "Petco", "Whole Foods", "Chick-fil-A", "Starbucks", "Target", "Safeway",
    "Main Street", "Elm Avenue", "Central Park", "High School", "Town Hall"
]

PATTERN_LOCATIONS = re.compile(
    r"\\b(" + "|".join(map(re.escape, LOCATIONS)) + r")\\b",
    re.IGNORECASE
)

PATTERN_ALERTS = re.compile(
    r"\b(" + "|".join(map(re.escape, ALERT_KEYWORDS)) + r")\b",
    re.IGNORECASE
) if ALERT_KEYWORDS else None

PATTERN_LOCATIONS = re.compile(
    r"\b(" + "|".join(map(re.escape, LOCATIONS)) + r")\b",
    re.IGNORECASE
) if LOCATIONS else None

# =============================================================================
# CALLSIGN REGEX RULES
# =============================================================================
NUM_WORDS = r"(?:\d{1,4}|one|won|two|to|too|three|four|for|ford|forth|five|six|seven|eight|ate|nine|ten)"

CALLSIGN_SPACED = re.compile(
    r"\b(" + "|".join(PHONETIC_UNITS) + r")\s+(" + NUM_WORDS + r")\b",
    re.IGNORECASE
)
CALLSIGN_JOINED = re.compile(
    r"\b(" + "|".join(PHONETIC_UNITS) + r")(\d{2,4})\b",
    re.IGNORECASE
)

CALLSIGN_START = re.compile(
    r"^\s*(" + "|".join(PHONETIC_UNITS) + r")\s+(" + NUM_WORDS + r")\b",
    re.IGNORECASE
)


_CALLSIGN_NUM_MAP = {
    "one":"1","won":"1","two":"2","to":"2","too":"2","three":"3","tree":"3",
    "four":"4","for":"4","ford":"4","forth":"4","five":"5","six":"6","seven":"7",
    "eight":"8","ate":"8","nine":"9","ten":"10",
}

def _normalize_callsign_prefix(text: str) -> str | None:
    """Return normalized callsign like 'Boy 4' if text begins with a callsign."""
    if not text:
        return None
    m = CALLSIGN_START.search(text)
    if not m:
        return None
    unit = m.group(1)
    num = m.group(2)
    n = num.lower()
    n = _CALLSIGN_NUM_MAP.get(n, num)
    # Title-case the unit (keep special cases like K9 untouched)
    unit_norm = unit[:1].upper() + unit[1:].lower()
    return f"{unit_norm} {n}"

CALLSIGN_MULTI = re.compile(
    r"\b(\d{0,2})\s*(" + "|".join(PHONETIC_UNITS) + r")(?:\s*(" + "|".join(PHONETIC_UNITS) + r")){1,4}\s*(\d{1,4})\b",
    re.IGNORECASE
)

# =============================================================================
# SPEAKER CLASSIFICATION (Dispatcher vs Officer)
# =============================================================================
# Heuristic rule from user: patrol officers identify themselves with a callsign
# (e.g., "Boy 12", "Charles 3", "King 5") before speaking; dispatch does not.
LEADING_CALLSIGN = re.compile(
    r"^\s*(?:" + "|".join(PHONETIC_UNITS) + r")\s*(?:" + NUM_WORDS + r")\b",
    re.IGNORECASE,
)

LEADING_CALLSIGN_JOINED = re.compile(
    r"^\s*(?:" + "|".join(PHONETIC_UNITS) + r")(\d{2,4})\b",
    re.IGNORECASE,
)

def classify_speaker(text: str) -> str:
    """Return 'O' for officer or 'D' for dispatch."""
    if not text:
        return "D"
    t = text.strip()
    # If the line begins with a callsign, treat as officer.
    if LEADING_CALLSIGN.search(t) or LEADING_CALLSIGN_JOINED.search(t):
        return "O"
    return "D"

# =============================================================================
# 10/11 CODE DETECTION (numeric + spoken)  ✅ UPDATED to include joined forms
# =============================================================================
PATTERN_CODE_ANY_NUMERIC = re.compile(
    r"\b("
    r"(?:10|11)\s*[- ]\s*\d{1,3}\s*[A-Z]{0,3}"   # 10-23 / 10 23 / 11-98
    r"|(?:10|11)\d{1,3}[A-Z]{0,3}"               # 10023 / 11098
    r"|CODE\s*\d{1,2}"                           # CODE 3, CODE 20, etc.
    r"|9\d{2}[A-Z]?"                             # 900 series codes (904, 925, 952, etc.)
    r"|1000|3000"                                # Special codes
    r"|911UNK"
    r")\b",
    re.IGNORECASE
)

NUMBER_WORDS = [
    "zero", "oh",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eighteen", "nineteen",
    "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred",
]
PATTERN_CODE_ANY_WORDS = re.compile(
    r"\b(ten|eleven)\s+((?:"
    + "|".join(NUMBER_WORDS)
    + r")(?:[-\s]+(?:"
    + "|".join(NUMBER_WORDS)
    + r")){0,3})\b",
    re.IGNORECASE
)

_WORD_TO_VAL = {
    "zero": 0, "oh": 0,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100,
}

def _parse_number_words_phrase(phrase: str) -> int | None:
    tokens = re.split(r"[-\s]+", phrase.lower().strip())
    tokens = [t for t in tokens if t]
    if not tokens:
        return None
    current = 0
    seen = False
    for t in tokens:
        v = _WORD_TO_VAL.get(t)
        if v is None:
            return None
        seen = True
        if v == 100:
            if current == 0:
                current = 1
            current *= 100
        else:
            current += v
    return current if seen else None

def convert_spoken_codes_to_numeric(text: str) -> str:
    if not text:
        return text

    def repl(m: re.Match) -> str:
        prefix_word = m.group(1).lower()
        rest_phrase = m.group(2)
        prefix = "10" if prefix_word == "ten" else "11" if prefix_word == "eleven" else ""
        n = _parse_number_words_phrase(rest_phrase)
        if not prefix or n is None:
            return m.group(0)
        candidate = f"{prefix}-{n}"
        return candidate if candidate in CODE_MEANINGS else m.group(0)

    return PATTERN_CODE_ANY_WORDS.sub(repl, text)

# ✅ NEW: convert joined numeric codes like 11098 -> 11-98 (only if it’s a known code)
JOINED_10_11 = re.compile(r"\b(?P<prefix>10|11)(?P<body>\d{1,3})(?P<suffix>[A-Z]{0,3})\b", re.IGNORECASE)

def convert_joined_numeric_codes(text: str) -> str:
    if not text:
        return text

    def repl(m: re.Match) -> str:
        prefix = m.group("prefix")
        body_raw = m.group("body")
        suffix = (m.group("suffix") or "").upper()

        try:
            body_int = int(body_raw)
        except ValueError:
            return m.group(0)

        candidate = f"{prefix}-{body_int}{suffix}"
        if candidate in CODE_MEANINGS:
            return candidate

        return m.group(0)

    return JOINED_10_11.sub(repl, text)

def normalize_code_key(raw_code: str) -> str:
    s = raw_code.strip()
    up = s.upper()
    if up in CODE_MEANINGS:
        return up
    if re.match(r"^9\d{2}[A-Z]?$", up) or up in ("1000", "3000", "911UNK"):
        return up
    code_match = re.match(r"^CODE\s*(\d{1,2})$", up, re.IGNORECASE)
    if code_match:
        return f"CODE {code_match.group(1)}"
    up = re.sub(r"\s+", " ", up)
    up = up.replace(" - ", "-").replace(" -", "-").replace("- ", "-")
    up = up.replace(" ", "-")
    up = re.sub(r"\b(10|11)-(\d{1,3})-([A-Z]{1,3})\b", r"\1-\2\3", up)
    return up

# ✅ UPDATED: joined -> spoken -> annotate
def annotate_codes(text: str) -> str:
    """Annotate known 10/11/CODE/900-series codes once (idempotent).

    Avoids re-annotating codes that are already followed by a meaning in parentheses,
    which can happen when post-processing runs multiple times or when upstream text
    already includes annotations.
    """
    if not text:
        return text

    text = convert_joined_numeric_codes(text)
    text = convert_spoken_codes_to_numeric(text)

    out_parts: list[str] = []
    last = 0

    for m in PATTERN_CODE_ANY_NUMERIC.finditer(text):
        raw = m.group(0)

        # If the code is already annotated like "10-4 (understood)", leave it alone.
        after = text[m.end():]
        if re.match(r"\s*\(", after):
            continue

        key = normalize_code_key(raw)
        meaning = CODE_MEANINGS.get(key)
        if not meaning:
            continue

        out_parts.append(text[last:m.start()])
        out_parts.append(f"{raw} ({meaning})")
        last = m.end()

    if not out_parts:
        return text

    out_parts.append(text[last:])
    return "".join(out_parts)
# =============================================================================
# "I'm 97" shorthand => treat as 10-97 if it exists
# =============================================================================
IM_SHORTHAND_NUMERIC = re.compile(
    r"\b(?P<prefix>I'?m|I am|we'?re|we are)\s+(?P<num>\d{1,3})\b",
    re.IGNORECASE
)
IM_SHORTHAND_WORDS = re.compile(
    r"\b(?P<prefix>I'?m|I am|we'?re|we are)\s+(?P<words>(?:"
    + "|".join(NUMBER_WORDS)
    + r")(?:[-\s]+(?:"
    + "|".join(NUMBER_WORDS)
    + r")){0,3})\b",
    re.IGNORECASE
)

def annotate_im_shorthand(text: str) -> str:
    if not text:
        return text

    def repl_num(m: re.Match) -> str:
        prefix = m.group("prefix")
        num = int(m.group("num"))
        key = f"10-{num}"
        meaning = CODE_MEANINGS.get(key)
        if not meaning:
            return m.group(0)
        return f"{prefix} {num} ({meaning})"

    text = IM_SHORTHAND_NUMERIC.sub(repl_num, text)

    def repl_words(m: re.Match) -> str:
        prefix = m.group("prefix")
        words = m.group("words")
        n = _parse_number_words_phrase(words)
        if n is None:
            return m.group(0)
        key = f"10-{n}"
        meaning = CODE_MEANINGS.get(key)
        if not meaning:
            return m.group(0)
        return f"{prefix} {n} ({meaning})"

    text = IM_SHORTHAND_WORDS.sub(repl_words, text)
    return text

# =============================================================================
# PC CODE DETECTION + ANNOTATION
# =============================================================================
PC_REF_1 = re.compile(r"\b(?P<code>\d{1,4}(?:\.\d+)?[a-z]?(?:\([^)]+\))?)\s*PC\b", re.IGNORECASE)
PC_REF_2 = re.compile(r"\bPC\s*(?P<code>\d{1,4}(?:\.\d+)?[a-z]?(?:\([^)]+\))?)\b", re.IGNORECASE)

def annotate_pc_codes(text: str) -> str:
    if not text:
        return text

    def repl1(m: re.Match) -> str:
        raw_code = m.group("code")
        meaning = PC_MEANINGS.get(raw_code) or PC_MEANINGS.get(raw_code.lower())
        if not meaning:
            return m.group(0)
        return f"{raw_code} PC ({meaning})"

    def repl2(m: re.Match) -> str:
        raw_code = m.group("code")
        meaning = PC_MEANINGS.get(raw_code) or PC_MEANINGS.get(raw_code.lower())
        if not meaning:
            return m.group(0)
        return f"PC {raw_code} ({meaning})"

    text = PC_REF_1.sub(repl1, text)
    text = PC_REF_2.sub(repl2, text)
    return text

# =============================================================================
# VC CODE DETECTION + ANNOTATION (Vehicle Codes)
# =============================================================================
VC_REF_1 = re.compile(r"\b(?P<code>\d{4,5}[a-z]?)\s*VC\b", re.IGNORECASE)
VC_REF_2 = re.compile(r"\bVC\s*(?P<code>\d{4,5}[a-z]?)\b", re.IGNORECASE)

def annotate_vc_codes(text: str) -> str:
    if not text:
        return text

    def repl1(m: re.Match) -> str:
        raw_code = m.group("code")
        meaning = VC_MEANINGS.get(raw_code) or VC_MEANINGS.get(raw_code.lower())
        if not meaning:
            return m.group(0)
        return f"{raw_code} VC ({meaning})"

    def repl2(m: re.Match) -> str:
        raw_code = m.group("code")
        meaning = VC_MEANINGS.get(raw_code) or VC_MEANINGS.get(raw_code.lower())
        if not meaning:
            return m.group(0)
        return f"VC {raw_code} ({meaning})"

    text = VC_REF_1.sub(repl1, text)
    text = VC_REF_2.sub(repl2, text)
    return text

# =============================================================================
# HS CODE DETECTION + ANNOTATION (Health & Safety Codes)
# =============================================================================
HS_REF_1 = re.compile(r"\b(?P<code>\d{4,5}[a-z]?)\s*(?:H\s*&?\s*S|HS)\b", re.IGNORECASE)
HS_REF_2 = re.compile(r"\b(?:H\s*&?\s*S|HS)\s*(?P<code>\d{4,5}[a-z]?)\b", re.IGNORECASE)

def annotate_hs_codes(text: str) -> str:
    if not text:
        return text

    def repl1(m: re.Match) -> str:
        raw_code = m.group("code")
        meaning = HS_MEANINGS.get(raw_code) or HS_MEANINGS.get(raw_code.lower())
        if not meaning:
            return m.group(0)
        return f"{raw_code} H&S ({meaning})"

    def repl2(m: re.Match) -> str:
        raw_code = m.group("code")
        meaning = HS_MEANINGS.get(raw_code) or HS_MEANINGS.get(raw_code.lower())
        if not meaning:
            return m.group(0)
        return f"H&S {raw_code} ({meaning})"

    text = HS_REF_1.sub(repl1, text)
    text = HS_REF_2.sub(repl2, text)
    return text

# =============================================================================
# 31 STATUS CODES (e.g., "31 valid", "31 A valid", "31 suspended")
# =============================================================================
STATUS_31_PATTERN = re.compile(
    r"\b(?:he'?s?|she'?s?|it'?s?|that'?s?|is|are|they'?re?)?\s*"
    r"31\s*(?P<modifier>[A-Za-z])?\s*(?P<status>valid|suspended|revoked|clear|expired)\b",
    re.IGNORECASE
)

STATUS_31_MEANINGS = {
    "valid": "Person/vehicle is valid and clear",
    "a valid": "Person/vehicle is valid and clear",
    "suspended": "License is suspended",
    "revoked": "License is revoked",
    "clear": "No wants or warrants",
    "expired": "Registration/license expired",
}

def annotate_status_31(text: str) -> str:
    """Annotate 31 status check responses like '31 valid', '31 A valid', etc."""
    if not text:
        return text
    
    def repl(m: re.Match) -> str:
        modifier = m.group("modifier") or ""
        status = m.group("status").lower()
        
        # Build the lookup key
        if modifier:
            lookup_key = f"{modifier.lower()} {status}"
        else:
            lookup_key = status
        
        meaning = STATUS_31_MEANINGS.get(lookup_key) or STATUS_31_MEANINGS.get(status)
        if not meaning:
            return m.group(0)
        
        # Format the output
        code_part = f"31 {modifier.upper()} {status}" if modifier else f"31 {status}"
        return f"{code_part} ({meaning})"
    
    return STATUS_31_PATTERN.sub(repl, text)

# =============================================================================
# 10-28 "INFO LOOKUP" (phonetic decode)
# =============================================================================
LOOKUP_TRIGGER = re.compile(
    r"\b(10[\s-]?28|1028|ten\s+twenty\s+eight|ten\s+28)\b",
    re.IGNORECASE
)

LOOKUP_STOP = re.compile(
    r"\b("
    r"dob|d\.o\.b|date\s+of\s+birth|birth\s+date|"
    r"cii|c\.i\.i|"
    r"returns?|returned|returning|comes?\s+back|"
    r"record|ro|wants?|probation|parole|"
    r"no\s+record|negative|clear|confirmed"
    r")\b",
    re.IGNORECASE
)

LOOKUP_SEPARATORS = re.compile(
    r"\b(space|last\s+name|surname|family\s+name|first\s+name|middle\s+name|last)\b",
    re.IGNORECASE
)

PHONETIC_TO_LETTER = {
    "adam": "A", "boy": "B", "charles": "C", "david": "D", "edward": "E",
    "frank": "F", "george": "G", "henry": "H", "ida": "I", "john": "J",
    "king": "K", "lincoln": "L", "mary": "M", "nora": "N", "ocean": "O",
    "paul": "P", "queen": "Q", "robert": "R", "sam": "S", "tom": "T",
    "union": "U", "victor": "V", "william": "W",
    "xray": "X", "x-ray": "X", "x": "X",
    "yellow": "Y", "zebra": "Z",
    "echo": "E",
}

def _clean_token(t: str) -> str:
    return t.strip().strip(",.;:!?()[]{}\"'<> ")

def normalize_phonetic_token(tok: str) -> str:
    t = tok.lower().strip()
    t = t.replace(".", "").replace(",", "").replace(";", "").replace(":", "")
    t = t.replace("x ray", "xray").replace("x-ray", "xray")
    return t

def is_number_token(tok: str) -> bool:
    t = tok.lower().strip()
    return re.fullmatch(r"(?:\d{1,4}|one|won|two|to|too|three|four|for|ford|forth|five|six|seven|eight|ate|nine|ten)", t, re.IGNORECASE) is not None

class InfoLookupDecoder:
    def __init__(self, window_seconds: float = 14.0, min_letters_per_word: int = 3):
        self.window_seconds = window_seconds
        self.min_letters_per_word = min_letters_per_word
        self.active_until: float = 0.0
        self.words: list[str] = []
        self.current_letters: list[str] = []

    def reset(self):
        self.active_until = 0.0
        self.words = []
        self.current_letters = []

    def _finalize_current_word(self):
        if self.current_letters:
            w = "".join(self.current_letters)
            self.current_letters = []
            if len(w) >= self.min_letters_per_word:
                self.words.append(w)

    def _extract_letters(self, transcript: str) -> list[str]:
        raw = re.split(r"\s+", transcript.strip())
        out: list[str] = []
        i = 0
        while i < len(raw):
            tok = normalize_phonetic_token(raw[i])
            if tok in PHONETIC_TO_LETTER:
                # skip callsigns like "Adam 12"
                if i + 1 < len(raw):
                    nxt = normalize_phonetic_token(raw[i + 1])
                    if is_number_token(nxt):
                        i += 2
                        continue
                out.append(PHONETIC_TO_LETTER[tok])
            i += 1
        return out

    def _emit_if_ready(self) -> str | None:
        self._finalize_current_word()
        if len(self.words) >= 2:
            result = " ".join(self.words[:2])
            self.reset()
            return result
        return None

    def process_final(self, transcript: str, now: float) -> str | None:
        if not transcript:
            return None

        if LOOKUP_TRIGGER.search(transcript):
            self.active_until = max(self.active_until, now + self.window_seconds)

        if now > self.active_until:
            self.reset()
            return None

        stop_hit = LOOKUP_STOP.search(transcript) is not None
        sep_hit = LOOKUP_SEPARATORS.search(transcript) is not None

        letters = self._extract_letters(transcript)
        if letters:
            self.current_letters.extend(letters)

        if sep_hit:
            self._finalize_current_word()

        if stop_hit:
            out = self._emit_if_ready()
            self.reset()
            return out

        out = self._emit_if_ready()
        if out:
            return out

        return None

lookup_decoder = InfoLookupDecoder()

# =============================================================================
# CASE NUMBER DETECTION
# =============================================================================
CASE_NUMBER_PATTERN = re.compile(
    r"\b(?:case\s+number\s+(?:is\s+)?|case\s*#?\s*)"
    r"(?:ex|x|number)?\s*"
    r"(?P<case>[A-Z]?\s*\d{4,10})\b",
    re.IGNORECASE
)

def annotate_case_numbers(text: str) -> str:
    if not text:
        return text

    def repl(m: re.Match) -> str:
        case_num = m.group("case").replace(" ", "")
        return f"{m.group(0)} [Case #: {case_num}]"

    return CASE_NUMBER_PATTERN.sub(repl, text)

# =============================================================================
# 10-29 PLATE / DL LOOKUP DECODER
# =============================================================================
PLATE_DL_TRIGGER = re.compile(
    r"\b(10[\s-]?29|1029|ten\s+twenty\s+nine|ten\s+29|29\s+on|got\s+a\s+29|run\s+a\s+29)\b",
    re.IGNORECASE
)

DL_INDICATOR = re.compile(
    r"\b(by\s+DL|by\s+driver'?s?\s+licen[cs]e|DL\s+number|driver'?s?\s+licen[cs]e)\b",
    re.IGNORECASE
)

PLATE_STOP = re.compile(
    r"\b("
    r"returns?|returned|coming\s+back|comes?\s+back|"
    r"registered\s+to|ro\s+is|registered\s+owner|"
    r"stolen|wants?|clear|no\s+wants|valid|expired|"
    r"code\s+\d|10[\s-]?\d{2}"
    r")\b",
    re.IGNORECASE
)

NUMBER_WORD_TO_DIGIT = {
    "zero": "0", "oh": "0", "o": "0",
    "one": "1", "won": "1",
    "two": "2", "to": "2", "too": "2",
    "three": "3", "tree": "3",
    "four": "4", "for": "4", "ford": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8", "ate": "8",
    "nine": "9", "niner": "9",
}


# =============================================================================
# 10-27 / 10-28 / 10-29 REQUEST FORMATTER (STRICTLY GATED)
# =============================================================================

CHECK_CODE_RE = re.compile(r"\b(?:10\s*[- ]?\s*)?(27|28|29)\b", re.IGNORECASE)

# Avoid accidental triggers like "28 minutes"
_CHECK_FALSE_FRIENDS = re.compile(r"\b(?:seconds?|minutes?|hours?|days?|years?)\b", re.IGNORECASE)

_PHONETIC_KEYS_RE = re.compile(r"\b(" + "|".join(sorted([re.escape(k) for k in PHONETIC_TO_LETTER.keys()], key=len, reverse=True)) + r")\b", re.IGNORECASE)

def _looks_like_check_request(text: str) -> bool:
    if not text:
        return False

    # Must contain a check code (27/28/29 or 10-27/10-28/10-29)
    if not CHECK_CODE_RE.search(text):
        return False

    # Reject time phrases like "28 minutes" unless explicitly "10-28"
    if _CHECK_FALSE_FRIENDS.search(text) and not re.search(r"\b10\s*[- ]?\s*(27|28|29)\b", text, re.IGNORECASE):
        return False

    # If it's explicit 10-xx, that's a request.
    if re.search(r"\b10\s*[- ]?\s*(27|28|29)\b", text, re.IGNORECASE):
        return True

    # Bare 27/28/29: require either request verbs OR 'on' + phonetics/digits.
    if re.search(r"\b(got|have|run|running|need|request|want|check)\b", text, re.IGNORECASE):
        return True

    if re.search(r"\bon\b", text, re.IGNORECASE) and _PHONETIC_KEYS_RE.search(text):
        return True

    return False

def _phonetic_digits_from_tokens(tokens: list[str]) -> str:
    out: list[str] = []
    for tok in tokens:
        tt = normalize_phonetic_token(tok)
        if tt in PHONETIC_TO_LETTER:
            out.append(PHONETIC_TO_LETTER[tt])
        elif tt in NUMBER_WORD_TO_DIGIT:
            out.append(NUMBER_WORD_TO_DIGIT[tt])
        elif tt.isdigit():
            out.append(tt)
    return "".join(out)

def _extract_after_code(text: str, code_num: str) -> str:
    raw_tokens = re.split(r"\s+", text.strip())

    # Prefer tokens after 'on'
    start = 0
    for i, tok in enumerate(raw_tokens):
        if normalize_phonetic_token(tok) == "on":
            start = i + 1
            break

    # Otherwise, start after the code token
    if start == 0:
        for i, tok in enumerate(raw_tokens):
            ct = _clean_token(tok)
            if not ct:
                continue
            m = CHECK_CODE_RE.fullmatch(ct)
            if m and m.group(1) == code_num:
                start = i + 1
                break
            if re.fullmatch(rf"10\s*[- ]?\s*{code_num}", ct, flags=re.IGNORECASE):
                start = i + 1
                break

    window: list[str] = []
    for tok in raw_tokens[start:start+14]:
        ct = _clean_token(tok)
        if not ct:
            continue
        # stop if we hit another 10-code
        if re.fullmatch(r"10\s*[- ]?\s*\d{2,3}[A-Z]?", ct, flags=re.IGNORECASE):
            break
        window.append(ct)

    converted = _phonetic_digits_from_tokens(window)

    # If no phonetics/digits, try first ID-like token
    if not converted:
        for ct in window:
            if re.fullmatch(r"[A-Za-z]\d{5,9}", ct):  # DL-like
                return ct.upper()
            if re.fullmatch(r"[A-Za-z]{1,4}\d{1,4}[A-Za-z]?", ct):  # plate-ish
                return ct.upper()

    return converted.upper()

def format_check_blocks(text: str) -> str:
    """If a 10-27/28/29 is requested, format a highlighted block with decoded ID."""
    if not text or not _looks_like_check_request(text):
        return text

    matches = list(CHECK_CODE_RE.finditer(text))
    if not matches:
        return text

    blocks: list[str] = []
    for m in matches:
        code_num = m.group(1)
        ident = _extract_after_code(text, code_num)
        if not ident:
            continue

        if code_num == "27":
            blocks.append(f"10-27\nDL # {ident}")
        elif code_num == "28":
            blocks.append(f"10-28\nLicense Plate # {ident}")
        elif code_num == "29":
            blocks.append(f"10-29\nWants/Warrants # {ident}")

    return "\n\n".join(blocks) if blocks else text


def phonetic_to_alphanumeric(tokens: list[str]) -> str:
    result = []
    for tok in tokens:
        t = normalize_phonetic_token(tok)
        if t in PHONETIC_TO_LETTER:
            result.append(PHONETIC_TO_LETTER[t])
        elif t in NUMBER_WORD_TO_DIGIT:
            result.append(NUMBER_WORD_TO_DIGIT[t])
        elif t.isdigit():
            result.append(t)
    return "".join(result)

class PlateDLDecoder:
    def __init__(self, window_seconds: float = 12.0):
        self.window_seconds = window_seconds
        self.active_until: float = 0.0
        self.is_dl_lookup: bool = False
        self.collected_tokens: list[str] = []

    def reset(self):
        self.active_until = 0.0
        self.is_dl_lookup = False
        self.collected_tokens = []

    def _extract_alphanumeric_tokens(self, transcript: str) -> list[str]:
        raw = re.split(r"\s+", transcript.strip())
        out: list[str] = []
        i = 0
        while i < len(raw):
            tok = normalize_phonetic_token(raw[i])
            if tok in PHONETIC_TO_LETTER:
                if i + 1 < len(raw):
                    nxt = normalize_phonetic_token(raw[i + 1])
                    if is_number_token(nxt) and len(nxt) <= 2:
                        i += 2
                        continue
                out.append(tok)
            elif tok in NUMBER_WORD_TO_DIGIT:
                out.append(tok)
            elif tok.isdigit() and len(tok) <= 4:
                out.append(tok)
            i += 1
        return out

    def _emit_if_ready(self) -> str | None:
        if len(self.collected_tokens) >= 4:
            converted = phonetic_to_alphanumeric(self.collected_tokens)
            if len(converted) >= 3:  # allow shorter plates for partial matches
                if self.is_dl_lookup:
                    result = f"DL#: {converted}"
                else:
                    result = f"Plate: {converted}"
                self.reset()
                return result
        return None

    def process_final(self, transcript: str, now: float) -> str | None:
        if not transcript:
            return None

        if PLATE_DL_TRIGGER.search(transcript):
            self.active_until = max(self.active_until, now + self.window_seconds)

        if now > self.active_until:
            self.reset()
            return None

        if DL_INDICATOR.search(transcript):
            self.is_dl_lookup = True

        stop_hit = PLATE_STOP.search(transcript) is not None

        tokens = self._extract_alphanumeric_tokens(transcript)
        if tokens:
            self.collected_tokens.extend(tokens)

        if stop_hit:
            out = self._emit_if_ready()
            self.reset()
            return out

        out = self._emit_if_ready()
        if out:
            return out

        return None

plate_dl_decoder = PlateDLDecoder()

# =============================================================================
# FILE IO (Windows / OBS lock-safe)
# =============================================================================
def atomic_write(path: Path, text: str, retries: int = 20, delay: float = 0.03) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    for _ in range(retries):
        try:
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(delay)
        except OSError:
            time.sleep(delay)
    try:
        path.write_text(text, encoding="utf-8")
    except Exception:
        pass

def append_flush_fsync(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass

def contains_alert(text: str) -> bool:
    if not text or PATTERN_ALERTS is None:
        return False
    return PATTERN_ALERTS.search(text) is not None

def is_probably_noise(text: str) -> bool:
    """Heuristic filter for obvious ASR garbage (e.g., long digit runs).

    Keeps real radio traffic, codes, and plate/DL decodes.
    """
    if not text:
        return True

    # Keep anything that looks like a real code, unit/callsign, or our decoders.
    if re.search(r"\b(10-|11-|CODE\s*\d|PC\b|VC\b|H\s*&\s*S|HS\b|Plate:|DL#:)\b", text, re.IGNORECASE):
        return False
    if CALLSIGN_SPACED.search(text) or CALLSIGN_JOINED.search(text) or CALLSIGN_MULTI.search(text):
        return False
    if re.search(r"\b(?:Adam|Boy|Charles|David|Edward|Frank|George|Henry|Ida|John|King|Lincoln|Mary|Nora|Ocean|Paul|Queen|Robert|Sam|Tom|Union|Victor|William|X-?ray|Yellow|Zebra)\b", text, re.IGNORECASE):
        return False

    compact = re.sub(r"\s+", "", text)
    digits = sum(c.isdigit() for c in compact)
    letters = sum(c.isalpha() for c in compact)

    # Pure/mostly digits (common when ASR latches onto noise)
    if digits >= 6 and letters == 0:
        return True

    # Very digit-heavy with little language signal.
    if len(compact) >= 10 and digits > (letters * 3):
        return True

    return False

def _maybe_split_joined_digits(digits: str):
    if len(digits) == 4 and digits.isdigit():
        unit_digit = digits[0]
        tail = digits[1:]
        tail_i = int(tail)
        if 100 <= tail_i <= 199:
            return unit_digit, tail
    return None, None

# =============================================================================
# HTML HIGHLIGHTING
# =============================================================================
def highlight_to_html(text: str) -> str:
    if not text:
        return ""

    safe = htmlmod.escape(text)

    # Speaker tag highlighting: [D] dispatcher, [O] officer
    safe = re.sub(r"^\[(D|O)\]\s*", lambda m: f"<span class='badge speaker {m.group(1)}'>[{m.group(1)}]</span> ", safe)

    # NOTE: Avoid emoji/icon badges in HTML output.
    # OBS Browser Source (Chromium) often renders emoji/icon glyphs inconsistently.

    if PATTERN_ALERTS:
        safe = PATTERN_ALERTS.sub(r"<span class='hl alert'>\1</span>", safe)

    safe = CALLSIGN_SPACED.sub(r"<span class='hl unit'>\g<0></span>", safe)

    def repl_joined(m: re.Match) -> str:
        unit = m.group(1)
        digits = m.group(2)
        unit_digit, tail = _maybe_split_joined_digits(digits)
        if unit_digit and tail:
            return (
                f"<span class='hl unit'>{htmlmod.escape(unit)} {htmlmod.escape(unit_digit)}</span> "
                f"<span class='hl ten'>{htmlmod.escape(tail)}</span>"
            )
        return f"<span class='hl unit'>{htmlmod.escape(m.group(0))}</span>"

    safe = CALLSIGN_JOINED.sub(repl_joined, safe)
    safe = CALLSIGN_MULTI.sub(r"<span class='hl unit'>\g<0></span>", safe)

    def highlight_10_codes(m):
        code = m.group(0)
        norm = normalize_code_key(code)
        if norm in CODE_MEANINGS:
            return f"<span class='hl ten'>{htmlmod.escape(code)}</span>"
        return f"<span class='hl warn'>{htmlmod.escape(code)}</span>"

    safe = PATTERN_CODE_ANY_NUMERIC.sub(highlight_10_codes, safe)
    safe = LOOKUP_TRIGGER.sub(r"<span class='hl lookup'>\1</span>", safe)
    safe = PLATE_DL_TRIGGER.sub(r"<span class='hl plate'>\1</span>", safe)

    safe = re.sub(
        r"\[Case #: (\w+)\]",
        r"<span class='hl case'>[Case #: \1]</span>",
        safe,
    )

    safe = re.sub(
        r"\bPC\s*\d{1,4}(?:\.\d+)?[a-z]?(?:\([^)]+\))?\b",
        lambda m: f"<span class='hl ten'>{m.group(0)}</span>",
        safe,
        flags=re.IGNORECASE,
    )

    
    # Plate / DL + name/license highlight
    safe = re.sub(r"\b(Plate:|DL#:)\b", r"<span class='hl plate'>\1</span>", safe)
    safe = re.sub(r"\bINFO LOOKUP:\b", r"<span class='hl name'>INFO LOOKUP:</span>", safe)
    if PATTERN_LOCATIONS:
        safe = PATTERN_LOCATIONS.sub(r"<span class='hl loc'>\1</span>", safe)

    return safe

# =============================================================================
# OBS + LOG WRITERS
# =============================================================================
class OBSCaptionWriter:
    def __init__(self):
        self.last_live = ""

    def update_live(self, text: str):
        text = text.strip()
        if not text:
            return
        if len(text) > LIVE_MAX_CHARS:
            text = text[-LIVE_MAX_CHARS:]
        if text == self.last_live:
            return
        self.last_live = text
        atomic_write(OBS_LIVE_FILE, text)

    def write_final(self, text: str):
        text = text.strip()
        if not text:
            return
        atomic_write(OBS_FINAL_FILE, text)
        ts = time.strftime(TS_FORMAT)
        append_flush_fsync(OBS_CAPTION_LOG_FILE, f"[{ts}] {text}\n")


    def write_training_block(
        self,
        raw: str,
        enhanced: str,
        final: str,
        decoded_lookup: str | None = None,
        decoded_plate_dl: str | None = None,
    ):
        """Append a training block in the exact format expected by prep_data.py.

        This writes to INCOMING_BLOCKS_FILE (project root) so run_pipeline.py can import it
        without you manually copying from caption_log.txt.
        """
        raw = (raw or "").strip()
        enhanced = (enhanced or "").strip()
        final = (final or "").strip()
        if not raw and not enhanced and not final:
            return

        lines = []
        lines.append("=== TRAINING MODE ===")
        lines.append(f"[RAW] {raw}")
        lines.append(f"[ENHANCED] {enhanced}")
        lines.append(f"[FINAL] {final}")
        if decoded_lookup:
            lines.append(f"[DECODED LOOKUP] {decoded_lookup.strip()}")
        if decoded_plate_dl:
            lines.append(f"[DECODED PLATE/DL] {decoded_plate_dl.strip()}")
        block = "\n".join(lines) + "\n"


        append_flush_fsync(INCOMING_BLOCKS_FILE, block)


class FullTranscriptLogger:
    def __init__(self, txt_path: Path, html_path: Path, lower_third_path: Path, gap_seconds: float):
        self.txt_path = txt_path
        self.html_path = html_path
        self.lower_third_path = lower_third_path
        self.gap_seconds = gap_seconds
        self.last_write_time: float | None = None
        self.blocks: list[dict] = []
        self.max_blocks = 500
        self._write_html()

    def _ts(self) -> str:
        return time.strftime(TS_FORMAT)

    def add_entry(self, text: str, kind: str = "final", lookup_decoded: str | None = None, plate_dl_decoded: str | None = None):
        text = text.strip()
        if not text:
            return

        now = time.time()
        
        # Check if "break" is in the text - indicates pause between broadcasts
        has_break = bool(re.search(r'\bbreak\b', text, re.IGNORECASE))
        
        start_new_entry = (
            self.last_write_time is None
            or (now - self.last_write_time) >= self.gap_seconds
            or has_break
        )

        if start_new_entry:
            append_flush_fsync(self.txt_path, f"[{self._ts()}] {text}\n")
        else:
            append_flush_fsync(self.txt_path, f"    {text}\n")

        if lookup_decoded:
            append_flush_fsync(self.txt_path, f"    INFO LOOKUP: {lookup_decoded}\n")

        if plate_dl_decoded:
            append_flush_fsync(self.txt_path, f"    {plate_dl_decoded}\n")

        if start_new_entry or not self.blocks:
            self.blocks.append({"ts": self._ts(), "lines": [], "lookups": []})

        line_to_store = text + (" [partial]" if kind == "partial" else "")
        self.blocks[-1]["lines"].append(line_to_store)

        if plate_dl_decoded:
            self.blocks[-1]["lookups"].append(plate_dl_decoded)

        if lookup_decoded:
            self.blocks[-1]["lookups"].append(lookup_decoded)

        if len(self.blocks) > self.max_blocks:
            self.blocks = self.blocks[-self.max_blocks:]

        self.last_write_time = now
        self._write_html()

    def _write_html(self):
        # Write full HTML log (all blocks)
        self._write_html_file(self.html_path, self.blocks[-FULL_HTML_MAX_BLOCKS:] if len(self.blocks) > FULL_HTML_MAX_BLOCKS else self.blocks, is_lower_third=False)
        
        # Write OBS lower-third overlay (limited blocks)
        if LOWER_THIRD_MODE:
            lower_blocks = self.blocks[-LOWER_THIRD_MAX_BLOCKS:] if len(self.blocks) > LOWER_THIRD_MAX_BLOCKS else self.blocks
            self._write_html_file(self.lower_third_path, lower_blocks, is_lower_third=True)

    def _write_html_file(self, path: Path, blocks: list, is_lower_third: bool = False):
        parts = []
        parts.append("<!doctype html>")
        parts.append("<html><head><meta charset='utf-8'>")
        
        if is_lower_third:
            parts.append("""
<style>
  :root {
    --bg: rgba(0,0,0,0.55);
    --bubble: rgba(255,255,255,0.08);
    --text: #ffffff;
  }
  html, body {
    margin:0; padding:0;
    width: 100%; height: 100%;
    overflow: hidden;
    background: transparent;
    font-family: Arial, sans-serif;
    color: var(--text);
  }
  .stage {
    position: relative;
    width: 100%; height: 100%;
    padding: 18px 24px;
    box-sizing: border-box;
  }
  .stack {
    position: absolute;
    left: 24px; right: 24px; bottom: 18px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .block {
    padding: 12px 14px;
    border-radius: 14px;
    background: var(--bg);
    box-shadow: 0 10px 24px rgba(0,0,0,0.35);
    backdrop-filter: blur(6px);
  }
  .ts { font-size: 18px; opacity: 0.80; margin-bottom: 6px; font-weight: 700; }
  .line { font-size: 28px; line-height: 1.20; margin: 0 0 6px 0; font-weight: 700; text-shadow: 0 2px 10px rgba(0,0,0,0.55); }
  .lookupline { margin-top: 8px; padding: 10px 12px; border-radius: 12px; background: var(--bubble); font-size: 24px; font-weight: 700; }
  .hl { padding: 0 8px; border-radius: 10px; font-weight: 900; }
  .hl.alert { background: rgba(255, 0, 0, 0.35); }
  .hl.unit { background: rgba(255, 140, 0, 0.28); }
  .hl.ten { background: rgba(255, 255, 0, 0.28); }
  .hl.loc { background: rgba(200, 150, 255, 0.20); }
  .hl.lookup { background: rgba(0, 255, 120, 0.20); }
  
  .hl.warn { background: rgba(255, 140, 0, 0.25); color: #ffb84d; }
  .hl.uncertain { background: rgba(255, 255, 0, 0.25); color: #ffff66; }

  
  .replay {
    float: right;
    font-size: 14px;
    background: #333;
    color: white;
    border: none;
    padding: 2px 6px;
    border-radius: 6px;
    cursor: pointer;
  }
  .replay:hover {
    background: #555;
  }

  .hl.case { background: rgba(255, 180, 0, 0.25); }
  .block[data-age="old"] { opacity: 0.82; }

  .hl.plate { background: rgba(0, 180, 255, 0.24); }
  .hl.name { background: rgba(0, 255, 120, 0.20); }

  .badge{display:inline-block; padding:0 10px; margin-right:8px; border-radius:999px; background:rgba(255,255,255,0.10); font-weight:900;}
  .badge.unit{background:rgba(255,255,255,0.12)}
  .badge.speaker.D{background:rgba(0,180,255,0.22)}
  .badge.speaker.O{background:rgba(255,140,0,0.22)}
  @keyframes fadeIn{from{opacity:0; transform:translateY(6px);} to{opacity:1; transform:translateY(0);}}
  .line{animation:fadeIn 0.3s ease-in;}
</style>
</head><body>
""".strip())
        else:
            parts.append("""
<style>
  :root {
    --bg: #1a1a2e;
    --bubble: rgba(255,255,255,0.08);
    --text: #ffffff;
  }
  html, body {
    margin: 0; padding: 20px;
    background: var(--bg);
    font-family: Arial, sans-serif;
    color: var(--text);
  }
  .stage { width: 100%; }
  .stack {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .block {
    padding: 14px 18px;
    border-radius: 10px;
    background: rgba(255,255,255,0.05);
    border-left: 4px solid #4a9eff;
  }
  .ts { font-size: 14px; opacity: 0.7; margin-bottom: 8px; font-weight: 600; color: #88aaff; }
  .line { font-size: 16px; line-height: 1.4; margin: 0 0 4px 0; }
  .lookupline { margin-top: 8px; padding: 8px 12px; border-radius: 6px; background: rgba(0,255,150,0.1); font-size: 14px; }
  .hl { padding: 2px 6px; border-radius: 4px; font-weight: 700; }
  .hl.alert { background: rgba(255, 0, 0, 0.35); color: #ff6b6b; }
  .hl.unit { background: rgba(255, 140, 0, 0.22); color: #ffd08a; }
  .hl.ten { background: rgba(255, 255, 0, 0.20); color: #fff59d; }
  .hl.loc { background: rgba(200, 150, 255, 0.15); color: #ce93d8; }
  .hl.lookup { background: rgba(0, 255, 120, 0.15); color: #69f0ae; }
  
  .hl.warn { background: rgba(255, 140, 0, 0.25); color: #ffb84d; }
  .hl.uncertain { background: rgba(255, 255, 0, 0.25); color: #ffff66; }

  
  .replay {
    float: right;
    font-size: 14px;
    background: #333;
    color: white;
    border: none;
    padding: 2px 6px;
    border-radius: 6px;
    cursor: pointer;
  }
  .replay:hover {
    background: #555;
  }

  .hl.case { background: rgba(255, 180, 0, 0.2); color: #ffb74d; }

  .hl.plate { background: rgba(0, 180, 255, 0.18); color: #81d4fa; }
  .hl.name { background: rgba(0, 255, 120, 0.15); color: #69f0ae; }

  .badge{display:inline-block; padding:0 8px; margin-right:6px; border-radius:999px; background:rgba(255,255,255,0.10); font-weight:800;}
  .badge.unit{background:rgba(255,255,255,0.10)}
  .badge.speaker.D{background:rgba(0,180,255,0.18)}
  .badge.speaker.O{background:rgba(255,140,0,0.18)}
  @keyframes fadeIn{from{opacity:0; transform:translateY(4px);} to{opacity:1; transform:translateY(0);}}
  .line{animation:fadeIn 0.3s ease-in;}
</style>
</head><body>
""".strip())

        parts.append("<div class='stage'>")
        parts.append("<div class='stack' id='stack'>")

        for idx, b in enumerate(blocks):
            age = "old" if idx < len(blocks) - 1 else "new"
            parts.append(f"<div class='block' data-age='{age}'>")
            parts.append(f"<div class='ts'>{htmlmod.escape(b['ts'])}</div>")

            for line in b["lines"]:
                parts.append(f"<div class='line' title='[{htmlmod.escape(b['ts'])}]'>{highlight_to_html(line)}</div>")

            for decoded in b.get("lookups", []):
                parts.append(
                    f"<div class='lookupline'><span class='hl lookup'>INFO LOOKUP:</span> {htmlmod.escape(decoded)}</div>"
                )

            parts.append("</div>")

        parts.append("</div></div>")

        parts.append("""
<script>
(function(){
  function pinBottom(){
    try {
      window.scrollTo(0, document.body.scrollHeight);
    } catch(e){}
  }
  pinBottom();
  setInterval(pinBottom, 250);
  setInterval(function(){
    location.reload();
  }, 1500);
})();
</script>
""".strip())

        parts.append("</body></html>")
        atomic_write(path, "\n".join(parts))

obs_writer = OBSCaptionWriter()
full_logger = FullTranscriptLogger(FULL_LOG_FILE, FULL_LOG_HTML_FILE, OBS_LOWER_THIRD_HTML, SILENCE_GAP_SECONDS)

# =============================================================================
# RADIO TUNER DSP
# =============================================================================
class RadioTuner:
    def __init__(self, sr: int):
        self.sr = sr
        self.hp_y = 0.0
        self.hp_x_prev = 0.0
        self.lp_y = 0.0
        self.pre_x_prev = 0.0
        self._update_coeffs()

    def _update_coeffs(self):
        dt = 1.0 / self.sr
        rc_hp = 1.0 / (2.0 * np.pi * max(1.0, HP_HZ))
        self.hp_a = rc_hp / (rc_hp + dt)
        rc_lp = 1.0 / (2.0 * np.pi * max(1.0, LP_HZ))
        self.lp_b = dt / (rc_lp + dt)

    def pre_emphasis(self, x: np.ndarray) -> np.ndarray:
        if not PREEMPH_ENABLED:
            return x
        y = np.empty_like(x)
        prev = self.pre_x_prev
        a = PREEMPH
        for i in range(len(x)):
            xi = x[i]
            y[i] = xi - a * prev
            prev = xi
        self.pre_x_prev = float(prev)
        return y

    def high_pass(self, x: np.ndarray) -> np.ndarray:
        y = np.empty_like(x)
        a = self.hp_a
        y_prev = self.hp_y
        x_prev = self.hp_x_prev
        for i in range(len(x)):
            xi = x[i]
            yi = a * (y_prev + xi - x_prev)
            y[i] = yi
            y_prev = yi
            x_prev = xi
        self.hp_y = float(y_prev)
        self.hp_x_prev = float(x_prev)
        return y

    def low_pass(self, x: np.ndarray) -> np.ndarray:
        y = np.empty_like(x)
        b = self.lp_b
        y_prev = self.lp_y
        for i in range(len(x)):
            xi = x[i]
            y_prev = y_prev + b * (xi - y_prev)
            y[i] = y_prev
        self.lp_y = float(y_prev)
        return y

    def noise_gate(self, x: np.ndarray) -> np.ndarray:
        if not GATE_ENABLED:
            return x
        rms = float(np.sqrt(np.mean(x * x) + 1e-12))
        if rms < GATE_RMS:
            return x * GATE_ATTENUATION
        return x

    def agc(self, x: np.ndarray) -> np.ndarray:
        if not AGC_ENABLED:
            return x
        rms = float(np.sqrt(np.mean(x * x) + 1e-12))
        if rms <= 1e-6:
            return x
        gain = AGC_TARGET_RMS / rms
        gain = float(np.clip(gain, AGC_MIN_GAIN, AGC_MAX_GAIN))
        return x * gain

    def limiter(self, x: np.ndarray) -> np.ndarray:
        if not LIMIT_ENABLED:
            return x
        return np.clip(x, -LIMIT_THRESHOLD, LIMIT_THRESHOLD)

    def softclip(self, x: np.ndarray) -> np.ndarray:
        if not SOFTCLIP_ENABLED:
            return x
        return np.tanh(2.2 * x) / np.tanh(2.2)

    def process(self, x: np.ndarray) -> np.ndarray:
        if not TUNE_ENABLED:
            return x
        x = self.pre_emphasis(x)
        x = self.high_pass(x)
        x = self.low_pass(x)
        x = self.noise_gate(x)
        x = self.agc(x)
        x = self.limiter(x)
        x = self.softclip(x)
        return np.clip(x, -1.0, 1.0)

tuner = RadioTuner(SAMPLE_RATE)

def audio_callback(indata, frames, time_info, status):
    if status:
        pass
    x = indata[:, 0].astype(np.float32)
    x = tuner.process(x)
    pcm16 = (x * 32767.0).astype(np.int16).tobytes()
    audio_q.put(pcm16)

# =============================================================================
# POST-PROCESS PIPELINE
# =============================================================================

# Common misrecognitions to fix
MISRECOGNITION_FIXES = [
    (r'\bfour\b', '4'),
    (r'\bfive\b', '5'),
    (r'\bsix\b', '6'),
    (r'\bseven\b', '7'),
    (r'\beight\b', '8'),
    (r'\bnine\b', '9'),
    (r'\bwon\b', '1'),
    (r'\btoo\b', '2'),
    (r'\btree\b', '3'),
    (r'\bcharking\b', 'Charles King'),
    (r'\bfour fifteen\b', '415'),
    (r'\bfour\s+fifty\s+nine\b', '459'),

    # 10-4 variations (very common)
    (r'\bgo for it\b', '10-4'),
    (r'\bgo for for\b', '10-4'),
    (r'\bgo 4\b', '10-4'),
    (r'\bten-four\b', '10-4'),
    (r'\bten four\b', '10-4'),
    (r'\b104\b', '10-4'),
    (r'\bcopy that\b', '10-4'),
    
    # Plus sign prefix fixes (+104 -> 10-4)
    (r'\+1\s*0?\s*4\b', '10-4'),
    (r'\+1\s*0?\s*7\b', '10-7'),
    (r'\+1\s*0?\s*8\b', '10-8'),
    (r'\+1\s*0?\s*9\b', '10-9'),
    (r'\+1\s*0?\s*14\b', '10-14'),
    (r'\+1\s*0?\s*22\b', '10-22'),
    (r'\+1\s*0?\s*28\b', '10-28'),
    (r'\+1\s*0?\s*29\b', '10-29'),
    (r'\+1\s*0?\s*62\b', '10-62'),
    (r'\+1\s*0?\s*97\b', '10-97'),
    (r'\+1\s*0?\s*98\b', '10-98'),
    
    # Decimal fixes (0.4 -> 10-4, 0.14 -> 10-14)
    (r'\b0\.1\b', '10-1'),
    (r'\b0\.3\b', '10-3'),
    (r'\b0\.4\b', '10-4'),
    (r'\b0\.7\b', '10-7'),
    (r'\b0\.14\b', '10-14'),
    
    # Joined numbers (108 -> 10-8, 1097 -> 10-97, etc.)
    (r'\b108\b', '10-8'),
    (r'\b1022\b', '10-22'),
    (r'\b1028\b', '10-28'),
    (r'\b1029\b', '10-29'),
    (r'\b1031\b', '10-31'),
    (r'\b1062\b', '10-62'),
    (r'\b1086\b', '11-86'),
    (r'\b1097\b', '10-97'),
    (r'\b1098\b', '10-98'),
    (r'\b11098\b', '11-98'),
    (r'\b1186\b', '11-86'),
    (r'\b1199\b', '11-99'),
    
    # Context-aware code detection ("put me at 97" = 10-97, "will be 49" = 10-49)
    (r'\bput me (?:at|on)\s+97\b', '10-97'),
    (r'\bput me (?:at|on)\s+98\b', '10-98'),
    (r'\bshow me\s+97\b', '10-97'),
    (r'\bshow me\s+98\b', '10-98'),
    (r'\bshow me\s+8\b', '10-8'),
    (r'\bshow me\s+7\b', '10-7'),
    (r'\b(?:will be|be)\s+49\b', '10-49'),
    (r'\b(?:will be|be)\s+97\b', '10-97'),
    (r'\b(?:will be|be)\s+98\b', '10-98'),
    (r"\b(?:I'?m\s+)?going\s+49\b", '10-49'),
    (r'\b49\s+(?:to|from|south|north|east|west)\b', '10-49'),
    
    # Standalone codes after unit callsigns (e.g., "charles 7 49" -> 10-49)
    (r'\b49\s+from\b', '10-49 from'),
    
    # "tonight" misheard as 10-8
    (r'\btonight\b(?=\s*\.?\s*$)', '10-8'),
    
    # 11-86 traffic stop variations
    (r'\b11\s*86\b', '11-86'),
    (r'\b11-86\b', '11-86'),
    
    # 5150 (mental health hold) - very common
    (r'\bfifty\s*one\s*fifty\b', '5150'),
    (r'\b51\s*50\b', '5150'),
    (r'\bfive\s*one\s*five\s*zero\b', '5150'),
    
    # 11-99 officer needs help
    (r'\beleven\s*ninety\s*nine\b', '11-99'),
    (r'\beleven\s*99\b', '11-99'),
    
    # Common spoken code patterns
    (r'\bcode\s*three\b', 'Code 3'),
    (r'\bcode\s*four\b', 'Code 4'),
    (r'\bcode\s*five\b', 'Code 5'),
    (r'\bcode\s*six\b', 'Code 6'),
    (r'\bcode\s*seven\b', 'Code 7'),
    
    # PC codes spoken
    (r'\bfour\s*fifty\s*nine\b', '459'),
    (r'\btwo\s*eleven\b', '211'),
    (r'\bone\s*eighty\s*seven\b', '187'),
    (r'\bfour\s*fifteen\b', '415'),
    
    # Phonetic homophones (Deepgram mishears)
    (r'\bvector\b', 'Victor'),
    (r'\bkan\b', 'King'),
    (r'\bcharking\b', 'Charles King'),
    (r'\bshutty\b', 'Charlie'),
    (r'\bcharlie\b', 'Charles'),
    (r'\benvoy\b', 'Edward'),
    
    # "ford" after phonetics = 4 (e.g., "charles ford" = "Charles 4")
    (r'\b(Adam|Boy|Charles|David|Edward|Frank|George|Henry|Ida|John|King|Lincoln|Mary|Nora|Ocean|Paul|Queen|Robert|Sam|Tom|Union|Victor|William|Zebra)\s+ford\b', r'\1 4'),
    (r'\b(Adam|Boy|Charles|David|Edward|Frank|George|Henry|Ida|John|King|Lincoln|Mary|Nora|Ocean|Paul|Queen|Robert|Sam|Tom|Union|Victor|William|Zebra)\s+for\b(?=\s+\d)', r'\1'),
    
    # Common abbreviations/terms
    (r'\bR\.?P\.?\b', 'RP'),  # Reporting Party
    (r'\bR\.?O\.?\b', 'RO'),  # Registered Owner
    (r'\bB\.?O\.?L\.?\b', 'BOL'),  # Be On Lookout
    (r'\bB\.?O\.?L\.?O\.?\b', 'BOLO'),
    (r'\bA\.?D\.?W\.?\b', 'ADW'),  # Assault Deadly Weapon
    (r'\bD\.?U\.?I\.?\b', 'DUI'),
    (r'\bG\.?O\.?A\.?\b', 'GOA'),  # Gone On Arrival
    (r'\bA\.?T\.?L\.?\b', 'ATL'),  # Attempt To Locate
    (r'\bU\.?T\.?L\.?\b', 'UTL'),  # Unable To Locate
    (r'\bD\.?V\.?\b', 'DV'),  # Domestic Violence
    (r'\bP\.?D\.?\b', 'PD'),  # Police Department
    (r'\bA\.?P\.?S\.?\b', 'APS'),  # Adult Protective Services
    
    # Direction abbreviations
    (r'\bnorth\s*bound\b', 'NB'),
    (r'\bsouth\s*bound\b', 'SB'),
    (r'\beast\s*bound\b', 'EB'),
    (r'\bwest\s*bound\b', 'WB'),
    
    # Clean up trailing periods after numbers (e.g., "97." -> "10-97")
    (r'\b(29|49|97|98)\.\s*$', r'10-\1'),
]


mistake_counter = {}

def track_misrecognition_fix(pattern):
    if pattern not in mistake_counter:
        mistake_counter[pattern] = 0
    mistake_counter[pattern] += 1

def print_top_mistakes():
    print("\n--- Top Misrecognition Fixes ---")
    top = sorted(mistake_counter.items(), key=lambda x: x[1], reverse=True)[:10]
    for pat, count in top:
        print(f"{pat}: {count} hits")


# Build phonetic separation pattern - matches consecutive phonetic words without spaces
PHONETIC_NAMES_LOWER = [p.lower() for p in PHONETIC_UNITS]
PHONETIC_PATTERN = re.compile(
    r'(' + '|'.join(PHONETIC_UNITS) + r')(' + '|'.join(PHONETIC_UNITS) + r')',
    re.IGNORECASE
)

def fix_misrecognitions(text: str) -> str:
    """Fix common speech recognition errors"""
    for pattern, replacement in MISRECOGNITION_FIXES:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

def separate_phonetic_letters(text: str) -> str:
    """Separate phonetic letters that run together like 'CharlesQueenLincoln' -> 'Charles Queen Lincoln'"""
    prev = ""
    while prev != text:
        prev = text
        text = PHONETIC_PATTERN.sub(r'\1 \2', text)
    return text

def separate_phonetics_from_numbers(text: str) -> str:
    """Separate phonetics from adjacent numbers like '5ZebraHenry' -> '5 Zebra Henry' or 'Zebra336' -> 'Zebra 336'"""
    pattern = re.compile(
        r'(\d)(' + '|'.join(PHONETIC_UNITS) + r')',
        re.IGNORECASE
    )
    text = pattern.sub(r'\1 \2', text)
    
    pattern2 = re.compile(
        r'(' + '|'.join(PHONETIC_UNITS) + r')(\d)',
        re.IGNORECASE
    )
    text = pattern2.sub(r'\1 \2', text)
    return text

# Pattern: phonetic name followed by combined numbers (with or without space)
# e.g., "lincoln3104" or "charles 497" -> split after unit number
CALLSIGN_NUMBER_PATTERN = re.compile(
    r'\b(' + '|'.join(PHONETIC_UNITS) + r')\s*(\d{1,2})(\d{2,3})\b',
    re.IGNORECASE
)

def split_callsign_from_code(text: str) -> str:
    """Split combined callsign+numbers like 'lincoln3104' into 'lincoln 3 104'"""
    def replacer(m):
        phonetic = m.group(1)
        unit_num = m.group(2)
        remaining = m.group(3)
        return f"{phonetic} {unit_num} {remaining}"
    return CALLSIGN_NUMBER_PATTERN.sub(replacer, text)




# =============================================================================
# CONTEXTUAL FIX: "Tom" misheard for "to"
# =============================================================================
# Deepgram sometimes outputs the phonetic word "Tom" when the speaker actually said "to".
# We ONLY correct this in safe grammatical contexts and NEVER when it looks like a callsign
# (e.g., "Tom 12") or when used as part of phonetic strings for lookups.
_TOM_TO_TO_CONTEXT = re.compile(
    r"\b(?P<prev>"
    r"able|about|ready|supposed|"
    r"want(?:s|ed)?|need(?:s|ed)?|try(?:ing|)|tries|going|gonna|gotta|"
    r"ask(?:ed|ing)?|told|tell(?:ing)?|"
    r"help(?:ing)?|copy"
    r")\s+Tom(?!\s*\d)\s+(?P<nextword>[a-z][a-z']{1,})\b",
    re.IGNORECASE
)

def fix_tom_to_to(text: str) -> str:
    if not text:
        return text
    return _TOM_TO_TO_CONTEXT.sub(lambda m: f"{m.group('prev')} to {m.group('nextword')}", text)


# =============================================================================
# CONTEXT MEMORY + FUZZY PHONETIC FIXES + UNRECOGNIZED TERM LOGGING
# =============================================================================

# Sliding memory of recent unit calls (helps fix missing phonetic like: "3 good night" -> "Charles 3 good night")
RECENT_CALLSIGNS_MAX = 5
recent_callsigns = deque(maxlen=RECENT_CALLSIGNS_MAX)  # stores strings like "Charles 3"
recent_unit_by_number: dict[str, str] = {}  # "3" -> "Charles"

# Simple Levenshtein distance (tiny + fast for short tokens)
def _levenshtein(a: str, b: str, max_dist: int = 2) -> int:
    if a == b:
        return 0
    if abs(len(a) - len(b)) > max_dist:
        return max_dist + 1
    if not a or not b:
        return max(len(a), len(b))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        # early exit tracking
        best_row = max_dist + 1
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            v = ins if ins < dele else dele
            v = sub if sub < v else v
            cur.append(v)
            if v < best_row:
                best_row = v
        prev = cur
        if best_row > max_dist:
            return max_dist + 1
    return prev[-1]

_PHONETIC_CANON = {p.lower(): p for p in PHONETIC_UNITS}
_PHONETIC_KEYS = list(_PHONETIC_CANON.keys())

def fuzzy_fix_phonetics(text: str) -> str:
    """Fix near-miss phonetic tokens (e.g., 'boi' -> 'Boy')."""
    if not text:
        return text
    tokens = re.split(r"(\s+)", text)  # keep whitespace
    for i, tok in enumerate(tokens):
        if tok.isspace():
            continue
        raw = re.sub(r"[^A-Za-z-]", "", tok).lower()
        if not raw or raw in _PHONETIC_CANON:
            continue
        # only consider plausible phonetic-ish tokens
        if not (2 <= len(raw) <= 8):
            continue
        best = None
        best_d = 3
        for cand in _PHONETIC_KEYS:
            d = _levenshtein(raw, cand, max_dist=1)
            if d < best_d:
                best_d = d
                best = cand
                if d == 0:
                    break
        if best is not None and best_d <= 1:
            # replace only the alpha chunk, preserve punctuation around it
            tokens[i] = re.sub(re.escape(re.sub(r"[^A-Za-z-]", "", tok)), _PHONETIC_CANON[best], tok, flags=re.IGNORECASE) if re.search(r"[A-Za-z-]", tok) else tok
    return "".join(tokens)

def _extract_callsigns(text: str) -> list[str]:
    out = []
    if not text:
        return out
    for m in CALLSIGN_SPACED.finditer(text):
        out.append(f"{m.group(1).title()} {m.group(2)}")
    for m in CALLSIGN_JOINED.finditer(text):
        out.append(f"{m.group(1).title()} {m.group(2)}")
    return out

def update_callsign_memory(text: str) -> None:
    for cs in _extract_callsigns(text):
        recent_callsigns.append(cs)
        parts = cs.split()
        if len(parts) == 2:
            recent_unit_by_number[str(parts[1])] = parts[0]

def fix_short_responses(text: str) -> str:
    """If a line starts with a bare unit number, try to restore missing phonetic from recent context."""
    if not text:
        return text
    t = text.strip()
    # e.g., "3 good night" or "3, good night"
    m = re.match(r"^(?P<num>\d{1,2})(?P<rest>\b.*)$", t)
    if m:
        num = m.group("num")
        rest = m.group("rest").lstrip()
        unit = recent_unit_by_number.get(num)
        if unit and not re.match(r"^(?:10|11)\s*[- ]?\d", t):  # don't touch actual 10/11 codes
            return f"{unit} {num} {rest}".strip()
    return text

def log_unrecognized_terms(raw_text: str, processed_text: str, decoded_plate_dl: str | None = None) -> None:
    """Append unknown codes/terms for later rule improvement."""
    try:
        ts = time.strftime(TS_FORMAT)
        unknowns = []

        # Unknown 10/11/code tokens
        for m in re.finditer(r"\b(?:(?:10|11)[- ]?\d{1,3}[A-Z]{0,3}|CODE\s*\d{1,2}|9\d{2}[A-Z]?|911UNK|1000|3000)\b", processed_text, flags=re.IGNORECASE):
            key = normalize_code_key(m.group(0))
            if key not in CODE_MEANINGS:
                unknowns.append(m.group(0))

        # If a plate/DL lookup was triggered but we failed to decode
        if PLATE_DL_TRIGGER.search(raw_text) and not decoded_plate_dl:
            unknowns.append("PLATE/DL_UNDECODED")

        if unknowns:
            line = f"[{ts}] {', '.join(sorted(set(unknowns)))} | RAW: {raw_text} | OUT: {processed_text}\n"
            append_flush_fsync(UNRECOGNIZED_TERMS_LOG, line)
    except Exception:
        pass


def _should_use_openai(text: str) -> bool:
    """Heuristic gate: only use OpenAI when the line looks like radio traffic and has enough context."""
    if not text:
        return False
    t = text.strip()
    # Too short / ambiguous (avoids "please" -> 10-4)
    if len(t) < 14:
        return False

    tl = t.lower()

    # If the line already contains parenthetical meanings, don't ask OpenAI to rewrite it.
    if "(" in t and ")" in t:
        return False

    # Avoid digit-heavy/noise-like lines (common ASR garbage).
    compact = re.sub(r"\s+", "", t)
    digits = sum(c.isdigit() for c in compact)
    letters = sum(c.isalpha() for c in compact)
    if len(compact) >= 10 and digits > (letters * 2) and not re.search(r"\b(?:10|11)\s*[- ]\s*\d{1,3}\b", t):
        return False

    # Looks like radio traffic if it has any of these signals
    signals = 0

    # Codes or numeric patterns
    if re.search(r"\b(?:10|11)\s*[- ]\s*\d{1,3}\b", t) or re.search(r"\b(?:10|11)\d{2,4}\b", t):
        signals += 1
    if re.search(r"\bcode\s*\d{1,2}\b", tl):
        signals += 1
    if re.search(r"\b(?:pc|vc|h&s|hs)\b", tl):
        signals += 1

    # Callsign patterns
    if CALLSIGN_SPACED.search(t) or CALLSIGN_JOINED.search(t) or CALLSIGN_MULTI.search(t):
        signals += 1

    # Dispatch / radio jargon
    if re.search(r"\b(dispatch|copy|roger|affirmative|negative|en route|on scene|responding|be advised|standby|go ahead|clear|break)\b", tl):
        signals += 1

    # Plate/DL/lookup language
    if PLATE_DL_TRIGGER.search(t) or LOOKUP_TRIGGER.search(t) or re.search(r"\b(plate|dl|license|registration|wants|warrant)\b", tl):
        signals += 1

    return signals >= 1


_CODE_EXTRACT = re.compile(r"\b(?:10|11)\s*[- ]\s*\d{1,3}[A-Z]{0,3}\b|\b(?:10|11)\d{1,3}[A-Z]{0,3}\b|\bCODE\s*\d{1,2}\b|\b9\d{2}[A-Z]?\b|\b(?:911UNK|1000|3000)\b", re.IGNORECASE)

def _extract_codes(text: str) -> set[str]:
    if not text:
        return set()
    found = set()
    for m in _CODE_EXTRACT.finditer(text):
        found.add(normalize_code_key(m.group(0)))
    return found

def _is_openai_output_safe(raw_in: str, out: str) -> bool:
    """Reject hallucinated codes or large content additions."""
    if not out:
        return False

    raw_codes = _extract_codes(post_process_transcript(raw_in))
    out_codes = _extract_codes(post_process_transcript(out))

    # Do not allow OpenAI to introduce new codes not present in input.
    if not out_codes.issubset(raw_codes):
        return False

    # If output is wildly longer than input, likely hallucination.
    in_len = len(raw_in.strip())
    out_len = len(out.strip())
    if in_len > 0 and out_len > int(in_len * 1.6) + 12:
        return False

    return True

def enhance_with_local_model(text: str) -> str:
    """Use the locally trained model to lightly clean up radio text (guarded)."""
    if not text or len(text.strip()) < 5:
        return text

    # Reuse the existing heuristic gate (originally for OpenAI) to avoid rewriting short/ambiguous lines.
    if not _should_use_openai(text):
        return text

    if not local_corrector:
        return text

    try:
        enhanced = (local_corrector.correct(text) or "").strip()
        if not enhanced:
            return text

        # Keep the same safety policy: do not introduce new radio codes.
        if _is_openai_output_safe(text, enhanced):
            return enhanced

        return text
    except Exception as e:
        print(f"[LOCAL MODEL ERROR] {e} - using original text")
        return text

def post_process_transcript(text: str) -> str:
    if not text:
        return text
    # Fuzzy phonetic recovery first (helps downstream rules)
    text = fuzzy_fix_phonetics(text)
    text = fix_misrecognitions(text)
    text = separate_phonetic_letters(text)
    text = separate_phonetics_from_numbers(text)
    text = split_callsign_from_code(text)
    text = fix_tom_to_to(text)

    # Context-aware restoration for short responses
    text = fix_short_responses(text)

    text = annotate_im_shorthand(text)
    text = annotate_codes(text)
    text = annotate_pc_codes(text)
    text = annotate_vc_codes(text)
    text = annotate_hs_codes(text)
    text = annotate_status_31(text)
    text = annotate_case_numbers(text)

    # Update callsign memory after all formatting
    update_callsign_memory(text)
    return text

# =============================================================================
# WEBSOCKET TASKS
# =============================================================================


# =============================================================================
# LOCAL ASR (faster-whisper) — replaces Deepgram
# =============================================================================

ASR_MODEL_ID = os.environ.get("ASR_MODEL_ID", "large-v3")  # or "large-v3-turbo"
ASR_DEVICE = os.environ.get("ASR_DEVICE", "cpu")  # cpu or cuda
ASR_COMPUTE_TYPE = os.environ.get("ASR_COMPUTE_TYPE", "float32")  # cpu: int8; cuda: float16/int8_float16

ASR_CHUNK_SEC = float(os.environ.get("ASR_CHUNK_SEC", "4"))
ASR_OVERLAP_SEC = float(os.environ.get("ASR_OVERLAP_SEC", "1"))
ASR_BEAM_SIZE = int(os.environ.get("ASR_BEAM_SIZE", "5"))

# Controls utterance finalization
ASR_SILENCE_SEC = float(os.environ.get("ASR_SILENCE_SEC", str(SILENCE_GAP_SECONDS)))


def _bytes_to_float32_pcm(pcm16_bytes: bytes) -> np.ndarray:
    a = np.frombuffer(pcm16_bytes, dtype=np.int16)
    return (a.astype(np.float32) / 32768.0).copy()


def _norm_words(s: str):
    # Normalize for overlap matching: lowercase, strip punctuation
    if not s:
        return []
    words = []
    for w in s.split():
        w2 = re.sub(r"[^A-Za-z0-9']+", "", w).lower()
        if w2:
            words.append(w2)
    return words

def _word_overlap_delta(prev: str, curr: str) -> str:
    """Return only the new tail of curr compared to prev using word overlap (normalized).

    This prevents repeating the same chunk over and over when chunking with overlap.
    """
    prev = (prev or "").strip()
    curr = (curr or "").strip()
    if not curr:
        return ""
    if not prev:
        return curr

    # If identical after normalization, no delta
    pw_n = _norm_words(prev)
    cw_n = _norm_words(curr)
    if pw_n and cw_n and pw_n == cw_n:
        return ""

    pw = prev.split()
    cw = curr.split()

    # Find best overlap of normalized word suffix/prefix, but slice using original cw
    max_k = min(len(pw_n), len(cw_n))
    best = 0
    for k in range(1, max_k + 1):
        if pw_n[-k:] == cw_n[:k]:
            best = k

    if best > 0:
        return " ".join(cw[best:]).strip()

    # If curr (normalized) is fully contained in prev (normalized), treat as no new info
    prev_n_join = " ".join(pw_n)
    curr_n_join = " ".join(cw_n)
    if curr_n_join and curr_n_join in prev_n_join:
        return ""

    # Otherwise return curr (as-is)
    return curr

    pw = prev.split()
    cw = curr.split()
    max_k = min(len(pw), len(cw))
    best = 0
    for k in range(1, max_k + 1):
        if pw[-k:] == cw[:k]:
            best = k
    if best > 0:
        return " ".join(cw[best:]).strip()
    # If curr is a substring of prev, no new info
    if curr in prev:
        return ""
    return curr


def process_utterance_text(raw_text: str, now: float):
    """Run the exact same post-process pipeline used for Deepgram utterances."""
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return

    combined = raw_text

    # First pass: basic regex processing
    combined_processed = post_process_transcript(combined)

    # Second pass: local model enhancement
    combined_enhanced = enhance_with_local_model(combined_processed)

    # Third pass: re-apply regex rules / formatting
    combined_final = post_process_transcript(combined_enhanced)

    # Drop obvious ASR garbage
    if is_probably_noise(combined_final):
        return

    decoded_lookup = lookup_decoder.process_final(combined, now)
    decoded_plate_dl = plate_dl_decoder.process_final(combined, now)

    # Speaker tagging (skip if formatted 10-27/28/29 block)
    if not (combined_final.strip().startswith("10-") and "\n" in combined_final):
        speaker_tag = classify_speaker(combined_final)
        combined_final = f"[{speaker_tag}] {combined_final}"

    # Auto-learn unknown terms / failed decodes
    log_unrecognized_terms(combined, combined_final, decoded_plate_dl)

    alert = contains_alert(combined)
    caption_text = f"🚨 {combined_final}" if alert else combined_final

    append_flush_fsync(OBS_CAPTION_LOG_FILE, f"    [RAW] {combined}\n")
    append_flush_fsync(OBS_CAPTION_LOG_FILE, f"    [ENHANCED] {combined_enhanced}\n")
    append_flush_fsync(OBS_CAPTION_LOG_FILE, f"    [FINAL] {combined_final}\n")

    if TRAINING_MODE:
        print("=== TRAINING MODE ===")
        print(f"[RAW] {combined}")
        print(f"[ENHANCED] {combined_enhanced}")
        print(f"[FINAL] {combined_final}")
        obs_writer.write_training_block(combined, combined_enhanced, combined_final, decoded_lookup=decoded_lookup, decoded_plate_dl=decoded_plate_dl)
        if decoded_lookup:
            print(f"[DECODED LOOKUP] {decoded_lookup}")
        if decoded_plate_dl:
            print(f"[DECODED PLATE/DL] {decoded_plate_dl}")
        print("=" * 50)

    print(f"\r{' ' * 120}\r{combined_final}")
    obs_writer.write_final(caption_text)
    obs_writer.update_live(caption_text)
    full_logger.add_entry(combined_final, kind="final", lookup_decoded=decoded_lookup, plate_dl_decoded=decoded_plate_dl)

    # Pinned alert area
    if re.search(r"\b(10\s*[- ]?33|11\s*[- ]?99|10-33|11-99)\b", combined_final, re.IGNORECASE):
        alert_html = """<!doctype html><html><head><meta charset='utf-8'>
<style>body{margin:0;background:black;color:#ff3b3b;font-family:Arial,sans-serif;font-weight:800}
.wrap{padding:16px} .big{font-size:48px;letter-spacing:1px}
.small{font-size:18px;color:#fff;margin-top:8px;opacity:.9}
</style></head><body><div class='wrap'>
<div class='big'>ALERT</div>
<div class='small'>{}</div>
</div></body></html>""".format(htmlmod.escape(combined_final))
        atomic_write(ALERTS_HTML, alert_html)


def local_asr_run_forever():
    """Consume mic audio from audio_q, transcribe via faster-whisper, segment by silence, and feed pipeline.

    Key properties:
    - Uses a hop+overlap audio window so we *consume* new audio and do not re-transcribe the same samples.
    - Uses word-overlap delta to avoid repeating text across overlapping windows.
    - Finalizes an utterance only after ASR_SILENCE_SEC of no *new* text.
    """
    print(f"[LocalASR] Loading model: {ASR_MODEL_ID} (device={ASR_DEVICE}, compute={ASR_COMPUTE_TYPE})")
    model = WhisperModel(ASR_MODEL_ID, device=ASR_DEVICE, compute_type=ASR_COMPUTE_TYPE)
    print("[LocalASR] Model loaded.")

    hop_samples = int(ASR_CHUNK_SEC * SAMPLE_RATE)
    overlap_samples = int(ASR_OVERLAP_SEC * SAMPLE_RATE)
    if overlap_samples < 0:
        overlap_samples = 0
    if hop_samples <= 0:
        hop_samples = int(4 * SAMPLE_RATE)

    # Audio consumption buffers
    pending = np.zeros(0, dtype=np.float32)           # unprocessed new audio
    overlap_buf = np.zeros(0, dtype=np.float32)       # last overlap_samples of previously processed audio

    # Rolling text state
    prev_chunk_text = ""
    utterance = ""
    last_speech_time = time.time()

    while True:
        # Pull audio from the queue (non-blocking-ish)
        try:
            pcm16 = audio_q.get(timeout=0.25)
            f32 = _bytes_to_float32_pcm(pcm16)
            pending = np.concatenate([pending, f32])
            # Prevent unbounded growth if something stalls
            max_pending = int((ASR_CHUNK_SEC + ASR_OVERLAP_SEC) * SAMPLE_RATE * 10)
            if len(pending) > max_pending:
                pending = pending[-max_pending:]
        except Exception:
            pass

        now = time.time()

        # Finalize on silence (no new delta for a bit)
        if utterance and (now - last_speech_time) >= ASR_SILENCE_SEC:
            process_utterance_text(utterance, now)
            utterance = ""
            prev_chunk_text = ""
            overlap_buf = np.zeros(0, dtype=np.float32)
            pending = np.zeros(0, dtype=np.float32)
            obs_writer.update_live("")

        # Not enough new audio yet for a hop
        if len(pending) < hop_samples:
            continue

        # Consume exactly one hop, but transcribe with overlap
        hop = pending[:hop_samples]
        pending = pending[hop_samples:]

        if overlap_samples > 0:
            # ensure overlap_buf is the right size
            if len(overlap_buf) > overlap_samples:
                overlap_buf = overlap_buf[-overlap_samples:]
            window = np.concatenate([overlap_buf, hop])
        else:
            window = hop

        # Update overlap for next time
        if overlap_samples > 0:
            overlap_buf = window[-overlap_samples:].copy()

        # Transcribe this window
        try:
            segments, _info = model.transcribe(
                window,
                language="en",
                vad_filter=True,
                beam_size=ASR_BEAM_SIZE,
            )
            chunk_text = " ".join(seg.text.strip() for seg in segments).strip()
        except Exception as e:
            print(f"[LocalASR] Transcribe error: {e}")
            chunk_text = ""

        if not chunk_text:
            continue

        # Compute delta vs previous chunk to reduce duplication
        delta = _word_overlap_delta(prev_chunk_text, chunk_text)
        prev_chunk_text = chunk_text

        if not delta:
            continue

        # Append delta to current utterance
        utterance = (utterance + " " + delta).strip() if utterance else delta
        last_speech_time = now

        # Live preview
        live = utterance
        if len(live) > LIVE_MAX_CHARS:
            live = "…" + live[-LIVE_MAX_CHARS:]
        obs_writer.update_live(live)


# =============================================================================
async def main():
    atomic_write(OBS_LIVE_FILE, "")
    atomic_write(OBS_FINAL_FILE, "")
    if not FULL_LOG_FILE.exists():
        FULL_LOG_FILE.write_text("", encoding="utf-8")
    if not FULL_LOG_HTML_FILE.exists():
        atomic_write(FULL_LOG_HTML_FILE, "<!doctype html><html><body></body></html>")
    if not OBS_LOWER_THIRD_HTML.exists():
        atomic_write(OBS_LOWER_THIRD_HTML, "<!doctype html><html><body></body></html>")
    blocksize = 800
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32", callback=audio_callback, blocksize=blocksize):
        await asyncio.to_thread(local_asr_run_forever)

if __name__ == "__main__":
    asyncio.run(main())