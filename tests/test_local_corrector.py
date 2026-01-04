from local_corrector import LocalCorrector

c = LocalCorrector()

tests = [
    "43 please send the warrant information Tom my screen please 13 affirm",
    "voice on a quick word",
    "clear31 A valid",
]

for t in tests:
    print("RAW :", t)
    print("OUT :", c.correct(t))
    print()
