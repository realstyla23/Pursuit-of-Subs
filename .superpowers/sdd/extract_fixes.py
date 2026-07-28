"""Extract high-impact reusable german_fixes from E01 pipeline vs reference comparison."""
import json, sys, os
sys.path.insert(0, os.getcwd())
sys.stdout.reconfigure(encoding="utf-8")

from translator.engine import safe_open_srt

pipe = safe_open_srt("e01/E01_ger.srt")
ref = safe_open_srt("config/e01_reference.srt")

with open("config/german_fixes.json", encoding="utf-8") as f:
    fixes = json.load(f)

existing_finds = {e["find"].strip().lower() for e in fixes}
existing_replaces = {e["replace"].strip() for e in fixes}

new_entries = []
def add_entry(find, replace):
    key = find.strip().lower()
    if key not in existing_finds:
        new_entries.append({"find": find.strip(), "replace": replace.strip()})
        existing_finds.add(key)
        print(f"  NEW: '{find}' -> '{replace}'")
    else:
        print(f"  SKIP (exists): '{find}' -> '{replace}'")

# Show name: Opus translates show title literally
add_entry("[Jadeverfolgung]", "[Pursuit of Jade]")
add_entry("Jadeverfolgung", "Pursuit of Jade")

# Episode marker: Opus translates "Episode" as "Episode" instead of "Folge"
add_entry("[Episode", "[Folge")

# Common NLLB mistranslation: "woman general" -> "Generalin" (already in auto_glossary?)
add_entry("Frau allgemein", "Generalin")
add_entry("junge Frau allgemein", "junge Generalin")
add_entry("einen weiblichen General", "eine Generalin")

# "the Fan couple/pair" -> "the Fans" (character name)
add_entry("Das Fan Paar", "Die Fans")
add_entry("Die Fan Paar", "Die Fans")
add_entry("Das Fan paar", "Die Fans")
add_entry("Fan Paar", "Fan Familie")

# "Episode" -> "Folge" in numeric context
add_entry("Episode 1", "Folge 1")
add_entry("Episode 2", "Folge 2")
add_entry("Episode 3", "Folge 3")

if new_entries:
    fixes.extend(new_entries)
    fixes.sort(key=lambda x: x.get("find", "").lower())
    with open("config/german_fixes.json", "w", encoding="utf-8") as f:
        json.dump(fixes, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"\nAdded {len(new_entries)} new entries to german_fixes.json")
else:
    print("\nNo new entries to add")

    # Now check the merge and run regression
print("\n" + "="*50)
print("Running regression...")
