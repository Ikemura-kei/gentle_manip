"""Broad LVIS category pool (500+ categories) via EXCLUSION filtering of the full 1156-category
LVIS vocabulary, rather than hand-picking buckets one by one (categories.py's approach) --
built under time pressure (user, 2026-09-08: "prioritize the 200 x 20 dataset ... prefilter
maybe 500 candidate categories"). Excludes: people, animals, vehicles, furniture/large
appliances, buildings/architecture/large fixed structures, and cloth/clothing (a different
simulation regime -- cloth-sim, not lumped-MPM-solid -- not "objects" in this pipeline's sense).
Everything else in LVIS survives, on the theory that the geometry filter (real min-width /
2cm-height test) is what actually decides graspability -- this stage only needs to not waste
downloads on things that can NEVER be small graspable objects regardless of rescale.
"""
from __future__ import annotations

import re

# Substrings (case-insensitive) anywhere in the LVIS category name -> excluded.
EXCLUDE_SUBSTRINGS = [
    # people
    "person",
    # animals (common LVIS animal names + a few compounds)
    "alligator", "baboon", "bat_(animal", "bear", "beetle", "bird", "cat", "chicken_(animal",
    "cow", "crab_(animal", "crow", "deer", "dog", "dolphin", "duck", "eagle", "elephant", "elk",
    "ferret", "fish", "flamingo", "fox", "frog", "gazelle", "giraffe", "goat", "goldfish",
    "goose", "gorilla", "grizzly", "gull", "hamster", "heron", "hippopotamus", "hog", "horse",
    "hummingbird", "kitten", "koala", "ladybug", "lamb_(animal", "lion", "lizard", "mammoth",
    "manatee", "monkey", "mouse_(animal", "octopus_(animal", "ostrich", "owl", "panda",
    "parakeet", "parrot", "pelican", "penguin", "pigeon", "polar_bear", "pony", "prawn",
    "pug-dog", "puppy", "rabbit", "ram_(animal", "rat", "rhinoceros", "rodent", "salmon_(fish",
    "seabird", "seahorse", "shark", "sheep", "shepherd_dog", "snake", "spider", "squirrel",
    "starfish", "tiger", "turtle", "vulture", "walrus", "wolf", "zebra", "crawfish", "dalmatian",
    "cub_(animal", "calf", "camel", "cockroach", "puffer_(fish", "puffin", "eel", "beef_(food",
    # vehicles
    "airplane", "ambulance", "army_tank", "bicycle", "boat", "bulldozer", "bus_(vehicle",
    "cab_(taxi", "cabin_car", "camper_(vehicle", "canoe", "car_(automobile", "car_battery",
    "cargo_ship", "convertible_(automobile", "cruise_ship", "dirt_bike", "dinghy",
    "fighter_jet", "fire_engine", "forklift", "freight_car", "garbage_truck", "golfcart",
    "gondola_(boat", "helicopter", "horse_buggy", "horse_carriage", "jeep", "jet_plane",
    "kayak", "limousine", "minivan", "motor_scooter", "motor_vehicle", "motorcycle",
    "passenger_car_(part", "passenger_ship", "pickup_truck", "police_cruiser", "race_car",
    "raft", "railcar", "river_boat", "school_bus", "seaplane", "space_shuttle", "stagecoach",
    "tow_truck", "tractor_(farm", "trailer_truck", "train_(railroad", "tricycle", "truck",
    "unicycle", "wagon", "wagon_wheel", "yacht",
    # furniture / large fixed items
    "armchair", "armoire", "bed", "bench", "bookcase", "bunk_bed", "cabinet", "chair",
    "chaise_longue", "coffee_table", "crib", "cupboard", "deck_chair", "desk", "dining_table",
    "dresser", "file_cabinet", "folding_chair", "footstool", "highchair", "loveseat",
    "ottoman", "recliner", "rocking_chair", "sofa", "stool", "table", "wardrobe", "dollhouse",
    "playpen", "kitchen_table", "music_stool", "sawhorse",
    # large appliances
    "air_conditioner", "dishwasher", "microwave_oven", "oven", "refrigerator", "stove",
    "automatic_washer", "water_heater", "water_cooler", "cooker", "vacuum_cleaner",
    "television_set", "fume_hood", "generator", "grill", "sewing_machine",
    # buildings / architecture / large fixed structures
    "aquarium", "awning", "bathtub", "bridal", "birdbath", "birdcage", "birdhouse", "cabana",
    "chandelier", "clock_tower", "coatrack", "dumpster", "elevator_car", "fireplace",
    "flagpole", "gravestone", "kennel", "lamppost", "mailbox", "manhole", "parking_meter",
    "silo", "stop_sign", "streetlight", "telephone_booth", "telephone_pole", "traffic_light",
    "water_tower", "weathervane", "windmill", "billboard", "blackboard", "bulletin_board",
    "gazebo", "pew_(church", "solar_array", "vending_machine", "sink", "kitchen_sink",
    "washbasin", "toilet", "urinal", "shower_curtain", "shower_head", "radiator", "heater",
    "spice_rack", "step_stool", "stepladder", "ladder", "playground_slide" if False else "slide",
    # clothing / cloth (different sim regime)
    "shirt", "jacket", "dress", "trousers", "pants" if False else "jean", "shoe", "boot",
    "sandal", "slipper", "sock", "sweater", "sweatshirt", "coat", "cape", "cloak", "vest",
    "hat", "cap_(headwear", "beanie", "bonnet", "sombrero", "fedora", "beret", "turban",
    "veil", "scarf", "glove", "mitten", "underwear", "underdrawers", "swimsuit", "bikini" if False else "swimsuit",
    "poncho", "robe", "kimono", "costume", "tux", "suit_(clothing", "bathrobe", "apron",
    "handkerchief", "necktie", "bow-tie", "wetsuit" if False else "wet_suit", "legging",
    "tights_(clothing", "corset", "breechcloth", "sportswear", "polo_shirt", "tank_top",
    "sweat_pants", "sweatband", "wristband", "wristlet", "flannel", "blouse", "blazer",
]
EXCLUDE_RE = re.compile("|".join(re.escape(s) for s in EXCLUDE_SUBSTRINGS), re.IGNORECASE)


def broad_categories(lvis_keys: list[str]) -> list[str]:
    return [c for c in lvis_keys if not EXCLUDE_RE.search(c)]


if __name__ == "__main__":
    import objaverse
    from pathlib import Path
    cache = Path(__file__).resolve().parents[3] / "dataset" / "object_expansion" / "objaverse_cache"
    objaverse.BASE_PATH = str(cache)
    objaverse._VERSIONED_PATH = str(cache / "hf-objaverse-v1")
    lvis = objaverse.load_lvis_annotations()
    cats = broad_categories(sorted(lvis.keys()))
    print(f"{len(cats)} / {len(lvis)} LVIS categories survive the exclusion filter")
