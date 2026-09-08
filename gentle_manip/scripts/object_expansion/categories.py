"""Curated Objaverse-LVIS category shortlist for the 200-object expansion.

Picked from the full 1156-category LVIS vocabulary (`objaverse.load_lvis_annotations()`),
excluding: people/animals (ethical + rigging complexity, no manipulation value), vehicles/
furniture/appliances (way over the 79mm grasp cap even rescaled — rescaling a car preserves
its car-ness, not a graspable object), building/architecture categories, and every category
that already duplicates a registered food family (tofu, mushroom, cherry_tomato, tomato,
banana, pasta_bundle, raspberry, strawberry).

Grouped by the SHAPE BUCKET the user asked to diversify toward, purely for reporting/
triage — the geometry filter (`geom_filter.py`) is what actually decides graspability, this
grouping just makes sure the candidate pool doesn't collapse onto one shape family.
"""

SHAPE_BUCKETS: dict[str, list[str]] = {
    "compact_blobby": [
        "apple", "avocado", "orange_(fruit)", "lemon", "lime", "peach", "pear", "potato",
        "onion", "egg", "kiwi_fruit", "fig_(fruit)", "artichoke", "coconut", "gemstone",
        "doorknob", "pumpkin", "garlic", "clementine", "date_(fruit)",
        # "tomato" is ALREADY a registered food family (tomato/tomato1/3/4/5) -- included here
        # anyway (2026-09-08, user) to pull EXTRA shape variants from a different online source
        # (Objaverse) than the existing TripoSG photo-reconstructions, registered as tomato6+.
        "tomato",
    ],
    "elongated_rope_stick": [
        "carrot", "cucumber", "zucchini", "baguet", "drumstick", "chopstick", "pencil", "pen",
        "crayon", "paintbrush", "toothbrush", "screwdriver", "wooden_spoon", "ladle", "spoon",
        "fork", "knife", "spatula", "rolling_pin", "tape_measure", "walking_stick", "celery",
        "asparagus", "green_bean", "wrench", "hammer", "corkscrew", "key", "thermometer",
        "nailfile", "hairbrush", "clothespin", "hook", "cigar_box", "candle",
    ],
    "tall_narrow": [
        "water_bottle", "beer_bottle", "wine_bottle", "saltshaker", "pepper_mill", "hourglass",
        "shot_glass", "teacup", "trophy_cup", "thermos_bottle", "lip_balm", "perfume",
        "spice_rack", "flute_glass", "vase",
    ],
    "thin_shell_hollow": [
        "wineglass", "mug", "cup", "teapot", "bowl", "vase", "flowerpot", "coffeepot",
        "creamer" if False else "cream_pitcher", "sugar_bowl", "gravy_boat", "measuring_cup",
        "funnel", "colander",
    ],
    "flat_thin": [
        "coaster", "cork_(bottle_plug)", "bottle_cap", "poker_chip", "penny_(coin)", "coin",
        "keycard", "cufflink", "button", "bookmark", "domino" if False else "die",
    ],
    "complex_concave": [
        "scissors", "pliers", "tongs", "padlock", "wrench", "hammer", "corkscrew", "measuring_cup",
        "gravy_boat", "eggbeater", "nutcracker", "can_opener", "peeler_(tool_for_fruit_and_vegetables)",
        "bottle_opener", "clothespin", "stapler_(stapling_machine)",
    ],
    "small_tools_office": [
        "stapler_(stapling_machine)", "puncher", "tape_measure", "remote_control", "eraser",
        "pencil_sharpener", "thumbtack", "paperweight", "compass", "calculator",
    ],
    "toys_misc": [
        "die", "ball", "tennis_ball", "ping-pong_ball", "baseball", "frisbee", "rubber_band",
        "figurine", "toy",
    ],
    "kitchen_containers": [
        "jar", "beer_can", "can", "canister", "saltshaker", "pepper_mill", "egg", "gelatin",
    ],
    "bath_personal": [
        "soap", "toothpaste", "toothbrush", "sponge", "shampoo", "lip_balm", "hairbrush",
    ],
}


def all_categories() -> list[str]:
    seen, out = set(), []
    for cats in SHAPE_BUCKETS.values():
        for c in cats:
            if c not in seen:
                seen.add(c)
                out.append(c)
    return out


def bucket_of(category: str) -> str:
    for bucket, cats in SHAPE_BUCKETS.items():
        if category in cats:
            return bucket
    return "unknown"


if __name__ == "__main__":
    cats = all_categories()
    print(f"{len(cats)} unique categories across {len(SHAPE_BUCKETS)} buckets")
    for b, cs in SHAPE_BUCKETS.items():
        print(f"  {b}: {len(cs)}")
