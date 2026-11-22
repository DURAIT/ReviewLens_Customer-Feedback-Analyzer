"""Generate sample_data/reviews_week40.csv (240 valid rows + 3 deliberately bad rows).

Usage:  python scripts/make_sample_data.py
"""
import csv
import random
from datetime import date, timedelta
from pathlib import Path

FOOD = ("Google", "Zomato", "Swiggy")
ANY = FOOD
AMAZON = ("Amazon",)
BRANCHES = ("T Nagar", "Velachery", "OMR", "Adyar", "Anna Nagar")

# (rating, platforms, text)
BASE = [
    # --- examples from the project brief
    (5, FOOD, "Biryani was amazing, packing also neat. Will order again!"),
    (1, FOOD, "Ordered at 8pm, came at 9:10. Food fully cold. Very bad."),
    (2, FOOD, "Saapadu nalla irundhuchu but bill la extra charge pottanga"),
    # --- English
    (5, FOOD, "Great taste and neat packaging, delivered on time."),
    (1, FOOD, "Got food poisoning after eating the chicken rice. Needed to see a doctor."),
    (2, FOOD, "Delivery partner was rude and refused to come upstairs."),
    (2, FOOD, "Charged extra packing and service fees that were not shown on the menu."),
    (2, FOOD, "Portion size is too small for the price."),
    (3, FOOD, "Food was decent, nothing special."),
    (2, ("Google",), "The waiter ignored us for 20 minutes. Poor service."),
    (5, ("Google",), "Love the new menu, the paneer wrap is amazing and the staff were lovely."),
    (3, FOOD, "Delivery was on time but the food was lukewarm."),
    (1, FOOD, "App kept crashing at payment and I was charged twice. Want my money back."),
    (4, FOOD, "Tasty food and quick delivery, just a bit oily."),
    (1, FOOD, "Found a hair in my biryani. Disgusting."),
    (1, AMAZON, "Wrong item delivered and the refund is still pending after 10 days."),
    (1, AMAZON, "Packaging was torn and the product inside was damaged."),
    (5, AMAZON, "Delivered two days early and well packed. Very happy."),
    (2, AMAZON, "Product quality is not as good as shown in the listing photos."),
    (4, AMAZON, "Good value for the price, works as described."),
    # --- Tamil script
    (5, FOOD, "சாப்பாடு ரொம்ப நல்லா இருந்தது, டெலிவரியும் சீக்கிரம் வந்துச்சு."),
    (1, FOOD, "ஆர்டர் ஒரு மணி நேரம் லேட்டா வந்துச்சு, சாப்பாடு முழுசா குளிர்ந்து போச்சு."),
    (1, FOOD, "பிரியாணியில் முடி இருந்தது. மிகவும் மோசம்."),
    (5, FOOD, "டெலிவரி பையன் ரொம்ப மரியாதையா நடந்துக்கிட்டார்."),
    (2, FOOD, "பில்லில் கூடுதல் கட்டணம் போட்டிருக்காங்க, பணம் திருப்பி தாங்க."),
    (1, FOOD, "சாப்பாடு சாப்பிட்ட பிறகு வயிற்று வலி வந்தது."),
    (2, FOOD, "பேக்கிங் சரியில்ல, சாம்பார் கொட்டிடுச்சு."),
    (2, ("Google",), "அளவு கம்மியா இருக்கு, விலைக்கு ஏத்த மாதிரி இல்ல."),
    (5, AMAZON, "பொருள் நல்ல தரம், சீக்கிரமாவே வந்துடுச்சு."),
    # --- Tanglish
    (1, FOOD, "Delivery romba late, food cold ah vandhuchu. Waste of money"),
    (1, FOOD, "Rider romba rude ah pesinaar, phone eduthu thittinaar"),
    (5, FOOD, "Food super ah irundhuchu, quantity um nalla irukku"),
    (1, FOOD, "App la payment fail aachu but paisa cut aayiduchu, refund illa"),
    (1, FOOD, "Wrong item anuppitaanga, biryani order pannen but pizza vandhadhu"),
    (2, FOOD, "Packing la oil leak aayiduchu, bag full ah ketta vaasam"),
    (2, FOOD, "Quantity romba kammi, price ku worth illa"),
    (4, ("Google",), "Staff friendly ah irundhanga, service fast"),
    (1, FOOD, "Chicken half cooked ah irundhuchu, odane vomit pannen"),
    (3, FOOD, "Okay okay dhaan, average taste"),
    (2, FOOD, "App crash aaguthu, order place panna mudiyala"),
    (4, AMAZON, "Product nalla irukku, delivery konjam late aana ok"),
]

OUT = Path(__file__).resolve().parent.parent / "sample_data" / "reviews_week40.csv"


def main(n: int = 240, seed: int = 42) -> None:
    rng = random.Random(seed)
    start = date(2026, 9, 22)  # two weeks, so week-over-week numbers and trends have data
    rows = []
    for i in range(n):
        rating, platforms, text = BASE[i] if i < len(BASE) else rng.choice(BASE)
        platform = rng.choice(platforms)
        rows.append([
            f"R-{1001 + i}",
            (start + timedelta(days=rng.randint(0, 13))).isoformat(),
            platform,
            "Online Store" if platform == "Amazon" else rng.choice(BRANCHES),
            rating,
            text,
        ])
    # deliberately bad rows to show validation
    rows += [
        [f"R-{1001 + n}", "2026-10-02", "Zomato", "OMR", 3, ""],
        [f"R-{1002 + n}", "2026-10-02", "Swiggy", "Adyar", 4, "   "],
        ["R-1001", "2026-10-03", "Google", "T Nagar", 5, "Duplicate id, should be dropped."],
    ]
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8-sig") as f:  # BOM so Excel shows Tamil
        w = csv.writer(f)
        w.writerow(["review_id", "date", "platform", "branch", "rating", "review_text"])
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
