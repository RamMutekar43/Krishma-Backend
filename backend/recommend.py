from collections import Counter
from bson.objectid import ObjectId

def hybrid_recommend(mongo, user_id, top_n=5):
    db = mongo.db

    # 1️⃣ Fetch all products
    all_products = list(db.products.find({}))
    if not all_products:
        return []

    product_dict = {str(p["_id"]): p for p in all_products}

    # 2️⃣ Fetch user events
    user_events = list(db.events.find({"userId": user_id}))
    user_history = set(str(e["productId"]) for e in user_events)

    # 3️⃣ Category-based recommendations
    liked_categories = set()
    for pid in user_history:
        product = product_dict.get(pid)
        if product:
            liked_categories.add(product.get("category", "Other"))

    category_candidates = [
        p for p in all_products
        if p.get("category") in liked_categories
    ]

    # 4️⃣ Co-occurrence recommendations
    co_occurrence = Counter()
    all_events = list(db.events.find({}))
    for e in all_events:
        other_user = e["userId"]
        other_pid = str(e["productId"])
        if other_user != user_id and other_pid not in user_history:
            # check overlap
            other_user_events = [str(ev["productId"]) for ev in db.events.find({"userId": other_user})]
            if user_history.intersection(other_user_events):
                co_occurrence[other_pid] += 1

    co_occurrence_candidates = [
        product_dict[pid] for pid, _ in co_occurrence.most_common()
        if pid in product_dict
    ]

    # 5️⃣ Popularity fallback
    popular_counts = Counter()
    for e in all_events:
        pid = str(e["productId"])
        popular_counts[pid] += 1

    popular_candidates = [
        product_dict[pid] for pid, _ in popular_counts.most_common()
        if pid in product_dict
    ]

    # 6️⃣ Merge all candidates without duplicates
    final = []
    seen = set()
    for group in [category_candidates, co_occurrence_candidates, popular_candidates]:
        for p in group:
            pid = str(p["_id"])
            if pid not in seen:
                p["_id"] = pid
                final.append(p)
                seen.add(pid)
            if len(final) >= top_n:
                break
        if len(final) >= top_n:
            break

    # 7️⃣ Return top N recommendations
    return final[:top_n]
