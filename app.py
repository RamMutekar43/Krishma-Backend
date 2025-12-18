from flask import Flask, request, jsonify
from flask_pymongo import PyMongo
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from bson.objectid import ObjectId
import os
import datetime
import random
import requests  # Added for Telegram

# ---------- ADDED IMPORTS FOR FORECASTING ----------
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
# ---------------------------------------------------

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# MongoDB Configuration
app.config["MONGO_URI"] = os.getenv("MONGO_URI", "mongodb://localhost:27017/krishma_dairy_db")
mongo = PyMongo(app)

# Telegram Config
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8351117643:AAFTll2dfBE-Hod2AlS6bRCCLQE2PLGCvyc")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "6148242881")

def send_telegram_alert(product_name_or_message, stock=None):
    """
    Flexible Telegram notifier:
      - If `stock` is provided: treat `product_name_or_message` as product name and send an inventory alert.
      - If `stock` is None: treat `product_name_or_message` as a full (possibly HTML) message to send as-is.
    """
    if stock is None:
        # Single-argument usage: send preformatted message (HTML allowed)
        message = product_name_or_message
        payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    else:
        # Two-argument usage: inventory alert
        message = (
            f"⚠ Inventory Alert!\n\n"
            f"Product: {product_name_or_message}\n"
            f"Current Stock: {stock}\n"
            f"Threshold: 10\n\n"
            f"Please restock this product immediately!"
        )
        payload = {"chat_id": CHAT_ID, "text": message}

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        resp = requests.post(url, data=payload, timeout=5)
        resp.raise_for_status()
        print(f"Alert sent (len={len(payload.get('text',''))})")
    except Exception as e:
        # keep failures non-fatal — just log
        print(f"Error sending Telegram alert: {e}")

# --- Helper Functions ---
def create_collections_if_not_exist():
    try:
        for col in ["customers", "admins", "products", "orders", "reviews", "events", "sales"]:
            if col not in mongo.db.list_collection_names():
                mongo.db.create_collection(col)
                print(f"Created '{col}' collection.")
    except Exception as e:
        print(f"Error creating collections: {e}")

def create_default_admin():
    admin_username = "krishma"
    if mongo.db.admins.find_one({"username": admin_username}) is None:
        hashed_password = generate_password_hash("krishma@123")
        admin_data = {
            "name": "Krishma Admin",
            "username": admin_username,
            "email": "admin@krishmadairy.com",
            "password": hashed_password,
            "phone": "9876543210",
            "address": "Krishma Dairy HQ, Mumbai",
            "bio": "Lead administrator for Krishma Dairy.",
            "profileImage": "https://example.com/admin-profile.jpg"
        }
        mongo.db.admins.insert_one(admin_data)
        print("Default admin user 'krishma' created.")

# ----------------- Deduct stock safely -----------------
def deduct_product_stock(order_items):
    """Deduct stock for each product atomically and send alert if below threshold"""
    for item in order_items:
        product_id = item.get("productId")
        quantity = int(item.get("quantity", 0))
        if not product_id or quantity <= 0:
            continue
        product = mongo.db.products.find_one({"_id": ObjectId(product_id)})
        if product:
            new_stock = product.get("stock", 0) - quantity
            mongo.db.products.update_one(
                {"_id": ObjectId(product_id)},
                {"$inc": {"stock": -quantity}}
            )
            # Check threshold
            if new_stock < 10:
                send_telegram_alert(product.get("name", "Unknown Product"), new_stock)

# Startup
with app.app_context():
    create_collections_if_not_exist()
    create_default_admin()

@app.route('/')
def home():
    return "Krishma Dairy API is running!"

# ---------------- Customer Endpoints ----------------
@app.route('/api/customer/signup', methods=['POST'])
def customer_signup():
    data = request.json
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')
    if not all([name, email, password]):
        return jsonify({"msg": "Missing fields"}), 400
    if mongo.db.customers.find_one({"email": email}):
        return jsonify({"msg": "Email already registered"}), 409
    hashed_password = generate_password_hash(password)
    customer_data = {
        "name": name,
        "email": email,
        "password": hashed_password,
        "mobile": data.get('mobile'),
        "address": data.get('address'),
        "profilePicture": data.get('profilePicture', ''),
        "joinDate": datetime.datetime.now().isoformat().split('T')[0],
        "totalOrders": 0,
        "totalSpent": 0,
        "status": "active"
    }
    mongo.db.customers.insert_one(customer_data)
    return jsonify({"msg": "Customer registered successfully"}), 201

@app.route('/api/customer/login', methods=['POST'])
def customer_login():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    customer = mongo.db.customers.find_one({"email": email})
    if customer and check_password_hash(customer['password'], password):
        return jsonify({"msg": "Login successful", "user": {"email": email}}), 200
    return jsonify({"msg": "Invalid email or password"}), 401

@app.route('/api/customer/profile/<email>', methods=['GET', 'PUT'])
def customer_profile(email):
    if request.method == 'GET':
        customer = mongo.db.customers.find_one({"email": {"$regex": f"^{email}$", "$options": "i"}}, {"password": 0})
        if customer:
            customer['_id'] = str(customer['_id'])
            return jsonify(customer), 200
        return jsonify({"msg": "Customer not found"}), 404
    elif request.method == 'PUT':
        data = request.json
        if not data:
            return jsonify({"msg": "No update data provided"}), 400
        if "email" in data:
            del data["email"]
        result = mongo.db.customers.find_one_and_update(
            {"email": {"$regex": f"^{email}$", "$options": "i"}},
            {"$set": data},
            return_document=True
        )
        if result:
            result["_id"] = str(result["_id"])
            if "password" in result:
                del result["password"]
            return jsonify({"msg": "Profile updated successfully", "customer": result}), 200
        return jsonify({"msg": "Customer not found"}), 404

@app.route('/api/customer/orders/<email>', methods=['GET'])
def customer_orders(email):
    orders = mongo.db.orders.find({"customer.email": email})
    orders_list = list(orders)
    for order in orders_list:
        order['_id'] = str(order['_id'])
    return jsonify(orders_list), 200

@app.route('/api/customer/orders', methods=['POST'])
def place_order():
    data = request.json
    customer_email = data.get('customerEmail')
    items = data.get('items')
    total = data.get('total')
    order_id = data.get('id')
    if not all([customer_email, items, total, order_id]):
        return jsonify({"msg": "Missing order data"}), 400
    customer = mongo.db.customers.find_one({"email": customer_email})
    if not customer:
        return jsonify({"msg": "Customer not found"}), 404
    order_details = {
        "id": order_id,
        "customer": {
            "name": customer.get('name'),
            "email": customer.get('email'),
            "phone": customer.get('mobile'),
            "address": customer.get('address'),
        },
        "items": items,
        "total": total,
        "status": "pending",
        "orderDate": data.get('orderDate'),
        "estimatedDelivery": (datetime.datetime.now() + datetime.timedelta(days=2)).isoformat().split('T')[0]
    }
    mongo.db.orders.insert_one(order_details)
    mongo.db.customers.update_one(
        {"email": customer_email},
        {"$inc": {"totalOrders": 1, "totalSpent": total}}
    )
    return jsonify({"msg": "Order placed successfully"}), 201

@app.route('/api/customer/reviews', methods=['POST'])
def submit_review():
    data = request.json
    mongo.db.reviews.insert_one(data)
    return jsonify({"msg": "Review submitted successfully"}), 201

# ---------------- Admin Endpoints ----------------
@app.route('/api/admin/login', methods=['POST'])
def admin_login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    admin = mongo.db.admins.find_one({"username": username})
    if admin and check_password_hash(admin['password'], password):
        return jsonify({"msg": "Login successful"}), 200
    return jsonify({"msg": "Invalid credentials"}), 401

@app.route('/api/admin/profile/<username>', methods=['GET', 'PUT'])
def admin_profile(username):
    if request.method == 'GET':
        admin = mongo.db.admins.find_one({"username": username}, {"password": 0})
        if admin:
            admin['_id'] = str(admin['_id'])
            return jsonify(admin), 200
        return jsonify({"msg": "Admin not found"}), 404
    elif request.method == 'PUT':
        data = request.json
        if "username" in data:
            del data["username"]
        if "email" in data:
            del data["email"]
        result = mongo.db.admins.find_one_and_update(
            {"username": username},
            {"$set": data},
            return_document=True
        )
        if result:
            result["_id"] = str(result["_id"])
            if "password" in result:
                del result["password"]
            return jsonify({"msg": "Profile updated successfully", "admin": result}), 200
        return jsonify({"msg": "Admin not found"}), 404

@app.route('/api/admin/change-password/<username>', methods=['PUT'])
def change_admin_password(username):
    data = request.json
    current_password = data.get('currentPassword')
    new_password = data.get('newPassword')
    admin = mongo.db.admins.find_one({"username": username})
    if admin and check_password_hash(admin['password'], current_password):
        hashed_password = generate_password_hash(new_password)
        mongo.db.admins.update_one(
            {"username": username},
            {"$set": {"password": hashed_password}}
        )
        return jsonify({"msg": "Password changed successfully"}), 200
    return jsonify({"msg": "Invalid current password"}), 401

@app.route('/api/admin/customers', methods=['GET'])
def get_customers():
    customers = mongo.db.customers.find({}, {"password": 0})
    customers_list = list(customers)
    for customer in customers_list:
        customer['_id'] = str(customer['_id'])
    return jsonify(customers_list), 200

@app.route('/api/admin/products', methods=['GET', 'POST'])
def handle_products():
    if request.method == 'GET':
        products = mongo.db.products.find()
        products_list = list(products)
        for product in products_list:
            product['_id'] = str(product['_id'])
        return jsonify(products_list), 200
    elif request.method == 'POST':
        data = request.json
        mongo.db.products.insert_one(data)
        return jsonify({"msg": "Product added successfully"}), 201

@app.route('/api/admin/products/<product_id>', methods=['PUT', 'DELETE'])
def update_delete_product(product_id):
    if request.method == 'PUT':
        data = request.json
        result = mongo.db.products.update_one(
            {"_id": ObjectId(product_id)},
            {"$set": data}
        )
        if result.modified_count > 0:
            # Check stock threshold
            if "stock" in data:
                try:
                    stock_value = int(data["stock"])
                    if stock_value < 10:  # Threshold
                        product = mongo.db.products.find_one({"_id": ObjectId(product_id)})
                        product_name = product.get("name", "Unknown Product")
                        send_telegram_alert(product_name, stock_value)
                except Exception as e:
                    print(f"Error checking stock threshold: {e}")
            return jsonify({"msg": "Product updated successfully"}), 200
        return jsonify({"msg": "No changes made"}), 200
    elif request.method == 'DELETE':
        result = mongo.db.products.delete_one({"_id": ObjectId(product_id)})
        if result.deleted_count > 0:
            return jsonify({"msg": "Product deleted successfully"}), 200
        return jsonify({"msg": "Product not found"}), 404

# ---------------- Orders ----------------
@app.route('/api/admin/orders', methods=['GET'])
def get_orders():
    orders = mongo.db.orders.find()
    orders_list = list(orders)
    for order in orders_list:
        order['_id'] = str(order['_id'])
    return jsonify(orders_list), 200
@app.route('/api/admin/orders/<order_id>', methods=['PUT'])
def update_order(order_id):
    data = request.json or {}
    new_status = data.get("status")

    # Try to find order by Mongo _id first, if that fails fall back to custom 'id' field
    filter_query = None
    order = None
    try:
        filter_query = {"_id": ObjectId(order_id)}
        order = mongo.db.orders.find_one(filter_query)
    except Exception:
        # Not a valid ObjectId, try by custom order id field
        filter_query = {"id": order_id}
        order = mongo.db.orders.find_one(filter_query)

    if not order:
        return jsonify({"msg": "Order not found"}), 404

    stock_updated = order.get("stockUpdated", False)

    result = mongo.db.orders.update_one(
        filter_query,
        {"$set": data}
    )

    # 🟢 Deduct stock when SHIPPED (only once)
    if new_status == "shipped" and not stock_updated:
        try:
            for item in order.get("items", []):
                product = mongo.db.products.find_one({"name": item.get("name")})
                if product:
                    new_stock = max(0, int(product.get("stock", 0)) - int(item.get("quantity", 0)))
                    mongo.db.products.update_one(
                        {"_id": product["_id"]},
                        {"$set": {"stock": new_stock}}
                    )
                    if new_stock < 10:
                        send_telegram_alert(product.get("name", "Unknown Product"), new_stock)
            mongo.db.orders.update_one(
                filter_query,
                {"$set": {"stockUpdated": True}}
            )
            message = (
                f"📦 <b>Order Shipped!</b>\n"
                f"Order ID: <code>{order.get('id') or str(order.get('_id'))}</code>\n"
                f"Customer: {order['customer']['name']}\n"
                f"Total: ₹{order['total']}\n"
                f"Status: SHIPPED 🚚"
            )
            send_telegram_alert(message)
        except Exception as e:
            return jsonify({"msg": f"Error deducting stock: {str(e)}"}), 500

    # 🔴 Reverse stock if status changed from SHIPPED → CANCELLED
    elif new_status == "cancelled" and stock_updated:
        try:
            for item in order.get("items", []):
                product = mongo.db.products.find_one({"name": item.get("name")})
                if product:
                    new_stock = int(product.get("stock", 0)) + int(item.get("quantity", 0))
                    mongo.db.products.update_one(
                        {"_id": product["_id"]},
                        {"$set": {"stock": new_stock}}
                    )
            mongo.db.orders.update_one(
                filter_query,
                {"$set": {"stockUpdated": False}}
            )
            message = (
                f"❌ <b>Order Cancelled!</b>\n"
                f"Order ID: <code>{order.get('id') or str(order.get('_id'))}</code>\n"
                f"Customer: {order['customer']['name']}\n"
                f"Stock restored to inventory ✅"
            )
            send_telegram_alert(message)
        except Exception as e:
            return jsonify({"msg": f"Error restoring stock: {str(e)}"}), 500

    # 📨 Generic Telegram alert for other statuses
    elif new_status and new_status not in ["shipped", "cancelled"]:
        message = (
            f"🔔 <b>Order Status Updated</b>\n"
            f"Order ID: <code>{order.get('id') or str(order.get('_id'))}</code>\n"
            f"New Status: {new_status.upper()}"
        )
        send_telegram_alert(message)

    # ---------------automatic sales collection for all statuses-------
    try:
        record_order_sales(order, new_status)
    except Exception as e:
        return jsonify({"msg": f"Error recording sales: {str(e)}"}), 500

    if result.modified_count > 0:
        return jsonify({"msg": "Order updated successfully"}), 200
    return jsonify({"msg": "No changes made"}), 200


# Automatic sales collection function (all statuses)
def record_order_sales(order, status):
    for item in order.get("items", []):
        product_name = item.get("name")
        quantity = int(item.get("quantity", 0))
        sale_date = datetime.datetime.now().strftime("%Y-%m-%d")

        # Increment for shipped/pending/delivered, decrement for cancelled
        if status in ["pending", "shipped", "delivered"]:
            change = quantity
        elif status == "cancelled":
            change = -quantity
        else:
            change = 0

        if change != 0:
            mongo.db.sales.update_one(
                {"name": product_name, "sale_date": sale_date},
                {"$inc": {"quantity": change}},
                upsert=True
            )


# ---------------- Reviews ----------------
@app.route('/api/admin/reviews', methods=['GET'])
def get_reviews():
    reviews = mongo.db.reviews.find()
    reviews_list = list(reviews)
    for review in reviews_list:
        review['_id'] = str(review['_id'])
    return jsonify(reviews_list), 200

@app.route('/api/admin/reviews/<review_id>', methods=['PUT'])
def update_review(review_id):
    data = request.json
    result = mongo.db.reviews.update_one(
        {"_id": ObjectId(review_id)},
        {"$set": data}
    )
    if result.modified_count > 0:
        return jsonify({"msg": "Review updated successfully"}), 200
    return jsonify({"msg": "No changes made"}), 200

# ---------------- Dashboard Stats ----------------
@app.route('/api/admin/dashboard', methods=['GET'])
def get_dashboard_stats():
    # --- Existing stats ---
    total_sales_cursor = mongo.db.orders.aggregate([
        {"$match": {"status": "delivered"}},
        {"$group": {"_id": None, "total": {"$sum": "$total"}}}
    ])
    total_sales = next(total_sales_cursor, {"total": 0})["total"]
    total_orders = mongo.db.orders.count_documents({})
    active_customers = mongo.db.customers.count_documents({"status": "active"})
    total_products = mongo.db.products.count_documents({})

    # --- Generate today's sales ---
    try:
        generate_daily_sales()
    except Exception as e:
        print(f"generate_daily_sales() error: {e}")

    # --- Prepare forecasts ---
    df = get_sales_df()
    forecasts = []
    if not df.empty:
        for product in df["name"].unique():
            forecast_df = forecast_sales_lstm(df, product)
            if not forecast_df.empty:
                total_forecast = float(forecast_df["forecast"].sum())
                forecasts.append({
                    "product": product,
                    "predicted_sales_next_7_days": round(total_forecast, 2)
                })
        forecasts.sort(key=lambda x: x["predicted_sales_next_7_days"], reverse=True)

    top5_forecast = forecasts[:5]  # top 5 predicted products

    stats = {
        "totalSales": total_sales,
        "ordersProcessed": total_orders,
        "activeCustomers": active_customers,
        "totalProducts": total_products,
        "top5ForecastedProducts": top5_forecast
    }

    return jsonify(stats), 200


# ---------------- Event Tracking ----------------
@app.route("/api/events", methods=["POST"])
def add_event():
    data = request.get_json()
    userId = data.get("userId", "guest")
    productId = data.get("productId")
    eventType = data.get("eventType")
    value = data.get("value", 1)
    if not productId or not eventType:
        return jsonify({"message": "productId and eventType are required"}), 400
    score_mapping = {"view": 1, "add_to_cart": 2, "purchase": 3}
    score = score_mapping.get(eventType, 1) * value
    event = {
        "userId": userId,
        "productId": productId,
        "eventType": eventType,
        "value": value,
        "score": score,
        "timestamp": datetime.datetime.utcnow()
    }
    mongo.db.events.insert_one(event)
    return jsonify({"message": "Event stored successfully!", "score": score}), 201

# ---------------- get  Sale ----------------
@app.route('/api/sales-data', methods=['GET'])
def get_sales_data():
    sales = list(mongo.db.sales.find())
    data = []
    for s in sales:
        data.append({
            "name": s.get("name", "Unknown"),
            "quantity": int(s.get("quantity", 0)),
            "sale_date": s.get("sale_date", "N/A")
        })
    return jsonify(data)

#record sale
@app.route('/api/record-sale', methods=['POST'])
def record_sale():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    product_name = data.get('name')
    try:
        quantity = int(data.get('quantity', 0))
    except ValueError:
        return jsonify({"error": "Quantity must be an integer"}), 400

    sale_date = data.get('sale_date', str(datetime.datetime.now().date()))

    mongo.db.sales.insert_one({
        "name": product_name,
        "quantity": quantity,
        "sale_date": sale_date
    })

    return jsonify({"message": "Sale recorded successfully!"}), 201



# ---------------- Recommendation ----------------
@app.route('/api/recommend/<user_id>', methods=['GET'])
def recommend_products(user_id):
    db_local = mongo.db
    user_events = list(db_local.events.find({"userId": user_id}).sort("timestamp", -1))

    # If no events, return trending
    if not user_events:
        trending = list(db_local.products.find().sort("totalSold", -1).limit(5))
        for t in trending:
            t["_id"] = str(t["_id"])
        return jsonify(trending)

    last_event = user_events[0]
    product_id = last_event.get("productId")

    # Handle string/ObjectId
    try:
        product_object_id = ObjectId(product_id)
    except:
        product_object_id = product_id

    base_product = db_local.products.find_one({"_id": product_object_id})
    if not base_product:
        return jsonify([])

    base_category = base_product.get("category", "")
    base_name = base_product.get("name", "").split()[0]

    similar = list(db_local.products.find({
        "$or": [
            {"category": base_category},
            {"name": {"$regex": base_name, "$options": "i"}}
        ],
        "_id": {"$ne": product_object_id}
    }).limit(6))

    # Fallback if less than 3 similar
    if len(similar) < 3:
        extra = list(db_local.products.find(
            {"category": base_category, "_id": {"$ne": product_object_id}}
        ).sort("totalSold", -1).limit(3))
        similar.extend(extra)

    final_recommend = []
    seen = set()
    for p in similar:
        pid = str(p["_id"])
        if pid not in seen:
            p["_id"] = pid
            final_recommend.append(p)
            seen.add(pid)

    if not final_recommend:
        trending = list(db_local.products.find().sort("totalSold", -1).limit(5))
        for t in trending:
            t["_id"] = str(t["_id"])
        return jsonify(trending)

    random.shuffle(final_recommend)
    return jsonify(final_recommend[:5])

# ------------------- SALES FORECASTING MODULE (FIXED) -------------------
# Use existing mongo.db collections (no new client)
sales_collection = mongo.db.sales
products_collection = mongo.db.products


# ------------------- GET SALES DATA -------------------
def get_sales_df():
    """
    Fetch sales data and normalize different document shapes into columns:
    name, sale_date, quantity
    """
    raw = list(sales_collection.find({}, {"_id": 0}))
    normalized = []
    for doc in raw:
        # determine name
        name = doc.get("name") or doc.get("productName")
        # fallback: if productId exists, try to fetch product name
        if not name and doc.get("productId"):
            try:
                prod = mongo.db.products.find_one({"_id": ObjectId(doc.get("productId"))}, {"name": 1})
                name = prod.get("name") if prod else None
            except Exception:
                name = None

        # determine sale_date
        sale_date = doc.get("sale_date") or doc.get("date")
        # normalize datetime types and strings
        if isinstance(sale_date, datetime.datetime):
            sale_date_norm = sale_date.strftime("%Y-%m-%d")
        else:
            sale_date_norm = str(sale_date) if sale_date is not None else None

        # determine quantity
        quantity = None
        if "quantity" in doc:
            quantity = doc.get("quantity")
        elif "quantitySold" in doc:
            quantity = doc.get("quantitySold")
        elif "sold" in doc:
            quantity = doc.get("sold")

        # only include valid rows
        if name and sale_date_norm and quantity is not None:
            try:
                qty_int = int(quantity)
                normalized.append({
                    "name": name,
                    "sale_date": sale_date_norm,
                    "quantity": qty_int
                })
            except Exception:
                # skip non-integer quantities
                continue

    df = pd.DataFrame(normalized)
    if df.empty:
        return pd.DataFrame(columns=["name", "sale_date", "quantity"])
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    df.sort_values("sale_date", inplace=True)
    return df

# ------------------- LSTM FORECAST FUNCTION -------------------
def forecast_sales_lstm(df, product_name):
    """
    Trains a lightweight LSTM per product if enough history exists.
    Returns a dataframe with 7 future dates and forecast values.
    """
    try:
        product_data = df[df["name"] == product_name]
        if product_data.empty:
            return pd.DataFrame(columns=["date", "forecast"])

        daily_sales = product_data.groupby("sale_date")["quantity"].sum().reset_index()

        # need at least 8 days to build one training sample with 7-day window
        if len(daily_sales) < 8:
            return pd.DataFrame(columns=["date", "forecast"])

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(daily_sales["quantity"].values.reshape(-1, 1))

        X, y = [], []
        for i in range(7, len(scaled_data)):
            X.append(scaled_data[i - 7:i, 0])
            y.append(scaled_data[i, 0])

        if len(X) == 0:
            return pd.DataFrame(columns=["date", "forecast"])

        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # Build small model to keep resources reasonable
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(LSTM(64))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mse")

        # Train (quiet)
        model.fit(X, y, epochs=10, batch_size=1, verbose=0)

        # Forecast next 7 days using last 7 scaled values
        last_7_days = scaled_data[-7:]
        forecast = []
        current_input = last_7_days.reshape(1, 7, 1)

        for _ in range(7):
            pred = model.predict(current_input, verbose=0)
            forecast.append(pred[0][0])
            current_input = np.append(current_input[:, 1:, :], [[pred]], axis=1)

        forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
        forecast_dates = [daily_sales["sale_date"].max() + timedelta(days=i + 1) for i in range(7)]

        return pd.DataFrame({"date": forecast_dates, "forecast": forecast.flatten()})
    except Exception as e:
        # don't crash the app if forecasting fails for any product
        print(f"Forecasting error for product '{product_name}': {e}")
        return pd.DataFrame(columns=["date", "forecast"])

# ------------------- API ENDPOINT -------------------
@app.route("/api/forecast-sales", methods=["GET"])
def forecast_sales():
    # Fetch all sales data (no date filter)
    sales_cursor = mongo.db.sales.find({})
    sales_list = list(sales_cursor)
    
    if not sales_list:
        return jsonify({"message": "No sales data available", "forecasts": []}), 200

    # Convert to DataFrame
    df = pd.DataFrame(sales_list)
    if "quantity" not in df.columns or "name" not in df.columns or "sale_date" not in df.columns:
        return jsonify({"message": "Sales data missing required fields", "forecasts": []}), 400

    df["sale_date"] = pd.to_datetime(df["sale_date"])
    df = df.sort_values(["name", "sale_date"])
    
    product_names = df["name"].unique()
    forecasts = []

    for product in product_names:
        product_df = df[df["name"] == product].copy()
        product_df.set_index("sale_date", inplace=True)
        product_df = product_df.resample("D").sum().fillna(0)

        # 7-day moving average forecast
        forecast = product_df["quantity"].rolling(window=7, min_periods=1).mean().iloc[-1]

        forecasts.append({
            "product": product,
            "predicted_sales_next_7_days": round(forecast * 7, 2)  # total 7-day forecast
        })

    # Sort top 5 products
    forecasts.sort(key=lambda x: x["predicted_sales_next_7_days"], reverse=True)
    top_forecasts = forecasts[:5]

    return jsonify({
        "message": "7-day sales forecast generated successfully",
        "forecasts": top_forecasts
    }), 200



# ---------- Fix: Correct entry point ----------
if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True,port=5000)
