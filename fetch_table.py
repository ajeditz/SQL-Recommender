import pandas as pd
import threading
import time
from sqlalchemy import create_engine

# Database Connection Details
DB_HOST = "13.247.208.85"
DB_PORT = "3306"
DB_DATABASE = "vgtechde_gopaddidbv2"
DB_USERNAME = "vgtechde_gopaddiv2"
DB_PASSWORD = "[VZNh-]E%{6q"

DATABASE_URL = f"mysql+pymysql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_DATABASE}"
engine = create_engine(DATABASE_URL)

# Tables to track
TABLES = {
    "post_interest_view": "updated_at",
    "user_followers_following": "updated_at",
    "post_activity": "updated_at"
}

CACHE_FILES = {table: f"{table}_cached.csv" for table in TABLES}
last_cached_times = {table: None for table in TABLES}  # Track last update times
cached_data = {table: None for table in TABLES}  # Store DataFrames in memory
refresh_interval = 3600  # Fetch data every 1 hour

stop_event = threading.Event()  # Event to signal the thread to stop

def fetch_incremental_updates(table_name, timestamp_col):
    """Fetch new or updated rows for a specific table and update cache."""
    global cached_data, last_cached_times

    try:
        # Fetch full data if first time, otherwise fetch incrementally
        if last_cached_times[table_name] is None:
            query = f"SELECT * FROM {table_name};"
        else:
            query = f"SELECT * FROM {table_name} WHERE {timestamp_col} > '{last_cached_times[table_name]}';"

        df = pd.read_sql(query, engine)

        if not df.empty:
            if cached_data[table_name] is not None:
                cached_data[table_name] = pd.concat([cached_data[table_name], df]).drop_duplicates()
            else:
                cached_data[table_name] = df
            
            # Update timestamp
            last_cached_times[table_name] = df[timestamp_col].max()
            cached_data[table_name].to_csv(CACHE_FILES[table_name], index=False)
            print(f"✅ {table_name} cache updated with {len(df)} new rows at {time.ctime()}")

    except Exception as e:
        print(f"❌ Error fetching {table_name}: {e}")

def periodic_refresh():
    """Background thread to refresh all table caches periodically."""
    while not stop_event.is_set():  # Run until stop_event is set
        for table, timestamp_col in TABLES.items():
            fetch_incremental_updates(table, timestamp_col)
        stop_event.wait(refresh_interval)  # Wait for the next update cycle

# Start the cache refresh in a non-daemon thread
thread = threading.Thread(target=periodic_refresh)
thread.start()

print("🚀 Incremental fetching for all tables started in background thread.")

try:
    thread.join()  # Wait for the thread to complete before exiting
except KeyboardInterrupt:
    print("\n🛑 Stopping background thread...")
    stop_event.set()  # Signal the thread to stop
    thread.join()  # Ensure it exits cleanly
    print("✅ Background thread stopped successfully.")
