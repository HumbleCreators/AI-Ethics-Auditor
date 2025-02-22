import sqlite3

DATABASE = "reports.db"

def get_connection():
    conn = sqlite3.connect(DATABASE)
    return conn

def create_table():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def store_report(report_data):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO reports (report) VALUES (?)', (str(report_data),))
    conn.commit()
    conn.close()

# Ensure the reports table exists when the module is imported.
create_table()
