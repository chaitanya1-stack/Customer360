"""Download, audit, and ingest the original UCI workbook without loading it all in RAM."""
from __future__ import annotations
import hashlib
import json
import math
import re
import sqlite3
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path
import openpyxl

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
SOURCE = "https://archive.ics.uci.edu/dataset/502/online+retail+ii"
SCHEMA = """
CREATE TABLE lines (
 id INTEGER PRIMARY KEY, invoice TEXT, stock_code TEXT, description TEXT,
 quantity REAL, invoice_date TEXT, unit_price REAL, customer_id TEXT, country TEXT,
 source_sheet TEXT, row_hash TEXT UNIQUE, reason TEXT, is_return INTEGER,
 value REAL
);
CREATE INDEX line_date ON lines(invoice_date);
CREATE INDEX line_customer ON lines(customer_id);
"""

def download(root: Path) -> Path:
    target = root / "data/raw/online_retail_II.xlsx"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return target
    archive = target.parent / "online_retail_ii.zip"
    if not archive.exists():
        part = archive.with_suffix(".part")
        try:
            req = urllib.request.Request(URL, headers={"User-Agent": "RetailLens/1.0"})
            with urllib.request.urlopen(req, timeout=90) as r, part.open("wb") as f:
                while block := r.read(1024 * 1024):
                    f.write(block)
            part.replace(archive)
        finally:
            part.unlink(missing_ok=True)
    with zipfile.ZipFile(archive) as z:
        with z.open("online_retail_II.xlsx") as r, target.open("wb") as f:
            while block := r.read(1024 * 1024):
                f.write(block)
    return target

def as_id(x):
    if x is None or str(x).strip() == "":
        return None
    if isinstance(x, (int, float)):
        return str(int(x)) if math.isfinite(x) and x == int(x) else str(x)
    return str(x).strip()

def normalize(row, sheet="test"):
    invoice, stock, desc, qty, date, price, customer, country = row
    invoice, stock, customer = as_id(invoice), as_id(stock), as_id(customer)
    desc = str(desc or "").strip()
    country = str(country or "Unknown").strip()
    valid_date = isinstance(date, datetime)
    date_text = date.isoformat(sep=" ") if valid_date else None
    try:
        qty, price = float(qty), float(price)
        valid_number = math.isfinite(qty) and math.isfinite(price)
    except (TypeError, ValueError):
        qty, price, valid_number = 0., 0., False
    # Mutually exclusive reasons; this order defines the audit waterfall.
    if not invoice or not valid_date or not valid_number:
        reason = "invalid_record"
    elif customer is None:
        reason = "missing_customer"
    elif not stock or not re.fullmatch(r"\d{5}[A-Za-z]{0,2}", stock):
        reason = "non_merchandise_code"
    elif price <= 0 or qty == 0:
        reason = "nonpositive_price_or_zero_quantity"
    elif invoice.upper().startswith("C") or qty < 0:
        reason = "return"
    else:
        reason = "purchase"
    values = [invoice, stock, desc, qty, date_text, price, customer, country]
    digest = hashlib.sha256(json.dumps(values, ensure_ascii=False).encode()).hexdigest()
    # Returns use positive magnitude and are reported separately from gross purchases.
    value = abs(qty * price) if valid_number else 0.
    return (*values, sheet, digest, reason, int(reason == "return"), value)

def ingest(source: Path, db: Path, audit_path: Path, rebuild=False):
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    if db.exists() and audit_path.exists() and not rebuild:
        audit = json.loads(audit_path.read_text())
        if audit.get("source_sha256") == checksum:
            return audit
        raise ValueError("Source changed; use --rebuild to refresh the database.")
    db.parent.mkdir(parents=True, exist_ok=True)
    db.unlink(missing_ok=True)
    con = sqlite3.connect(db)
    con.executescript(SCHEMA)
    book = openpyxl.load_workbook(source, read_only=True, data_only=True)
    n_raw, sheet_counts = 0, {}
    sql = """INSERT OR IGNORE INTO lines(invoice,stock_code,description,quantity,
        invoice_date,unit_price,customer_id,country,source_sheet,row_hash,reason,
        is_return,value) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)"""
    try:
        for sheet in book:
            rows = sheet.iter_rows(values_only=True)
            header = next(rows)
            expected = ("Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country")
            if tuple(header) != expected:
                raise ValueError(f"Unexpected columns: {header}")
            batch, count = [], 0
            for row in rows:
                batch.append(normalize(row, sheet.title))
                count += 1
                if len(batch) == 10000:
                    con.executemany(sql, batch)
                    con.commit()
                    batch.clear()
            con.executemany(sql, batch)
            con.commit()
            sheet_counts[sheet.title] = count
            n_raw += count
            print(f"  Imported {sheet.title}: {count:,} rows", flush=True)
        counts = dict(con.execute("SELECT reason,COUNT(*) FROM lines GROUP BY reason"))
        n_unique = sum(counts.values())
        audit = dict(source_url=SOURCE, download_url=URL, source_sha256=checksum,
                     raw_rows=n_raw, sheet_rows=sheet_counts, deduplicated_rows=n_unique,
                     exact_duplicates_removed=n_raw-n_unique, row_disposition=counts,
                     duplicate_policy="Exact normalized eight-column duplicate; keep first across both sheets",
                     merchandise_policy="Stock code: five digits followed by zero to two letters",
                     date_range=list(con.execute("SELECT MIN(invoice_date),MAX(invoice_date) FROM lines").fetchone()))
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps(audit, indent=2))
        return audit
    finally:
        book.close()
        con.close()
