"""
Short Expiry Alert System Endpoints
===================================
Add these endpoints to the main WMS API by importing this module.

Logic:
- WAREHOUSE (location "WAREHOUSE"): Batches IN from SupInvoice + StockReceive (ItemRemark3)
                                    Batches OUT via Transit (ItemRemark3), SupCN (ItemRemark3)
- OUTLETS: Batches IN from Transfer (ItemRemark3 from Transit)
- Only valid ItemRemark3 format counts: YYYY-MM-DD pattern
"""

from datetime import datetime, date, timedelta
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
import asyncpg

router = APIRouter(prefix="/api/v1/expiry", tags=["Expiry Alert"])

# Warehouse location ID (matches AcLocation.AcLocationID in DynaMod)
WAREHOUSE_LOCATION_ID = "WAREHOUSE"


def create_expiry_router(pool_getter, verify_api_key_func):
    """Create expiry router with pool and api key verification injected."""

    @router.get("/summary")
    async def get_expiry_summary(api_key: str = Query(...)):
        """Get company-wide expiry summary grouped by time windows."""
        verify_api_key_func(api_key)
        pool = pool_getter()

        try:
            async with pool.acquire() as conn:
                today = date.today()

                # Warehouse batches
                warehouse_batches = await conn.fetch("""
                    WITH warehouse_in AS (
                        SELECT d."AcStockID" as stock_id,
                               SUBSTRING(d."ItemRemark3" FROM 1 FOR 10) as expiry_str,
                               SUM(COALESCE(d."ItemQuantity", 0)) as qty_in
                        FROM "AcSupInvoiceD" d
                        INNER JOIN "AcSupInvoiceM" m ON d."AcSupInvoiceMID" = m."AcSupInvoiceMID"
                        WHERE d."ItemRemark3" ~ '^20[2-9][0-9]-[01][0-9]-[0-3][0-9]'
                          AND m."AcLocationID" = $1
                        GROUP BY d."AcStockID", SUBSTRING(d."ItemRemark3" FROM 1 FOR 10)
                        UNION ALL
                        SELECT d."AcStockID", SUBSTRING(d."ItemRemark3" FROM 1 FOR 10),
                               SUM(COALESCE(d."ItemQuantity", 0))
                        FROM "AcStockReceiveD" d
                        INNER JOIN "AcStockReceiveM" m ON d."AcStockReceiveMID" = m."AcStockReceiveMID"
                        WHERE d."ItemRemark3" ~ '^20[2-9][0-9]-[01][0-9]-[0-3][0-9]'
                          AND m."AcLocationID" = $1
                        GROUP BY d."AcStockID", SUBSTRING(d."ItemRemark3" FROM 1 FOR 10)
                    ),
                    warehouse_out AS (
                        SELECT d."AcStockID" as stock_id,
                               SUBSTRING(d."ItemRemark3" FROM 1 FOR 10) as expiry_str,
                               SUM(COALESCE(d."ItemQuantity", 0)) as qty_out
                        FROM "AcStockTransitD" d
                        INNER JOIN "AcStockTransitM" m ON d."AcStockTransitMID" = m."AcStockTransitMID"
                        WHERE d."ItemRemark3" ~ '^20[2-9][0-9]-[01][0-9]-[0-3][0-9]'
                          AND m."AcLocationIDFrom" = $1
                        GROUP BY d."AcStockID", SUBSTRING(d."ItemRemark3" FROM 1 FOR 10)
                        UNION ALL
                        SELECT d."AcStockID", SUBSTRING(d."ItemRemark3" FROM 1 FOR 10),
                               SUM(COALESCE(d."ItemQuantity", 0))
                        FROM "AcSupCreditNoteD" d
                        INNER JOIN "AcSupCreditNoteM" m ON d."AcSupCreditNoteMID" = m."AcSupCreditNoteMID"
                        WHERE d."ItemRemark3" ~ '^20[2-9][0-9]-[01][0-9]-[0-3][0-9]'
                          AND m."AcLocationID" = $1
                        GROUP BY d."AcStockID", SUBSTRING(d."ItemRemark3" FROM 1 FOR 10)
                    ),
                    batch_balance AS (
                        SELECT COALESCE(i.stock_id, o.stock_id) as stock_id,
                               COALESCE(i.expiry_str, o.expiry_str) as expiry_str,
                               COALESCE(SUM(i.qty_in), 0) - COALESCE(SUM(o.qty_out), 0) as remaining
                        FROM warehouse_in i
                        FULL OUTER JOIN warehouse_out o ON i.stock_id = o.stock_id AND i.expiry_str = o.expiry_str
                        GROUP BY COALESCE(i.stock_id, o.stock_id), COALESCE(i.expiry_str, o.expiry_str)
                        HAVING COALESCE(SUM(i.qty_in), 0) - COALESCE(SUM(o.qty_out), 0) > 0
                    )
                    SELECT stock_id, expiry_str, remaining,
                           sc."StockDescription1" as stock_name, COALESCE(sc."StockCost", 0) as unit_cost
                    FROM batch_balance b
                    LEFT JOIN "AcStockCompany" sc ON b.stock_id = sc."AcStockID"
                    WHERE remaining > 0
                """, WAREHOUSE_LOCATION_ID)

                # Outlet batches
                outlet_batches = await conn.fetch("""
                    SELECT d."AcStockID" as stock_id, m."AcLocationIDTo" as location_id,
                           SUBSTRING(d."ItemRemark3" FROM 1 FOR 10) as expiry_str,
                           SUM(COALESCE(d."ItemQuantity", 0)) as qty,
                           COALESCE(sc."StockCost", 0) as unit_cost
                    FROM "AcStockTransferD" d
                    INNER JOIN "AcStockTransferM" m ON d."AcStockTransferMID" = m."AcStockTransferMID"
                    LEFT JOIN "AcStockCompany" sc ON d."AcStockID" = sc."AcStockID"
                    WHERE d."ItemRemark3" ~ '^20[2-9][0-9]-[01][0-9]-[0-3][0-9]'
                      AND m."AcLocationIDTo" != $1
                    GROUP BY d."AcStockID", m."AcLocationIDTo", SUBSTRING(d."ItemRemark3" FROM 1 FOR 10), sc."StockCost"
                    HAVING SUM(COALESCE(d."ItemQuantity", 0)) > 0
                """, WAREHOUSE_LOCATION_ID)

                windows = ["expired", "0_30_days", "31_60_days", "61_90_days", "91_180_days", "over_180_days"]
                wh_summary = {k: {"batches": 0, "units": 0, "value": 0.0} for k in windows}
                outlet_summary = {k: {"batches": 0, "units": 0, "value": 0.0} for k in windows}

                def categorize(expiry_str, qty, cost):
                    try:
                        # Handle dates with day=00 (end of month)
                        exp_str = expiry_str
                        if exp_str.endswith('-00'):
                            exp_str = exp_str[:-2] + '28'  # Use 28 as safe last day
                        exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
                        days = (exp_date - today).days
                        val = float(qty) * float(cost)
                        if days < 0: return "expired", val
                        elif days <= 30: return "0_30_days", val
                        elif days <= 60: return "31_60_days", val
                        elif days <= 90: return "61_90_days", val
                        elif days <= 180: return "91_180_days", val
                        else: return "over_180_days", val
                    except: return None, 0

                for b in warehouse_batches:
                    cat, val = categorize(b['expiry_str'], b['remaining'], b['unit_cost'])
                    if cat:
                        wh_summary[cat]["batches"] += 1
                        wh_summary[cat]["units"] += int(b['remaining'])
                        wh_summary[cat]["value"] += val

                for b in outlet_batches:
                    cat, val = categorize(b['expiry_str'], b['qty'], b['unit_cost'])
                    if cat:
                        outlet_summary[cat]["batches"] += 1
                        outlet_summary[cat]["units"] += int(b['qty'])
                        outlet_summary[cat]["value"] += val

                total = {k: {
                    "batches": wh_summary[k]["batches"] + outlet_summary[k]["batches"],
                    "units": wh_summary[k]["units"] + outlet_summary[k]["units"],
                    "value": round(wh_summary[k]["value"] + outlet_summary[k]["value"], 2)
                } for k in windows}

                at_risk_keys = ["expired", "0_30_days", "31_60_days", "61_90_days"]
                at_risk = {
                    "batches": sum(total[k]["batches"] for k in at_risk_keys),
                    "units": sum(total[k]["units"] for k in at_risk_keys),
                    "value": round(sum(total[k]["value"] for k in at_risk_keys), 2)
                }

                return {
                    "generated_at": datetime.now().isoformat(),
                    "as_of_date": today.isoformat(),
                    "at_risk_summary": at_risk,
                    "by_window": total,
                    "warehouse": wh_summary,
                    "outlets_combined": outlet_summary
                }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/by-location")
    async def get_expiry_by_location(
        days_threshold: int = Query(90, description="Days threshold for at-risk"),
        api_key: str = Query(...)
    ):
        """Get expiry summary by location."""
        verify_api_key_func(api_key)
        pool = pool_getter()

        try:
            async with pool.acquire() as conn:
                today = date.today()
                threshold = today + timedelta(days=days_threshold)

                wh_data = await conn.fetch("""
                    WITH wh_in AS (
                        SELECT d."AcStockID" as stock_id,
                               SUBSTRING(d."ItemRemark3" FROM 1 FOR 10) as expiry_str,
                               SUM(COALESCE(d."ItemQuantity", 0)) as qty_in
                        FROM "AcSupInvoiceD" d
                        INNER JOIN "AcSupInvoiceM" m ON d."AcSupInvoiceMID" = m."AcSupInvoiceMID"
                        WHERE d."ItemRemark3" ~ '^20[2-9][0-9]-[01][0-9]-[0-3][0-9]' AND m."AcLocationID" = $1
                        GROUP BY d."AcStockID", SUBSTRING(d."ItemRemark3" FROM 1 FOR 10)
                        UNION ALL
                        SELECT d."AcStockID", SUBSTRING(d."ItemRemark3" FROM 1 FOR 10), SUM(COALESCE(d."ItemQuantity", 0))
                        FROM "AcStockReceiveD" d
                        INNER JOIN "AcStockReceiveM" m ON d."AcStockReceiveMID" = m."AcStockReceiveMID"
                        WHERE d."ItemRemark3" ~ '^20[2-9][0-9]-[01][0-9]-[0-3][0-9]' AND m."AcLocationID" = $1
                        GROUP BY d."AcStockID", SUBSTRING(d."ItemRemark3" FROM 1 FOR 10)
                    ),
                    wh_out AS (
                        SELECT d."AcStockID" as stock_id,
                               SUBSTRING(d."ItemRemark3" FROM 1 FOR 10) as expiry_str,
                               SUM(COALESCE(d."ItemQuantity", 0)) as qty_out
                        FROM "AcStockTransitD" d
                        INNER JOIN "AcStockTransitM" m ON d."AcStockTransitMID" = m."AcStockTransitMID"
                        WHERE d."ItemRemark3" ~ '^20[2-9][0-9]-[01][0-9]-[0-3][0-9]' AND m."AcLocationIDFrom" = $1
                        GROUP BY d."AcStockID", SUBSTRING(d."ItemRemark3" FROM 1 FOR 10)
                        UNION ALL
                        SELECT d."AcStockID", SUBSTRING(d."ItemRemark3" FROM 1 FOR 10), SUM(COALESCE(d."ItemQuantity", 0))
                        FROM "AcSupCreditNoteD" d
                        INNER JOIN "AcSupCreditNoteM" m ON d."AcSupCreditNoteMID" = m."AcSupCreditNoteMID"
                        WHERE d."ItemRemark3" ~ '^20[2-9][0-9]-[01][0-9]-[0-3][0-9]' AND m."AcLocationID" = $1
                        GROUP BY d."AcStockID", SUBSTRING(d."ItemRemark3" FROM 1 FOR 10)
                    )
                    SELECT COALESCE(i.stock_id, o.stock_id) as stock_id,
                           COALESCE(i.expiry_str, o.expiry_str) as expiry_str,
                           COALESCE(SUM(i.qty_in), 0) - COALESCE(SUM(o.qty_out), 0) as remaining,
                           sc."StockCost" as unit_cost
                    FROM wh_in i
                    FULL OUTER JOIN wh_out o ON i.stock_id = o.stock_id AND i.expiry_str = o.expiry_str
                    LEFT JOIN "AcStockCompany" sc ON COALESCE(i.stock_id, o.stock_id) = sc."AcStockID"
                    GROUP BY COALESCE(i.stock_id, o.stock_id), COALESCE(i.expiry_str, o.expiry_str), sc."StockCost"
                    HAVING COALESCE(SUM(i.qty_in), 0) - COALESCE(SUM(o.qty_out), 0) > 0
                """, WAREHOUSE_LOCATION_ID)

                wh_exp = {"batches": 0, "units": 0, "value": 0.0}
                wh_risk = {"batches": 0, "units": 0, "value": 0.0}
                wh_safe = {"batches": 0, "units": 0, "value": 0.0}

                for b in wh_data:
                    try:
                        exp_str = b['expiry_str']
                        if exp_str.endswith('-00'):
                            exp_str = exp_str[:-2] + '28'
                        exp = datetime.strptime(exp_str, '%Y-%m-%d').date()
                        qty = int(b['remaining'])
                        val = float(qty) * float(b['unit_cost'] or 0)
                        if exp < today:
                            wh_exp["batches"] += 1; wh_exp["units"] += qty; wh_exp["value"] += val
                        elif exp <= threshold:
                            wh_risk["batches"] += 1; wh_risk["units"] += qty; wh_risk["value"] += val
                        else:
                            wh_safe["batches"] += 1; wh_safe["units"] += qty; wh_safe["value"] += val
                    except: pass

                outlet_data = await conn.fetch("""
                    SELECT m."AcLocationIDTo" as loc_id, l."AcLocationDesc" as loc_name,
                           SUBSTRING(d."ItemRemark3" FROM 1 FOR 10) as expiry_str,
                           SUM(COALESCE(d."ItemQuantity", 0)) as qty, sc."StockCost" as unit_cost
                    FROM "AcStockTransferD" d
                    INNER JOIN "AcStockTransferM" m ON d."AcStockTransferMID" = m."AcStockTransferMID"
                    LEFT JOIN "AcLocation" l ON m."AcLocationIDTo" = l."AcLocationID"
                    LEFT JOIN "AcStockCompany" sc ON d."AcStockID" = sc."AcStockID"
                    WHERE d."ItemRemark3" ~ '^20[2-9][0-9]-[01][0-9]-[0-3][0-9]' AND m."AcLocationIDTo" != $1
                    GROUP BY m."AcLocationIDTo", l."AcLocationDesc", SUBSTRING(d."ItemRemark3" FROM 1 FOR 10), sc."StockCost"
                    HAVING SUM(COALESCE(d."ItemQuantity", 0)) > 0
                """, WAREHOUSE_LOCATION_ID)

                locs = {}
                for b in outlet_data:
                    lid = b['loc_id']
                    if lid not in locs:
                        locs[lid] = {"location_id": lid, "location_name": b['loc_name'],
                                     "expired": {"batches": 0, "units": 0, "value": 0.0},
                                     "at_risk": {"batches": 0, "units": 0, "value": 0.0},
                                     "safe": {"batches": 0, "units": 0, "value": 0.0}}
                    try:
                        exp_str = b['expiry_str']
                        if exp_str.endswith('-00'):
                            exp_str = exp_str[:-2] + '28'
                        exp = datetime.strptime(exp_str, '%Y-%m-%d').date()
                        qty = int(b['qty'])
                        val = float(qty) * float(b['unit_cost'] or 0)
                        cat = "expired" if exp < today else "at_risk" if exp <= threshold else "safe"
                        locs[lid][cat]["batches"] += 1
                        locs[lid][cat]["units"] += qty
                        locs[lid][cat]["value"] += val
                    except: pass

                for loc in locs.values():
                    for c in ["expired", "at_risk", "safe"]:
                        loc[c]["value"] = round(loc[c]["value"], 2)

                return {
                    "generated_at": datetime.now().isoformat(),
                    "days_threshold": days_threshold,
                    "warehouse": {
                        "location_id": WAREHOUSE_LOCATION_ID, "location_name": "WAREHOUSE",
                        "expired": {**wh_exp, "value": round(wh_exp["value"], 2)},
                        "at_risk": {**wh_risk, "value": round(wh_risk["value"], 2)},
                        "safe": {**wh_safe, "value": round(wh_safe["value"], 2)}
                    },
                    "outlets": sorted(locs.values(), key=lambda x: x["at_risk"]["value"], reverse=True)
                }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/items")
    async def get_expiry_items(
        location_id: Optional[str] = Query(None),
        window: str = Query("at_risk", description="expired, at_risk, 30_days, 60_days, 90_days, all"),
        limit: int = Query(100),
        api_key: str = Query(...)
    ):
        """Get detailed list of items with expiry concerns."""
        verify_api_key_func(api_key)
        pool = pool_getter()

        try:
            async with pool.acquire() as conn:
                today = date.today()

                if window == "expired":
                    date_cond = f"< '{today}'"
                elif window == "30_days":
                    date_cond = f"BETWEEN '{today}' AND '{today + timedelta(days=30)}'"
                elif window == "60_days":
                    date_cond = f"BETWEEN '{today}' AND '{today + timedelta(days=60)}'"
                elif window in ["90_days", "at_risk"]:
                    date_cond = f"BETWEEN '{today - timedelta(days=365)}' AND '{today + timedelta(days=90)}'"
                else:
                    date_cond = ">= '2020-01-01'"

                items = []

                if location_id == WAREHOUSE_LOCATION_ID or location_id is None:
                    wh_items = await conn.fetch(f"""
                        WITH wh_in AS (
                            SELECT d."AcStockID" as stock_id, SUBSTRING(d."ItemRemark3" FROM 1 FOR 10) as expiry_str,
                                   SUM(COALESCE(d."ItemQuantity", 0)) as qty
                            FROM "AcSupInvoiceD" d
                            INNER JOIN "AcSupInvoiceM" m ON d."AcSupInvoiceMID" = m."AcSupInvoiceMID"
                            WHERE d."ItemRemark3" ~ '^20[2-9][0-9]-[01][0-9]-[0-3][0-9]' AND m."AcLocationID" = $1
                            GROUP BY d."AcStockID", SUBSTRING(d."ItemRemark3" FROM 1 FOR 10)
                            UNION ALL
                            SELECT d."AcStockID", SUBSTRING(d."ItemRemark3" FROM 1 FOR 10), SUM(COALESCE(d."ItemQuantity", 0))
                            FROM "AcStockReceiveD" d INNER JOIN "AcStockReceiveM" m ON d."AcStockReceiveMID" = m."AcStockReceiveMID"
                            WHERE d."ItemRemark3" ~ '^20[2-9][0-9]-[01][0-9]-[0-3][0-9]' AND m."AcLocationID" = $1
                            GROUP BY d."AcStockID", SUBSTRING(d."ItemRemark3" FROM 1 FOR 10)
                        ),
                        wh_out AS (
                            SELECT d."AcStockID" as stock_id, SUBSTRING(d."ItemRemark3" FROM 1 FOR 10) as expiry_str,
                                   SUM(COALESCE(d."ItemQuantity", 0)) as qty
                            FROM "AcStockTransitD" d INNER JOIN "AcStockTransitM" m ON d."AcStockTransitMID" = m."AcStockTransitMID"
                            WHERE d."ItemRemark3" ~ '^20[2-9][0-9]-[01][0-9]-[0-3][0-9]' AND m."AcLocationIDFrom" = $1
                            GROUP BY d."AcStockID", SUBSTRING(d."ItemRemark3" FROM 1 FOR 10)
                            UNION ALL
                            SELECT d."AcStockID", SUBSTRING(d."ItemRemark3" FROM 1 FOR 10), SUM(COALESCE(d."ItemQuantity", 0))
                            FROM "AcSupCreditNoteD" d INNER JOIN "AcSupCreditNoteM" m ON d."AcSupCreditNoteMID" = m."AcSupCreditNoteMID"
                            WHERE d."ItemRemark3" ~ '^20[2-9][0-9]-[01][0-9]-[0-3][0-9]' AND m."AcLocationID" = $1
                            GROUP BY d."AcStockID", SUBSTRING(d."ItemRemark3" FROM 1 FOR 10)
                        ),
                        bal AS (
                            SELECT COALESCE(i.stock_id, o.stock_id) as stock_id,
                                   COALESCE(i.expiry_str, o.expiry_str) as expiry_str,
                                   COALESCE(SUM(i.qty), 0) - COALESCE(SUM(o.qty), 0) as remaining
                            FROM wh_in i FULL OUTER JOIN wh_out o ON i.stock_id = o.stock_id AND i.expiry_str = o.expiry_str
                            GROUP BY COALESCE(i.stock_id, o.stock_id), COALESCE(i.expiry_str, o.expiry_str)
                            HAVING COALESCE(SUM(i.qty), 0) - COALESCE(SUM(o.qty), 0) > 0
                        )
                        SELECT $1 as location_id, 'WAREHOUSE' as location_name, b.stock_id,
                               sc."StockDescription1" as stock_name, sc."StockBarcode" as barcode, b.expiry_str,
                               b.remaining as qty, COALESCE(sc."StockCost", 0) as unit_cost,
                               b.remaining * COALESCE(sc."StockCost", 0) as total_value,
                               CASE WHEN b.expiry_str LIKE '%-00'
                                    THEN TO_DATE(SUBSTRING(b.expiry_str FROM 1 FOR 7) || '-28', 'YYYY-MM-DD') - CURRENT_DATE
                                    ELSE TO_DATE(b.expiry_str, 'YYYY-MM-DD') - CURRENT_DATE END as days_until
                        FROM bal b
                        LEFT JOIN "AcStockCompany" sc ON b.stock_id = sc."AcStockID"
                        WHERE b.remaining > 0
                          AND CASE WHEN b.expiry_str LIKE '%-00'
                                   THEN TO_DATE(SUBSTRING(b.expiry_str FROM 1 FOR 7) || '-28', 'YYYY-MM-DD')
                                   ELSE TO_DATE(b.expiry_str, 'YYYY-MM-DD') END {date_cond}
                        ORDER BY days_until LIMIT $2
                    """, WAREHOUSE_LOCATION_ID, limit)
                    for r in wh_items:
                        items.append({
                            "location_id": r['location_id'], "location_name": r['location_name'],
                            "stock_id": r['stock_id'], "stock_name": r['stock_name'],
                            "barcode": r['barcode'], "expiry_date": r['expiry_str'],
                            "quantity": int(r['qty']), "unit_cost": float(r['unit_cost']),
                            "total_value": round(float(r['total_value']), 2),
                            "days_until_expiry": int(r['days_until'])
                        })

                if location_id != WAREHOUSE_LOCATION_ID:
                    loc_filter = f"= '{location_id}'" if location_id else f"!= '{WAREHOUSE_LOCATION_ID}'"
                    outlet_items = await conn.fetch(f"""
                        SELECT m."AcLocationIDTo" as location_id, l."AcLocationDesc" as location_name,
                               d."AcStockID" as stock_id, sc."StockDescription1" as stock_name, sc."StockBarcode" as barcode,
                               SUBSTRING(d."ItemRemark3" FROM 1 FOR 10) as expiry_str,
                               SUM(COALESCE(d."ItemQuantity", 0)) as qty, COALESCE(sc."StockCost", 0) as unit_cost,
                               SUM(COALESCE(d."ItemQuantity", 0)) * COALESCE(sc."StockCost", 0) as total_value,
                               CASE WHEN SUBSTRING(d."ItemRemark3" FROM 1 FOR 10) LIKE '%-00'
                                    THEN TO_DATE(SUBSTRING(d."ItemRemark3" FROM 1 FOR 7) || '-28', 'YYYY-MM-DD') - CURRENT_DATE
                                    ELSE TO_DATE(SUBSTRING(d."ItemRemark3" FROM 1 FOR 10), 'YYYY-MM-DD') - CURRENT_DATE END as days_until
                        FROM "AcStockTransferD" d
                        INNER JOIN "AcStockTransferM" m ON d."AcStockTransferMID" = m."AcStockTransferMID"
                        LEFT JOIN "AcLocation" l ON m."AcLocationIDTo" = l."AcLocationID"
                        LEFT JOIN "AcStockCompany" sc ON d."AcStockID" = sc."AcStockID"
                        WHERE d."ItemRemark3" ~ '^20[2-9][0-9]-[01][0-9]-[0-3][0-9]'
                          AND m."AcLocationIDTo" {loc_filter}
                          AND CASE WHEN SUBSTRING(d."ItemRemark3" FROM 1 FOR 10) LIKE '%-00'
                                   THEN TO_DATE(SUBSTRING(d."ItemRemark3" FROM 1 FOR 7) || '-28', 'YYYY-MM-DD')
                                   ELSE TO_DATE(SUBSTRING(d."ItemRemark3" FROM 1 FOR 10), 'YYYY-MM-DD') END {date_cond}
                        GROUP BY m."AcLocationIDTo", l."AcLocationDesc", d."AcStockID", sc."StockDescription1",
                                 sc."StockBarcode", SUBSTRING(d."ItemRemark3" FROM 1 FOR 10), sc."StockCost"
                        HAVING SUM(COALESCE(d."ItemQuantity", 0)) > 0
                        ORDER BY days_until LIMIT $1
                    """, limit)
                    for r in outlet_items:
                        items.append({
                            "location_id": r['location_id'], "location_name": r['location_name'],
                            "stock_id": r['stock_id'], "stock_name": r['stock_name'],
                            "barcode": r['barcode'], "expiry_date": r['expiry_str'],
                            "quantity": int(r['qty']), "unit_cost": float(r['unit_cost']),
                            "total_value": round(float(r['total_value']), 2),
                            "days_until_expiry": int(r['days_until'])
                        })

                items.sort(key=lambda x: x['days_until_expiry'])
                return {
                    "generated_at": datetime.now().isoformat(),
                    "filter": {"location_id": location_id, "window": window, "limit": limit},
                    "count": len(items), "items": items[:limit]
                }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return router
