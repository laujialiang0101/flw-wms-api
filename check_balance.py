import asyncio
import asyncpg

async def check_balance():
    conn = await asyncpg.connect(
        host='dpg-d4pr99je5dus73eb5730-a.singapore-postgres.render.com',
        port=5432,
        database='flt_sales_commission_db',
        user='flt_sales_commission_db_user',
        password='Wy0ZP1wjLPsIta0YLpYLeRWgdITbya2m',
        ssl='require'
    )

    # Check Hydrocyn TPH00002
    print('=== HYDROCYN AQUA (TPH00002) ===')
    rows = await conn.fetch('''
        SELECT sb."AcStockID", sb."AcLocationID", sb."BalanceQuantity",
               sc."StockDescription1", sc."AcStockUOMID", sc."AcStockUOMIDBaseID"
        FROM "AcStockBalanceLocation" sb
        JOIN "AcStockCompany" sc ON sb."AcStockID" = sc."AcStockID"
        WHERE sb."AcStockID" LIKE 'TPH00002%'
        ORDER BY sb."AcStockID", sb."AcLocationID"
    ''')
    for r in rows[:20]:
        print(f'{r[0]} | Loc: {r[1]} | Bal: {r[2]} | UOM: {r[4]} | BaseUOM: {r[5]}')

    total = await conn.fetchval('SELECT SUM("BalanceQuantity") FROM "AcStockBalanceLocation" WHERE "AcStockID" = $1', 'TPH00002')
    print(f'>>> Total for TPH00002: {total}')

    # Check for other UOM variants
    variants = await conn.fetch('''
        SELECT "AcStockID", "StockDescription1", "AcStockUOMID", "AcStockUOMIDBaseID"
        FROM "AcStockCompany" WHERE "AcStockID" LIKE 'TPH00002%'
    ''')
    print('Variants:')
    for v in variants:
        print(f'  {v[0]} | {v[1]} | UOM: {v[2]} | BaseUOM: {v[3]}')

    # Check Koolfever TTK00032
    print()
    print('=== KOOLFEVER BABIES (TTK00032) ===')
    rows = await conn.fetch('''
        SELECT sb."AcStockID", sb."AcLocationID", sb."BalanceQuantity",
               sc."StockDescription1", sc."AcStockUOMID", sc."AcStockUOMIDBaseID"
        FROM "AcStockBalanceLocation" sb
        JOIN "AcStockCompany" sc ON sb."AcStockID" = sc."AcStockID"
        WHERE sb."AcStockID" LIKE 'TTK00032%'
        ORDER BY sb."AcStockID", sb."AcLocationID"
    ''')
    for r in rows[:20]:
        print(f'{r[0]} | Loc: {r[1]} | Bal: {r[2]} | UOM: {r[4]} | BaseUOM: {r[5]}')

    total = await conn.fetchval('SELECT SUM("BalanceQuantity") FROM "AcStockBalanceLocation" WHERE "AcStockID" = $1', 'TTK00032')
    print(f'>>> Total for TTK00032: {total}')

    # Check for other UOM variants
    variants = await conn.fetch('''
        SELECT "AcStockID", "StockDescription1", "AcStockUOMID", "AcStockUOMIDBaseID"
        FROM "AcStockCompany" WHERE "AcStockID" LIKE 'TTK00032%'
    ''')
    print('Variants:')
    for v in variants:
        print(f'  {v[0]} | {v[1]} | UOM: {v[2]} | BaseUOM: {v[3]}')

    # Check Topseal WC519047
    print()
    print('=== TOPSEAL DRESSING (WC519047) ===')
    rows = await conn.fetch('''
        SELECT sb."AcStockID", sb."AcLocationID", sb."BalanceQuantity",
               sc."StockDescription1", sc."AcStockUOMID", sc."AcStockUOMIDBaseID"
        FROM "AcStockBalanceLocation" sb
        JOIN "AcStockCompany" sc ON sb."AcStockID" = sc."AcStockID"
        WHERE sb."AcStockID" LIKE 'WC519047%'
        ORDER BY sb."AcStockID", sb."AcLocationID"
    ''')
    for r in rows[:20]:
        print(f'{r[0]} | Loc: {r[1]} | Bal: {r[2]} | UOM: {r[4]} | BaseUOM: {r[5]}')

    total = await conn.fetchval('SELECT SUM("BalanceQuantity") FROM "AcStockBalanceLocation" WHERE "AcStockID" = $1', 'WC519047')
    print(f'>>> Total for WC519047: {total}')

    # Check for other UOM variants
    variants = await conn.fetch('''
        SELECT "AcStockID", "StockDescription1", "AcStockUOMID", "AcStockUOMIDBaseID"
        FROM "AcStockCompany" WHERE "AcStockID" LIKE 'WC519047%'
    ''')
    print('Variants:')
    for v in variants:
        print(f'  {v[0]} | {v[1]} | UOM: {v[2]} | BaseUOM: {v[3]}')

    # Check what was stored in the movement summary table
    print()
    print('=== WHAT IS IN STOCK MOVEMENT TABLE ===')
    rows = await conn.fetch('''
        SELECT stock_id, stock_name, current_balance, days_of_inventory, avg_daily_30d
        FROM wms.stock_movement_summary
        WHERE stock_id IN ('TPH00002', 'TTK00032', 'WC519047')
    ''')
    for r in rows:
        print(f'{r[0]} | Balance: {r[2]} | DOI: {r[3]} | AvgDaily: {r[4]}')

    await conn.close()

asyncio.run(check_balance())
