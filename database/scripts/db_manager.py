import sys
import argparse
import pandas as pd
import pymysql
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from config import DB_CONFIG, EXCEL_FILE, SQL_FILE

class DatabaseManager:
    def __init__(self):
        self.config = DB_CONFIG.copy()
        self.config['cursorclass'] = pymysql.cursors.DictCursor
    
    def connect(self):
        try:
            return pymysql.connect(**self.config)
        except Exception as e:
            print(f"Connection failed: {e}")
            sys.exit(1)
    
    def test_connection(self):
        conn = self.connect()
        with conn.cursor() as cursor:
            cursor.execute("SELECT VERSION()")
            version = cursor.fetchone()
            print(f"✓ MySQL {version['VERSION()']}")
            
            cursor.execute(f"USE {self.config['database']}")
            cursor.execute("SELECT COUNT(*) as count FROM flight")
            count = cursor.fetchone()
            print(f"✓ Database connected: {count['count']} flights")
        conn.close()
    
    def import_sql(self):
        if not SQL_FILE.exists():
            print(f"Error: {SQL_FILE} not found")
            sys.exit(1)
        
        import subprocess
        cmd = f"mysql -u {self.config['user']} -p{self.config['password']} < {SQL_FILE}"
        subprocess.run(cmd, shell=True, check=True)
        print("✓ Database imported")
    
    def import_excel(self):
        # df = pd.read_excel(EXCEL_FILE)
        flights_df = pd.read_excel(EXCEL_FILE, sheet_name='flights')
        assignments_df = pd.read_excel(EXCEL_FILE, sheet_name='assignments', header=2)
    
        cand = flights_df.merge(
            assignments_df,
            left_on='Flight ID',
            right_on='Call Sign (VDGS)',
            how='left'
        )
    
        cand['inblock_diff'] = (
            pd.to_datetime(cand['Actual In Block Time (VDGS) (US Pacific)'], errors='coerce') -
            pd.to_datetime(cand['Actual In Block Time (Aerobahn) (US Pacific)'], errors='coerce')
        ).abs()
    
        cand['offblock_diff'] = (
            pd.to_datetime(cand['Actual Off Block Time (VDGS) (US Pacific)'], errors='coerce') -
            pd.to_datetime(cand['Actual Off Block Time (Aerobahn) (US Pacific)'], errors='coerce')
        ).abs()
    
        cand['match_diff'] = cand['inblock_diff'].combine_first(cand['offblock_diff'])
    
        cand = cand.reset_index(drop=True)
        cand['merge_row_id'] = cand.index
    

        best = cand.sort_values('match_diff', na_position='last').drop_duplicates(
            subset=['Call Sign', 'Flight ID', 'Actual Off Block Time (Aerobahn) (US Pacific)', 'Actual In Block Time (Aerobahn) (US Pacific)'],
            keep='first'
        )
    
        df = best

        
        conn = self.connect()
        
        with conn.cursor() as cursor:
            cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
            cursor.execute("TRUNCATE flight_event")
            cursor.execute("TRUNCATE flight")
            cursor.execute("TRUNCATE aircraft_type")
            cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
        
        self._import_aircraft_types(conn, df)
        self._import_flights(conn, df)
        self._import_events(conn, df)
        
        conn.commit()
        conn.close()
        print("✓ Excel data imported")
    
    def _import_aircraft_types(self, conn, df):
        aircraft_df = df[['Aircraft Type (AODB)', 'Weight Class (ICAO)', 
                          'Wake Category (Aerobahn)', 'Wingspan (feet)', 
                          'Wingspan (meters)']].drop_duplicates()
        
        with conn.cursor() as cursor:
            for _, row in aircraft_df.iterrows():
                if pd.notna(row['Aircraft Type (AODB)']):
                    cursor.execute("""
                        INSERT INTO aircraft_type VALUES (%s,%s,%s,%s,%s)
                        ON DUPLICATE KEY UPDATE weight_class=VALUES(weight_class)
                    """, tuple(row))
    
    def _import_flights(self, conn, df):
        flight_cols = {
            'Call Sign': 'call_sign',
            'Flight ID': 'flight_ID',
            'Aircraft Type (AODB).1': 'aircraft_type',
            'Flight Number': 'flight_number',
            'Registration (AODB)': 'aircraft_registration',
            'Origination Airport (ICAO)': 'origin_airport',
            'Destination Airport (ICAO)': 'destination_airport',
            'Operation': 'operation'
        }
        
        flight_df = df[list(flight_cols.keys())].rename(columns=flight_cols)
        
        # Clean operation: must be DEPARTURE or ARRIVAL, otherwise NULL
        def clean_operation(val):
            if pd.isna(val):
                return None
            val = str(val).strip().upper()
            return val if val in ('DEPARTURE', 'ARRIVAL') else None
        
        flight_df['operation'] = flight_df['operation'].apply(clean_operation)
        
        with conn.cursor() as cursor:
            for _, row in flight_df.iterrows():
                values = [None if pd.isna(v) else v for v in row.values]
                cursor.execute("""
                    INSERT INTO flight (call_sign, flight_ID, aircraft_type, 
                                      flight_number, aircraft_registration,
                                      origin_airport, destination_airport, operation)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                """, values)
    
    def _import_events(self, conn, df):
        df = df.where(pd.notna(df), None)
        event_mappings = [
            # Gate / Runway 
            ('Actual Off Block Time (Aerobahn) (US Pacific)', 'Actual_Off_Block', 'Gate'),
            ('Actual Take Off Time (Aerobahn) (US Pacific)', 'Actual_Take_Off', 'Runway'),
            ('Actual Landing Time (Aerobahn) (US Pacific)', 'Actual_Landing', 'Runway'),
            ('Actual In Block Time (Aerobahn) (US Pacific)', 'Actual_In_Block', 'Gate'),
        
            # Ramp
            ('North Ramp  Enter Time (US Pacific)', 'North_Ramp_Enter', 'North_Ramp'),
            ('North Ramp  Exit Time (US Pacific)', 'North_Ramp_Exit', 'North_Ramp'),
            ('South Ramp  Enter Time (US Pacific)', 'South_Ramp_Enter', 'South_Ramp'),
            ('South Ramp  Exit Time (US Pacific)', 'South_Ramp_Exit', 'South_Ramp'),
        ]
        
        with conn.cursor() as cursor:
            for _, row in df.iterrows():
                op = str(row.get('Operation', '')).strip().upper()
                if op not in ('DEPARTURE', 'ARRIVAL'):
                    continue
                    
                for time_col, event_type, loc_type in event_mappings:
                    # if pd.notna(row.get(time_col)):
                    if time_col in df.columns and pd.notna(row.get(time_col)):
                        location = self._get_location(row, loc_type)
                        cursor.execute("""
                            INSERT INTO flight_event 
                            (call_sign, operation, event_type, event_time, location)
                            VALUES (%s,%s,%s,%s,%s)
                        """, (row['Call Sign'], op, event_type, row[time_col], location))
    
    def _get_location(self, row, loc_type):
        if loc_type == 'Gate':
            return row.get('Gate Assigned (Aerobahn)')
        elif loc_type == 'Runway':
            return row.get('Runway Assigned (Aerobahn)')
        elif loc_type == 'North_Ramp':
            return 'North Ramp'
        elif loc_type == 'South_Ramp':
            return 'South Ramp'
        return None
    
    def verify(self):
        conn = self.connect()
        
        with conn.cursor() as cursor:
            cursor.execute("SELECT VERSION()")
            version = cursor.fetchone()
            print(f"✓ MySQL {version['VERSION()']}")

            cursor.execute(f"SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA WHERE SCHEMA_NAME = '{self.config['database']}'")
            if cursor.fetchone():
                print(f"✓ Database '{self.config['database']}' exists")

            expected = {'aircraft_type': 16, 'flight': 725, 'flight_event': 7995}
            all_passed = True
            
            for table, expected_count in expected.items():
                cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
                result = cursor.fetchone()
                actual = result['count']
                status = "✓" if actual == expected_count else "⚠"
                print(f"  {status} {table}: {actual} records")
                if actual != expected_count:
                    all_passed = False
        
        conn.close()
        
        if all_passed:
            print("✅ VERIFICATION PASSED")
        else:
            print("⚠️ VERIFICATION COMPLETED WITH WARNINGS")
    
    # =========================================================================
    # Smart UNKNOWN Matcher
    # =========================================================================
    
    def fix_nulls(self):
        """Smart matching for NULL aircraft_type using Excel properties."""
        if not EXCEL_FILE.exists():
            print(f"⚠ Excel not found, using simple fix")
            self._fix_nulls_simple()
            return
        
        print("=" * 60)
        print("SMART UNKNOWN AIRCRAFT MATCHER")
        print("=" * 60)
        
        df = pd.read_excel(EXCEL_FILE)
        conn = self.connect()
        
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT DISTINCT call_sign FROM flight WHERE aircraft_type IS NULL")
                null_flights = [row['call_sign'] for row in cursor.fetchall()]
            
            if not null_flights:
                print("✓ No flights with NULL aircraft_type")
                conn.close()
                return
            
            print(f"Found {len(null_flights)} flight(s) with NULL aircraft_type\n")
            
            stats = {'reused': 0, 'created': 0, 'skipped': 0}
            
            with conn.cursor() as cursor:
                for call_sign in null_flights:
                    props = self._get_aircraft_properties(call_sign, df)
                    
                    if props is None:
                        print(f"  ⚠ {call_sign}: not in Excel, skipped")
                        stats['skipped'] += 1
                        continue
                    
                    matching = self._find_matching_unknown(props, cursor)
                    
                    if matching:
                        aircraft_type = matching
                        stats['reused'] += 1
                        print(f"  ✓ {call_sign} → reused {aircraft_type}")
                    else:
                        next_num = self._get_next_unknown_number(cursor)
                        aircraft_type = f"UNKNOWN_{next_num:03d}"
                        self._create_unknown_record(props, aircraft_type, cursor)
                        stats['created'] += 1
                        print(f"  + {call_sign} → created {aircraft_type}")
                    
                    cursor.execute("""
                        UPDATE flight SET aircraft_type = %s 
                        WHERE call_sign = %s AND aircraft_type IS NULL
                    """, (aircraft_type, call_sign))
            
            conn.commit()
            print(f"\n✓ Reused: {stats['reused']}, Created: {stats['created']}, Skipped: {stats['skipped']}")
            
        except Exception as e:
            print(f"Error: {e}")
            conn.rollback()
            import traceback
            traceback.print_exc()
        finally:
            conn.close()
    
    def _fix_nulls_simple(self):
        """Fallback: set NULL aircraft_type to 'UNKNOWN'."""
        conn = self.connect()
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) as count FROM flight WHERE aircraft_type IS NULL")
            nulls = cursor.fetchone()['count']
            
            if nulls > 0:
                cursor.execute("UPDATE flight SET aircraft_type = 'UNKNOWN' WHERE aircraft_type IS NULL")
                conn.commit()
                print(f"✓ Fixed {nulls} NULL aircraft types (simple mode)")
            else:
                print("✓ No NULL aircraft types found")
        conn.close()
    
    def _get_aircraft_properties(self, call_sign, df):
        """Get aircraft properties from Excel for a call_sign."""
        rows = df[df['Call Sign'] == call_sign]
        if rows.empty:
            return None
        row = rows.iloc[0]
        return {
            'weight_class': row.get('Weight Class (ICAO)'),
            'wake_category': row.get('Wake Category (Aerobahn)'),
            'wingspan_ft': row.get('Wingspan (feet)'),
            'wingspan_m': row.get('Wingspan (meters)')
        }
    
    def _find_matching_unknown(self, props, cursor):
        """Find existing UNKNOWN_XXX with matching properties."""
        span_ft = Decimal(str(props['wingspan_ft'])) if pd.notna(props['wingspan_ft']) else None
        span_m = Decimal(str(props['wingspan_m'])) if pd.notna(props['wingspan_m']) else None
        
        cursor.execute("""
            SELECT aircraft_type FROM aircraft_type 
            WHERE aircraft_type LIKE 'UNKNOWN_%%'
              AND weight_class <=> %s AND wake_category <=> %s
              AND wingspan_ft <=> %s AND wingspan_m <=> %s
            LIMIT 1
        """, (props['weight_class'], props['wake_category'], span_ft, span_m))
        
        result = cursor.fetchone()
        return result['aircraft_type'] if result else None
    
    def _get_next_unknown_number(self, cursor):
        """Get next sequential UNKNOWN_XXX number."""
        cursor.execute("""
            SELECT aircraft_type FROM aircraft_type 
            WHERE aircraft_type LIKE 'UNKNOWN_%%'
            ORDER BY aircraft_type DESC LIMIT 1
        """)
        result = cursor.fetchone()
        if result:
            try:
                return int(result['aircraft_type'].split('_')[1]) + 1
            except:
                pass
        return 1
    
    def _create_unknown_record(self, props, unknown_name, cursor):
        """Create new UNKNOWN_XXX aircraft_type record."""
        cursor.execute("""
            INSERT INTO aircraft_type (aircraft_type, weight_class, wake_category, wingspan_ft, wingspan_m)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            unknown_name,
            props['weight_class'] if pd.notna(props['weight_class']) else None,
            props['wake_category'] if pd.notna(props['wake_category']) else None,
            props['wingspan_ft'] if pd.notna(props['wingspan_ft']) else None,
            props['wingspan_m'] if pd.notna(props['wingspan_m']) else None
        ))


def main():
    parser = argparse.ArgumentParser(description='AIplane Database Manager')
    parser.add_argument('command', choices=['test', 'import', 'import-excel', 
                                           'verify', 'fix-nulls'])
    parser.add_argument('--password', help='MySQL password')
    
    args = parser.parse_args()
    
    if args.password:
        DB_CONFIG['password'] = args.password
    
    manager = DatabaseManager()
    
    if args.command == 'test':
        manager.test_connection()
    elif args.command == 'import':
        manager.import_sql()
    elif args.command == 'import-excel':
        manager.import_excel()
    elif args.command == 'verify':
        manager.verify()
    elif args.command == 'fix-nulls':
        manager.fix_nulls()

if __name__ == "__main__":
    main()
