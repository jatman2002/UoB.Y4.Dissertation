import pandas as pd
import pyodbc

def setup_connection(server_name, db_name):
    conn = pyodbc.connect(
        'Driver={SQL Server};'
        f'Server={server_name};'
        f'Database={db_name};'
        'Trusted_Connection=yes;')
    return conn

def write_to_csv(dataframe, file_name):
    dataframe.to_csv(file_name, ',', encoding='utf-8', index=False)

def get_all_site_codes(connection):
    query = f'SELECT SiteCode FROM Site'
    df = pd.read_sql_query(query, connection)
    return df['SiteCode'].values.tolist()


def get_bookings_for_a_site(connection, site_code):
    query = f'SELECT BookingCode, GuestCount, BookingDate, BookingTime, Duration, CreatedOn FROM Booking WHERE SiteCode={site_code} ORDER BY BookingDate, CreatedOn'
    df = pd.read_sql_query(query, connection)
    return df

def get_tables_for_a_site(connection, site_code):
    query = f'SELECT SiteCode, TableCode, MinCovers, MaxCovers FROM Tables WHERE SiteCode={site_code}'
    df = pd.read_sql_query(query, connection)
    return df
    
def get_existing_bookings(connection, site_code):
    query = f'SELECT TableCode, BookingTab.BookingCode, GuestCount, BookingDate, BookingTime, Duration FROM BookingTab INNER JOIN Booking ON BookingTab.BookingCode = Booking.BookingCode WHERE SiteCode={site_code} ORDER BY BookingDate, CreatedOn'
    df = pd.read_sql_query(query, connection)
    return df

if __name__ == '__main__':

    server_name = 'DESKTOP-TDNFFB7\SQLEXPRESS2017'
    db_name = 'jatin_booking'
    connection = setup_connection(server_name, db_name)


    site_codes = get_all_site_codes(connection)

    for i in range(0,len(site_codes)):
        site_code = site_codes[i]

        file_path = f'C:/git/Dissertation/SQL-DATA/Restaurant-{i+1}/'

        #----------------------------------------------------------------------------------#
        table_df = get_tables_for_a_site(connection, site_code)
        write_to_csv(table_df, file_path + 'tables.csv')
        #----------------------------------------------------------------------------------#
        existing_df = get_existing_bookings(connection, site_code)
        write_to_csv(existing_df, file_path + 'raw-existing.csv')

        # ON A TABLE THAT DOES NOT EXIST ANYMORE
        valid_table_codes = table_df['TableCode'].unique()
        mask = existing_df['TableCode'].isin(valid_table_codes)
        # removed_booking_codes = removed_booking_codes + list(existing_df[~mask]['BookingCode'].unique())
        removed_booking_codes = list(existing_df[~mask]['BookingCode'].unique())
        existing_df = existing_df[mask]

        # # MULTIPLE TABLES FOR ONE BOOKING
        duplicated = existing_df[existing_df.duplicated(subset='BookingCode', keep=False)]
        removed_booking_codes = removed_booking_codes + list(duplicated['BookingCode'].unique())
        existing_df = existing_df.drop(duplicated.index)
        
        write_to_csv(existing_df, file_path + 'existing.csv')

        removed_booking_codes_df = pd.DataFrame({'BookingCode': removed_booking_codes})
        write_to_csv(removed_booking_codes_df, file_path + 'ignore.csv')
        #----------------------------------------------------------------------------------#

        booking_df = get_bookings_for_a_site(connection, site_code)
        write_to_csv(booking_df, file_path + 'raw-reservations.csv')
        mask = booking_df['BookingCode'].isin(removed_booking_codes_df['BookingCode'])
        booking_df = booking_df[~mask]
        write_to_csv(booking_df, file_path + 'reservations.csv')
        #----------------------------------------------------------------------------------#
    
