import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import urllib

def setup_connection(server_name, db_name):
    connection_string = (
        f"DRIVER={{SQL Server}};"
        f"SERVER={server_name};"
        f"DATABASE={db_name};"
        "Trusted_Connection=yes;"
    )
    
    # Encode the connection string for SQLAlchemy
    encoded_conn_str = urllib.parse.quote_plus(connection_string)
    
    # Create SQLAlchemy engine
    engine = create_engine(f"mssql+pyodbc:///?odbc_connect={encoded_conn_str}")
    
    return engine


def write_to_csv(dataframe, file_name):
    dataframe.to_csv(file_name, sep=',', encoding='utf-8', index=False)

def get_all_site_codes(connection):
    query = f'SELECT SiteCode FROM Site'
    df = pd.read_sql_query(query, connection)
    return df['SiteCode'].values.tolist()


# def get_bookings_for_a_site(connection, site_code):
#     query = f'SELECT BookingCode, GuestCount, BookingDate, BookingTime, Duration, CreatedOn FROM Booking WHERE SiteCode={site_code} ORDER BY BookingDate, CreatedOn'
#     df = pd.read_sql_query(query, connection)
#     return df

def get_tables_for_a_site(connection, site_code):
    query = f'SELECT TableCode, MinCovers, MaxCovers FROM Tables WHERE SiteCode={site_code}'
    df = pd.read_sql_query(query, connection)
    return df
    
# def get_existing_bookings(connection, site_code):
#     query = f'SELECT TableCode, BookingTab.BookingCode, GuestCount, BookingDate, BookingTime, Duration FROM BookingTab INNER JOIN Booking ON BookingTab.BookingCode = Booking.BookingCode WHERE SiteCode={site_code} ORDER BY BookingDate, CreatedOn'
#     df = pd.read_sql_query(query, connection)
#     return df

def get_bookings(connection, site_code):
    cols = 'b.BookingCode, GuestCount, BookingDate, BookingStartTime, BookingEndTime, Duration, b.CreatedOn, t.TableCode, t.MinCovers, t.MaxCovers'
    query = f'SELECT {cols} '
    query += f'FROM Booking b '
    query += f'INNER JOIN BookingTab bt '
    query += f'ON b.BookingCode = bt.BookingCode '
    query += f'LEFT JOIN Tables t '
    query += f'ON bt.TableCode = t.TableCode '
    query += f'WHERE b.SiteCode={site_code} '
    query += f'ORDER BY BookingDate'

    df = pd.read_sql_query(query, connection)
    return df


if __name__ == '__main__':

    server_name = 'DESKTOP-TDNFFB7\SQLEXPRESS2017'
    db_name = 'jatin_favouritetable_booking'
    connection = setup_connection(server_name, db_name)


    site_codes = get_all_site_codes(connection)
    print(site_codes)

    for i in range(len(site_codes)):
        site_code = site_codes[i]

        file_path = f'C:/git/UOB.Y4.Dissertation/src/SQL-DATA/Restaurant-{i+1}-'

        #----------------------------------------------------------------------------------#
        table_df = get_tables_for_a_site(connection, site_code)
        write_to_csv(table_df, file_path + 'tables.csv')
        #----------------------------------------------------------------------------------#

        bookings = get_bookings(connection, site_code)

        # masks
        null_table_mask = bookings['TableCode'].isnull()
        duplicate_table_mask = bookings['BookingCode'].duplicated(keep=False)

        # update guest count where multiple tables for one booking
        bookings.loc[duplicate_table_mask & ~null_table_mask, 'GuestCount'] = \
            np.ceil((bookings.loc[duplicate_table_mask & ~null_table_mask,'MaxCovers'] + bookings.loc[duplicate_table_mask & ~null_table_mask,'MinCovers'])/2)

        # remove where table doesn't exist anymore
        bookings = bookings[~null_table_mask]

        bookings = bookings.drop(columns=['MinCovers', 'MaxCovers'])

        write_to_csv(bookings, f'{file_path}bookings.csv')