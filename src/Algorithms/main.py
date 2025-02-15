from input_parsing import *
from Classes.RestaurantState import RestaurantState
from Reallocation import run

restaurant = RestaurantState(36000, 64, 15, 6)

parse_tables(restaurant, 'C:/git/UoB.Y4.Dissertation/src/Restaurant-1/tables.csv')

all_requests = parse_reservations(restaurant, 'C:/git/UoB.Y4.Dissertation/src/Restaurant-1/reservations.csv')
all_requests = all_requests[1082:]

for day in all_requests:

    print(f'STARTING DAY {all_requests.index(day)} / {len(all_requests)}', end='\r')

    r = restaurant.deepcopy()

    r.reservation_requests = day

    r, rejected = run(r)

    output_schedule(r, day[0].booking_date, rejected)