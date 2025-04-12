from input_parsing import *
from Classes.RestaurantState import RestaurantState
from Reallocation import run

restaurant = RestaurantState(36000, 64, 15, 6)

r_name = 5

parse_tables(restaurant, f'C:/git/UoB.Y4.Dissertation/src/SQL-DATA/Restaurant-{r_name}-tables.csv')

all_requests = parse_reservations(restaurant, f'C:/git/UoB.Y4.Dissertation/src/SQL-DATA/Restaurant-{r_name}-test.csv')

for day in all_requests:

    print(f'STARTING DAY {all_requests.index(day)} / {len(all_requests)}', end='\r')

    r = restaurant.deepcopy()

    r.reservation_requests = day

    r, rejected = run(r)

    output_schedule(r, day[0].booking_date, rejected, r_name)