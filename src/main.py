from input_parsing import *
from Classes.RestaurantState import RestaurantState
from Algorithms.Reallocation import run

restaurant = RestaurantState(36000, 64, 15, 6)

parse_tables(restaurant, 'C:/git/UoB.Y4.Dissertation/src/Restaurant-1/tables.csv')

restaurant.reservation_requests = parse_reservations(restaurant, 'C:/git/UoB.Y4.Dissertation/src/Restaurant-1/reservations.csv')

run(restaurant)

output_schedule(restaurant)