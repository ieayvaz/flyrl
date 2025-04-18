from dronekit import connect
import time
import csv

connection_string = '127.0.0.1:5560'
vehicle = connect(connection_string, wait_ready=True)

with open('flight_data3.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Time', 'Lat', 'Lon', 'Alt', 'Heading', 'Roll', 'Pitch', 'Yaw'])

    while vehicle.armed:
        writer.writerow([
            time.time(),
            vehicle.location.global_frame.lat,
            vehicle.location.global_frame.lon,
            vehicle.location.global_frame.alt,
            vehicle.heading,
            vehicle.attitude.roll,
            vehicle.attitude.pitch,
            vehicle.attitude.yaw
        ])
        time.sleep(0.1)

vehicle.close()