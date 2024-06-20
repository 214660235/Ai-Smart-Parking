
import pymysql.cursors
from PIL import Image
import pymysql.cursors
import pymysql
from cropbypoint import cut_and_display
import cropbypoint
import numpy as np



def extract_locations():
    try:
        connection = pymysql.connect(
            host="localhost",
            user="root",
            password="mysql24",
            database="parkinglot",
            cursorclass=pymysql.cursors.DictCursor
        )


        with connection.cursor() as cursor:
            cursor.execute("SELECT Location1, Location2, Location3, Location4, Location5, Location6, Location7, Location8 FROM PARKINGLOT3")
            rows = cursor.fetchall()

            # Extract Location1, Location2, Location3, and Location4 from each row
            locations = []
            for row in rows:
                points = [(row[f"Location{i}"], row[f"Location{i+1}"]) for i in range(1, 8, 2)]
                locations.append(points)

            return locations

    except pymysql.Error as error:
        print("Error while connecting to MySQL:", error)

    finally:
        if 'connection' in locals() and connection.open:
            connection.close()
            print("MySQL connection is closed")

locations = extract_locations()
# print(locations)
locations = np.array(locations)
image_path1 = r"C:\Users\User\Downloads\photo-1506521781263-d8422e82f27a-1920w.webp"
# image_path1 = r"C:\Users\326022910\Downloads\john-matychuk-yvfp5YHWGsc-unsplash.jpg"
print(locations)
print(locations.shape)
locations.reshape((-1,8))
print(locations.shape)

image_path1=r"C:\Users\User\Downloads\photo-1506521781263-d8422e82f27a-1920w.webp"
cut_and_display(image_path1,locations)
# print(locations)


def crop_image_from_mysql_and_cut(lot_number):
    try:
        connection = pymysql.connect(
            host="localhost",
            user="root",
            password="mysql24",
            database="parkinglot",
            cursorclass=pymysql.cursors.DictCursor
        )

        with connection.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS PARKINGLOT3 (
                    lot_number INT,
                    Location1 INT,
                    Location2 INT,
                    Location3 INT,
                    Location4 INT,
                    Location5 INT,
                    Location6 INT,
                    Location7 INT,
                    Location8 INT,
                    occupancy_status VARCHAR(50),
                    image_path  VARCHAR(255),
                )
            """)

            connection.commit()

            cursor.execute("SELECT Location1, Location2, Location3, Location4, Location5, Location6, Location7, Location8 FROM PARKINGLOT3 WHERE lot_number = %s", (lot_number,))
            locations = cursor.fetchone()

            if locations is None:
                print("No data found for the given lot number.")
                return None

            location1, location2, location3, location4, location5, location6, location7, location8 = map(int, (locations["Location1"], locations["Location2"], locations["Location3"], locations["Location4"], locations["Location5"], locations["Location6"], locations["Location7"], locations["Location8"]))

            return location1, location2, location3, location4, location5, location6, location7, location8

    except pymysql.Error as error:
        print("Error while connecting to MySQL:", error)

    finally:
        if 'connection' in locals() and connection.open:
            connection.close()
            print("MySQL connection is closed")




def inserttosql(lot_number, drawn_lines, occupancy_status, image_path):
    try:
        connection = pymysql.connect(
            host="localhost",
            user="root",
            password="mysql24",
            database="parkinglot",
            cursorclass=pymysql.cursors.DictCursor
        )

        with connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO PARKINGLOT3 (lot_number, location1, location2, location3, location4, location5, location6, location7, Location8, occupancy_status, image_path)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (lot_number, drawn_lines[0][0], drawn_lines[0][1], drawn_lines[1][0], drawn_lines[1][1], drawn_lines[2][0], drawn_lines[2][1], drawn_lines[3][0], drawn_lines[3][1], occupancy_status, image_path))

            connection.commit()

    except pymysql.Error as error:
        print("Error while connecting to MySQL:", error)

    finally:
        if 'connection' in locals() and connection.open:
            connection.close()
            print("MySQL connection is closed")


