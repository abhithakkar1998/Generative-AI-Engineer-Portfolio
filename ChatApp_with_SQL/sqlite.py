import sqlite3

connection = sqlite3.connect('/home/abhi/AI_Workspace/personal/Generative-AI-Engineer-Portfolio/ChatApp_with_SQL/student.db')

cursor = connection.cursor()

table_info = """
CREATE TABLE STUDENT(
        ID INT PRIMARY KEY NOT NULL,
        NAME VARCHAR(50),
        CLASS VARCHAR(20),
        MARKS INT
    )
"""

cursor.execute(table_info)

cursor.executemany("INSERT INTO STUDENT (ID, NAME, CLASS, MARKS) VALUES (?, ?, ?, ?)", [
    (1, 'Alice', '10th Grade', 85),
    (2, 'Bob', '10th Grade', 90),
    (3, 'Charlie', '10th Grade', 78),
    (4, 'David', '10th Grade', 92),
    (5, 'Eva', '10th Grade', 88),
    (6, 'Frank', '10th Grade', 75),
    (7, 'Grace', '10th Grade', 95),
    (8, 'Hannah', '10th Grade', 80),
    (9, 'Ian', '10th Grade', 89),
    (10, 'Jack', '10th Grade', 84)
])

print("Table created and data inserted successfully.")
print("-------------------------------")
print("Inserted Records:")
data=cursor.execute("SELECT * FROM STUDENT")
for i, row in enumerate(data):
    print(f"Row {i+1}: {row}")

connection.commit() 
connection.close()