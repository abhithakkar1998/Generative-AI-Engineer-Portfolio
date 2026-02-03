# Chat with SQL Database - Project Setup Guide

This project allows you to chat with SQL databases (SQLite or MySQL) using LangChain and Azure OpenAI.

## Prerequisites

- Python 3.10 or higher
- Azure OpenAI API credentials
- Virtual environment (recommended)

## Installation Steps

### 1. Install Python Dependencies

```bash
pip install streamlit langchain langchain-community langchain-openai sqlalchemy python-dotenv
```

### 2. Install MySQL Connector (Required for MySQL support)

```bash
pip install mysql-connector-python
```

## Database Setup

You have two options: **SQLite** (local, no installation required) or **MySQL** (requires server installation).

---

## Option 1: SQLite Database Setup (Recommended for Quick Start)

### Step 1: Create the SQLite Database

Run the `sqlite.py` script to create the local database and populate it with sample data:

```bash
python sqlite.py
```

This will:
- Create a `student.db` file in the current directory
- Create a `STUDENT` table with the following schema:
  - `ID` (INT PRIMARY KEY)
  - `NAME` (VARCHAR)
  - `CLASS` (VARCHAR)
  - `MARKS` (INT)
- Insert 10 sample student records

### Step 2: Verify Database Creation

You should see output similar to:
```
Table created and data inserted successfully.
-------------------------------
Inserted Records:
Row 1: (1, 'Alice', '10th Grade', 85)
Row 2: (2, 'Bob', '10th Grade', 90)
...
```

---

## Option 2: MySQL Database Setup

### Step 1: Install MySQL Server

**On Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install mysql-server
```

**On macOS (using Homebrew):**
```bash
brew install mysql
```

**On Windows:**
Download and install from [MySQL Official Website](https://dev.mysql.com/downloads/installer/)

### Step 2: Start MySQL Service

**On Ubuntu/Debian:**
```bash
sudo systemctl start mysql
sudo systemctl enable mysql
```

**On macOS:**
```bash
brew services start mysql
```

### Step 3: Secure MySQL Installation

```bash
sudo mysql_secure_installation
```

Follow the prompts to:
- Set root password
- Remove anonymous users
- Disallow root login remotely
- Remove test database

### Step 4: Create Database and Table

1. Log into MySQL:
```bash
sudo mysql -u root -p
```

2. Create the database:
```sql
CREATE DATABASE student;
```

3. Create a user and grant privileges (optional, for security):
```sql
CREATE USER 'your_username'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON student.* TO 'your_username'@'localhost';
FLUSH PRIVILEGES;
```

4. Use the database:
```sql
USE student;
```

5. Create the STUDENT table:
```sql
CREATE TABLE STUDENT (
    ID INT PRIMARY KEY NOT NULL,
    NAME VARCHAR(50),
    CLASS VARCHAR(20),
    MARKS INT
);
```

6. Insert sample data:
```sql
INSERT INTO STUDENT (ID, NAME, CLASS, MARKS) VALUES
    (1, 'Alice', '10th Grade', 85),
    (2, 'Bob', '10th Grade', 90),
    (3, 'Charlie', '10th Grade', 78),
    (4, 'David', '10th Grade', 92),
    (5, 'Eva', '10th Grade', 88),
    (6, 'Frank', '10th Grade', 75),
    (7, 'Grace', '10th Grade', 95),
    (8, 'Hannah', '10th Grade', 80),
    (9, 'Ian', '10th Grade', 89),
    (10, 'Jack', '10th Grade', 84);
```

7. Verify the data:
```sql
SELECT * FROM STUDENT;
```

8. Exit MySQL:
```sql
EXIT;
```

---

## Azure OpenAI Configuration

### Step 1: Create `.env` File

Create a `.env` file in your project root directory with the following variables:

```env
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_LLM_MODEL=your-deployment-name
```

Replace the values with your actual Azure OpenAI credentials.

### Step 2: Update the Path in `app.py`

If your `.env` file is not in the project root, update the path in `app.py`:

```python
load_dotenv("/path/to/your/.env")
```

---

## Running the Application

### Step 1: Start the Streamlit App

```bash
streamlit run app.py
```

### Step 2: Configure Database Connection

**For SQLite:**
1. In the sidebar, select **"Use SQLite Local DB - student.db"**
2. The app will automatically connect to your local `student.db` file

**For MySQL:**
1. In the sidebar, select **"Connect to your SQL Database (MySQL)"**
2. Enter your MySQL credentials:
   - **MySQL User**: `root` (or your custom user)
   - **MySQL Password**: Your MySQL password
   - **MySQL Host**: `localhost`
   - **MySQL Port**: `3306`
   - **MySQL Database Name**: `student`

### Step 3: Start Chatting!

Ask questions like:
- "How many records are there in the database?"
- "Show me all students with marks greater than 85"
- "What is the average marks of all students?"
- "List students from 10th Grade"
- "Who has the highest marks?"

---

## Troubleshooting

### Issue: "Can't connect to MySQL server"
- Ensure MySQL service is running: `sudo systemctl status mysql`
- Verify credentials are correct
- Check if MySQL is listening on port 3306: `sudo netstat -tlnp | grep 3306`

### Issue: "No module named 'mysql'"
- Install the MySQL connector: `pip install mysql-connector-python`

### Issue: "student.db not found"
- Run `python sqlite.py` to create the database first
- Ensure you're running the app from the correct directory

### Issue: Azure OpenAI API errors
- Verify your API key and endpoint are correct in `.env`
- Check your Azure OpenAI deployment name
- Ensure you have sufficient quota

---

## Project Structure

```
ChatApp_with_SQL/
├── app.py                 # Main Streamlit application
├── sqlite.py             # SQLite database creation script
├── student.db            # SQLite database file (created after running sqlite.py)
├── PROJECT_SETUP.md      # This setup guide
└── .env                  # Environment variables (create this)
```

---

## Features

- 🤖 **Natural Language Queries**: Ask questions in plain English
- 🔄 **Multi-Database Support**: Works with SQLite and MySQL
- 💬 **Conversation History**: Maintains chat context
- 🔍 **SQL Query Visualization**: See the generated SQL queries
- 🎯 **Smart Agent**: Uses LangChain agents with SQL tools for accurate responses

---

## Security Notes

- Never commit your `.env` file to version control
- Use strong passwords for MySQL users
- For production, create dedicated MySQL users with limited privileges
- Consider using read-only database connections for query-only applications

---

## Next Steps

After setup, you can:
1. Add more tables to your database
2. Modify the system prompt to customize agent behavior
3. Add custom tools for specific queries
4. Deploy the app to cloud platforms (Streamlit Cloud, Heroku, etc.)

Happy querying! 🎉
