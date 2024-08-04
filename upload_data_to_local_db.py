import mysql.connector as mysql
from datetime import datetime
import json
import os
from tqdm import tqdm
from pathlib import Path
import traceback

def connect_to_db():
    db = mysql.connect(
        host = "localhost",
        port = '3306',
        user = "local",
        password = "local",
        database = "mail_data"
    )
    
    return db


def create_mails_table(db):
    cursor = db.cursor()
    command = """
    CREATE TABLE IF NOT EXISTS mails(
        id INT PRIMARY KEY,
        message_id VARCHAR(255),
        label_ids VARCHAR(255),
        date DATETIME,
        sender VARCHAR(255),
        subject VARCHAR(511),
        body TEXT,
        category VARCHAR(255),
        action VARCHAR(255)
        )
        """
    cursor.execute(command)
    db.commit()
    cursor.close()
    

def insert_mail(db, entries: dict):
    
    cursor = db.cursor()
    command = """INSERT INTO mails (id, message_id, label_ids, date, sender, subject, body, category, action)"""
    command += "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) ON DUPLICATE KEY UPDATE id=id"
    values = (int(entries['queue_id']), 
              entries['message_id'], 
              entries['label_ids'], 
              entries['date'], 
              entries['from'], 
              entries['subject'], 
              entries['body'], 
              entries['category'], 
              entries['gist'])
    cursor.execute(command, values)
    db.commit()
    cursor.close()


def get_datetime(date_str):
    if date_str.strip() == "":
        return None
    date_str = date_str.split(" (")[0]
    date_str = date_str.replace("GMT", "+0000")
    datetime_patterns = ["%a, %d %b %Y %H:%M:%S %z",
                         "%d %b %Y %H:%M:%S %z",
                         "%a, %d %b %Y %H:%M:%S",
                         "%d-%b-%Y %H:%M:%S"]
    
    for datetime_pattern in datetime_patterns:
        try:
            return datetime.strptime(date_str, datetime_pattern).strftime("%Y-%m-%d %H:%M:%S")
        except:
            continue
    else:
        raise ValueError(f"Date string {date_str} does not match any of the patterns {datetime_patterns}")


def insert_emails(db, run_plan_filepath, processed_mail_metadata_folder):
    """
    Insert email data into local database
    
    :param db: mysql.connector.connection.MySQLConnection - connection to the local database
    :param run_plan_filepath: str - path to the run plan file
    :param processed_mail_metadata_folder: str - path to the folder containing processed mail metadata
    
    """
    
    with open(run_plan_filepath, 'r') as f:
        run_plan = json.load(f)
        
    for mail in tqdm(run_plan):
        try:
            mail['label_ids'] = "#".join(mail['label_ids'])
            mail['date'] = get_datetime(mail['date'])
            mail_metadata_filepath = f"{processed_mail_metadata_folder}/{mail['queue_id']}.json"
            with open(mail_metadata_filepath, 'r') as f:
                mail_metadata = json.load(f)
            mail = {**mail, **mail_metadata}
                
            insert_mail(db, mail)
        except Exception as e:
            print(f"Error inserting mail {mail['queue_id']}: {e}")
            traceback.print_exc()
            print(mail['date'])
            raise e
        
        
if __name__ == "__main__":
    
    db = connect_to_db()
    create_mails_table(db)
    run_plan_filepath = Path("mount/mail_processing_logs/run_plan.json")
    processed_mail_metadata_folder = Path("mount/processed_mail_metadata")
    insert_emails(db, run_plan_filepath, processed_mail_metadata_folder)