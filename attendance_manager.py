import pandas as pd
from datetime import datetime
from config import ATTENDANCE_FILE

class AttendanceManager:
    def __init__(self):
        """Initialize attendance file if it does not exist."""
        if not pd.io.common.file_exists(ATTENDANCE_FILE):
            df = pd.DataFrame(columns=["Name", "Date", "In-Time", "Out-Time"])
            df.to_csv(ATTENDANCE_FILE, index=False)

    def mark_attendance(self, name):
        """Mark attendance for a recognized person."""
        df = pd.read_csv(ATTENDANCE_FILE)
        today = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M:%S")

        if not ((df["Name"] == name) & (df["Date"] == today)).any():
            new_entry = pd.DataFrame([[name, today, current_time, ""]], columns=df.columns)
            df = pd.concat([df, new_entry], ignore_index=True)
        else:
            df.loc[(df["Name"] == name) & (df["Date"] == today), "Out-Time"] = current_time

        df.to_csv(ATTENDANCE_FILE, index=False)
