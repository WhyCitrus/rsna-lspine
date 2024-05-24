import datetime 
import pandas as pd
import re

from collections import defaultdict


def convert_date(d):
    return datetime.datetime.strptime(str(d), "%B %d, %Y")


schedule = pd.read_excel("qgenda_schedule.xlsx", header=None).iloc[3:].reset_index(drop=True) # remove first 3 rows

# remove stuff at the bottom
try:
    last_row = schedule.iloc[:, 0].tolist().index("Schedule Notes:")
except ValueError:
    last_row = schedule.iloc[:, 0].tolist().index("Phone Numbers")


schedule = schedule.iloc[:last_row].reset_index(drop=True)

# remove days of week rows
remove_rows = []
for idx, i in enumerate(schedule.iloc[:, 0].tolist()):
    if i == "Sunday":
        remove_rows.append(idx)

schedule = schedule.drop(remove_rows).reset_index(drop=True)

# split the schedule by weeks
sundays = [], week_rows = []
for idx, i in enumerate(schedule.iloc[:, 0].tolist()):
    try:
        tmp_date = convert_date(i)
        sundays.append(tmp_date)
        week_rows.append(idx)
    except ValueError:
        continue

weekly_schedule = {}
for idx, each_sunday in enumerate(sundays):
    if idx == len(sundays) - 1:
        weekly_schedule[each_sunday] = schedule.iloc[week_rows[idx]:].reset_index(drop=True)
    else:
        weekly_schedule[each_sunday] = schedule.iloc[week_rows[idx]:week_rows[idx+1]].reset_index(drop=True)


for each_week, each_schedule in weekly_schedule.items():
    daily_schedule = {}
    list_by_day = [each_schedule.iloc[:, idx:idx+2] for idx in range(0, each_schedule.shape[1], 2)]
    for each_day in list_by_day:
        tmp_date = convert_date(each_day.iloc[0, 0])
        daily_schedule[tmp_date] = defaultdict(list)
        for resident, task in zip(each_day.iloc[1:, 0].tolist(), each_day.iloc[1:, 1].tolist()):
            if str(resident) == "nan":
                continue
            task = re.sub(r" \([A-Z]+\)", "", task)
            daily_schedule[tmp_date][resident].append(task)
    break



# it's easier to keep track of NF, consult by days rather than weeks due to potential random swaps, sick call, and split weeks
