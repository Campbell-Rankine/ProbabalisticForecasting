"""
Experiment Dashboarding Module:
---

Use python coroutine architecture to build a cached (in memory) dashboarding service. Essentially no overhead and Async pushes to the DB
"""

from datetime import datetime
import json

save_path = "./experiment.json"


def experiment_data():
    # output object
    try:
        output = None

        flags = ["STOP_CODE", "SAVE"]

        # coroutine internal data storage
        data = {
            "date": datetime.now().strftime("%d/%m/%YYYY"),
            "avg_loss": -1.0,
            "avg_grad_norm": -1.0,
            "avg_iqr": -1.0,
            "total_iters": 0.0,
            "data": [],
        }

        while user_io := (yield output):
            # check input
            assert type(user_io) == dict
            assert len(user_io.keys()) == 1
            assert list(user_io.keys())[0] in data.keys()

            key = list(user_io.keys())[0]  # match key to key in data

            if key in flags:
                match key:
                    case "STOP_CODE":
                        with open(save_path, "w") as file:
                            json.dump(data, file)
                        output = data
                    case "SAVE":
                        with open(save_path, "w") as file:
                            json.dump(data, file)

            if type(user_io[key]) == list:
                data[key].extend(user_io[key])  # if list then insert into data[data]

            elif type(user_io[key]) == float:
                data[key] = user_io[key]

            else:
                raise ValueError("Unable to find correct type")
    except Exception as e:
        print(f"shutdown: {e}")
        with open(save_path, "w") as file:
            json.dump(data, file)
        output = data
