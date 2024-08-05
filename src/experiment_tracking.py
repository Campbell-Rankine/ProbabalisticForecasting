"""
Class to track and build experiment data over the training run. Use python coroutine architecture to build experiment data
"""

from datetime import datetime
import json

save_path = "./experiment.json"


def experiment_data():
    # output object
    try:
        output = None

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
            if type(user_io) == str and user_io == "STOP_CODE":
                output = data

            if type(user_io) == str and user_io == "SAVE":
                with open(save_path, "wb") as file:
                    json.dump(data, file)

            assert type(user_io) == dict and len(user_io.keys()) == 1

            key = list(user_io.keys())[0]  # match key to key in data
            assert key in data.keys()

            if type(user_io[key]) == list:
                data[key].extend(user_io[key])  # if list then insert into data[data]

            elif type(user_io[key]) == float:
                data[key] = user_io[key]

            else:
                raise ValueError("Unable to find correct type")
    except:
        print("shutdown")
        output = None
