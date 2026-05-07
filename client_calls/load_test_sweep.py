import pandas as pd
import requests
import concurrent.futures
import time
import json
import pandas
import random

# New URL for the POST request
url = "http://localhost:8000/generate/"
headers = {"Content-Type": "application/json"}

# The data payload for the POST request (without the model)
base_data = {
    "prompts": ["Sample input text for load testing the inference endpoint. Replace with whatever shape of input your fine-tuned model expects."],
    "max_length": 512
}


def make_request():
    start_time = time.time()
    model = random.choice(["domain_a", "domain_b"])
    request_data = {**base_data, "model": model}

    response = requests.post(url, headers=headers, data=json.dumps(request_data))
    end_time = time.time()
    elp_time = round(end_time - start_time, 2)

    if response.status_code == 200:
        return response.status_code, response.json(), elp_time

    return response.status_code, "Error", elp_time


def main():
    # instance details
    # cpu, ram, gpu = 8, 54, 23
    instance = 1
    workers, threads = 8, 16

    num_requests = 250
    # max_workers = 12

    for max_workers in range(workers-3, 0, -1):
        print('\n max_workers: ', max_workers)

        start_time = time.time()
        all_data = []
        # Use ThreadPoolExecutor to make concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:

            future_to_request = {executor.submit(make_request): i for i in range(num_requests)}
            for future in concurrent.futures.as_completed(future_to_request):

                status_code, response_data, elp_time = future.result()
                data = {}
                data['status_code'] = status_code
                data['response_data'] = response_data
                data['elp_time'] = elp_time

                all_data.append(data)
                if len(all_data) % 25 == 0:
                    print(len(all_data)/250, end=' - ')

        end_time = time.time()
        total_time = round(end_time - start_time, 2)
        tps = num_requests/total_time

        df = pd.DataFrame(all_data)

        # Additional parameters - Ignore this
        df['tps'] = [tps]*len(df)
        df['total_time'] = [total_time] * len(df)

        file_name = f"data/Instance{instance}_w_{workers}_t_{threads}_con_{max_workers}.csv"
        df.to_csv(file_name)

        print('TPS: ', tps)
        print('average_time_per_request: ', sum(df['elp_time']) / len(df['elp_time']))
        print(f"Total Time for {num_requests} requests: {total_time:.2f} seconds")

        print('Break for 1 Min!')
        time.sleep(60)


if __name__ == "__main__":
    main()
