import requests
import concurrent.futures
import time
import json
import csv
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
    """Function to make the POST request with a random model."""
    # Randomly select the model
    start_time = time.time()
    model = random.choice(["domain_a", "domain_b"])
    # Create the request data with the randomly selected model
    request_data = {**base_data, "model": model}

    response = requests.post(url, headers=headers, data=json.dumps(request_data))
    end_time = time.time()
    elp_time = round(end_time - start_time, 2)
    return response.status_code, response.json(), elp_time


def main():
    num_requests = 2
    timings = []

    # Start the timer
    start_time = time.time()

    # Use ThreadPoolExecutor to make concurrent requests
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_request = {executor.submit(make_request): i for i in range(num_requests)}

        for future in concurrent.futures.as_completed(future_to_request):
            try:
                status_code, response_data, elp_time = future.result()
                first_text = response_data.get('generated_texts', [''])[0]
                print(f"Response Code: {status_code}, Response: {first_text.split('Output:')[-1]}")
                timings.append(elp_time)

                if len(timings) % 250 == 0:
                    filename = f"timings_{num_requests}.csv"
                    with open(filename, mode='w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(timings)  # Write the list as a single row
                    print(f"Data saved to {filename}")

            except Exception as e:
                print(f"Request generated an exception: {e}")

    # End the timer
    end_time = time.time()
    total_time = round(end_time - start_time, 2)
    average_time = total_time / num_requests

    print(f"\nTotal Time for {num_requests} requests: {total_time:.2f} seconds")
    print(f"Average Time per request: {average_time:.2f} seconds")
    print(timings)



if __name__ == "__main__":
    main()
