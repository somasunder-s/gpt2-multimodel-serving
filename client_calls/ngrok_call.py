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
    "prompt": [ """Jane Doe  Certified Quality Assurance Engineer  As a fresher, I look forward to explore my knowledge and abilities to be in best interest of organization and society and build a career with a leading corporate to brush up my capabilities.\njane.doe@example.com +10000000000   Springfield, Anytown, Country linkedin.com/in/jane-doe   EDUCATION   B.Tech  Generic Institute of Technology  07/2016 - 07/2025,  8.22   HSC  Generic High School  04/2014 - 04/2015,  76   SSC  Generic High School  04/2012 - 04/2013,  9.0   PROJECTS   OpenCart , e-Commerce Platform  Web based software providing a professional & reliable foundation, user friendly interface to use, to new shop owners."""],
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
                print(f"Response Code: {status_code}, Response: {response_data['generated_text'].split('Output:')[-1]}")
                # print(f"Response Code: {status_code}, Response: {response_data['generated_text']['Output:'][-1]}")
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
