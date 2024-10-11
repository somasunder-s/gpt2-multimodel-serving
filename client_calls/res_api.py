import requests

# Define the API endpoint
url = "http://localhost:8000/generate/"

# prompt_ =
# Define the payload with the prompt, max_length, and model type
payload = {
    "prompt": """Jane Doe  Certified Quality Assurance Engineer  As a fresher, I look forward to explore my knowledge and abilities to be in best interest of organization and society and build a career with a leading corporate to brush up my capabilities.
jane.doe@example.com +10000000000   Springfield, Anytown, Country linkedin.com/in/jane-doe   EDUCATION   B.Tech  Generic Institute of Technology  07/2016 - 07/2025,  8.22   HSC  Generic High School  04/2014 - 04/2015,  76   SSC  Generic High School  04/2012 - 04/2013,  9.0   PROJECTS   OpenCart , e-Commerce Platform  Web based software providing a professional & reliable foundation, user friendly interface to use, to new shop owners.
""",
    "max_length": 512,
    "model": "domain_a"  # or "domain_b" for the other model
}

# Make the POST request
response = requests.post(url, json=payload)

# Check the response
if response.status_code == 200:
    # Parse the JSON response
    generated_text = response.json().get("generated_text")
    print("Generated Text:", generated_text)
else:
    print(f"Error: {response.status_code}, {response.text}")
