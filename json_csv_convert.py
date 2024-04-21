import json
import csv
import os

print(os.getcwd())
# Load the JSON file
with open('test_webmd_squad_v2_full.json', 'r') as file:
    data = json.load(file)

# Open a CSV file for writing
with open('test_output.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    
    # Write the header of the CSV file
    writer.writerow(['URL', 'Question ID', 'Question', 'Answer', 'Answer Start'])

    # Parse the JSON data
    for article in data['data']:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                question = qa['question']
                question_id = qa['id']
                url = qa.get('url', 'No URL provided')
                for answer in qa['answers']:
                    answer_text = answer['text']
                    answer_start = answer['answer_start'] if 'answer_start' in answer else 'No start'

                    # Write the extracted data to the CSV file
                    writer.writerow([url, question_id, question, answer_text, answer_start])
