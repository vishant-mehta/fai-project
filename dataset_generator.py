import os
import csv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
import json
import pandas as pd

os.environ['OPENAI_API_KEY'] = "APIKEY" #Insert your api key

def load_and_extract_questions_answers(csv_path, limit=100):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_path)
    
    # Extract the 'Question' and 'Answer' columns, limiting the number of rows
    questions = df['Question'].head(limit).tolist()
    answers = df['Answer'].head(limit).tolist()
    
    # Pair each question with its answer
    qa_pairs = list(zip(questions, answers))
    
    return qa_pairs

def generate_q_answers(charecteristics):
    final_qa_pairs = []
    for charecteristic in charecteristics.values():
        response = generate_qa_pairs(charecteristic["Question"], charecteristic["Context"])
        final_qa_pairs.extend(response)
        # print("Final QA pairs", final_qa_pairs)
    return final_qa_pairs

def generate_qa_pairs(question, context):

    prompt_template = PromptTemplate(
        input_variables=['question', 'context'],
        template="""
        Generate answers for the given question :
        {question}

        Consider two students: Student A and Student B.

    Student A is a genius and always gives the correct answer.
    Student B is a bit lazy and gives a partial answer, which might be close to correct or sometimes almost incorrect ( have a uniform distirbution of the answers).

    Based on the context {context}, generate two answers (atleast 5-10 words)for each question. Evaluate these answers to generate a score. Score can be between 0 - 10. 
    For eg. saying Saturn is the largest planet is wrong (even if it has some words)
    Also, as a teacher, provide feedback for each answer, highlighting what was missing or how it could be improved. The feedback should be about 10-20 words.

    Format the response as follows: 

    Question: [Question text]
    Answer 1: [Answer from Student A]
    Score 1: [Score for Answer 1]
    Feedback 1: [Feedback for Answer 1]
    Answer 2: [Answer from Student B]
    Score 2: [Score for Answer 2]
    Feedback 2: [Feedback for Answer 2]

    In this format, each piece of information is on a new line, and the type of information (Answer, Score, Feedback) is always at the start of the line, followed by a colon and space, then the actual content

        """
    )

    llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.7, max_tokens=1000)
    qa_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)
    response = []
    while not response:
        response = qa_chain.run(question=question, context=context)
    # print("Response", response)
    parsed_response = parse_response(response, question, context)
    # print( "Parsed respone", parsed_response)
    return parsed_response

def parse_response(response, question, context):
    qa_entry = {"question": question, "context": context}
    lines = response.split('\n')

    for line in lines:
        if line.startswith('Answer 1:'):
            qa_entry['answer_1'] = line.split(':', 1)[1].strip()
        elif line.startswith('Score 1:'):
            qa_entry['score_1'] = line.split(':', 1)[1].strip()
        elif line.startswith('Feedback 1:'):
            qa_entry['feedback_1'] = line.split(':', 1)[1].strip()
        elif line.startswith('Answer 2:'):
            qa_entry['answer_2'] = line.split(':', 1)[1].strip()
        elif line.startswith('Score 2:'):
            qa_entry['score_2'] = line.split(':', 1)[1].strip()
        elif line.startswith('Feedback 2:'):
            qa_entry['feedback_2'] = line.split(':', 1)[1].strip()
    return [qa_entry]

def write_to_csv(qa_list, filename='test_qa_dataset.csv'):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Question', 'Context', 'Answer 1', 'Score 1', 'Feedback 1', 'Answer 2', 'Score 2', 'Feedback 2'])
        
        for qa in qa_list:
            writer.writerow([
                qa.get('question', ''),
                qa.get('context', ''),
                qa.get('answer_1', ''),
                qa.get('score_1', ''),
                qa.get('feedback_1', ''),
                qa.get('answer_2', ''),
                qa.get('score_2', ''),
                qa.get('feedback_2', '')
            ])

def main():
    # characteristics = "subjects: science, mathematics; level: high school"
    characteristics = {}
    csv_path = 'test_output.csv'
    qa_pairs = load_and_extract_questions_answers(csv_path)

    for count, (question, answer) in enumerate(qa_pairs):
        characteristic = {
            "Question": question,
            "Context": answer,  # Using the answer as context
            "Requirement": "Descriptive answers with at least 15 word answer"
        }
        characteristics[count] = characteristic
    qa_list = generate_q_answers(characteristics)
    write_to_csv(qa_list)

if __name__ == '__main__':
    main()
