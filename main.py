import csv
import os.path
import random
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the pre-trained transformer model
model = AutoModelForCausalLM.from_pretrained("microsoftDialoGPT-large")
tokenizer = AutoTokenizer.from_pretrained("microsoftDialoGPT-large", padding_side="right")


# Define a function to generate a response from the model
def generate_response(text):
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors='pt')

    # generated a response while limiting the total chat history to 1000 tokens,
    response = model.generate(new_user_input_ids, max_length=1000, num_return_sequences=1,
                              pad_token_id=tokenizer.eos_token_id)

    response = response.tolist()
    answer = response[0][response[0].index(tokenizer.eos_token_id) + 1:]
    result = tokenizer.decode(answer, skip_special_tokens=True)

    return result


# Define a function to add new data to the training dataset
def fine_tune_model(data):
    encoded_training_dataset = []

    for question, answer in data.items():
        encoded_question = tokenizer.encode(question + tokenizer.eos_token, return_tensors='pt')
        encoded_answer = tokenizer.encode(answer + tokenizer.eos_token, return_tensors='pt')
        encoded_training_dataset.append((encoded_question, encoded_answer))

    # Add the new data to the training dataset
    model.train_dataset = encoded_training_dataset
    model.train()
    print("Model is fine-tuned with stored questions and answers")


# Save the dictionary as a CSV file
def save_dictionary_to_csv(file_path, data):
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Question', 'Answer'])
        for question, answer in data.items():
            writer.writerow([question, answer])


# Load the dictionary from a CSV file
def load_dictionary_from_csv(file_path):
    loaded_data = {}
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            question, answer = row
            loaded_data[question] = answer
    return loaded_data


# Start a conversation with the chatbot
def main():
    if os.path.exists("main.csv"):
        questions = load_dictionary_from_csv("main.csv")
    else:
        questions = {}

    step = 0
    while True:
        step += 1
        # Get user input
        text = input(">> You(you can enter 'q' to quit): ")

        if text == "q":
            break

        # Check if the question is already in the dictionary
        if text in questions:
            print("Answering from local dictionary!")
            # Get the answer from the dictionary
            answer = questions[text]

            # Print the answer
            print(">> Chatbot: " + answer)
        else:
            # The question is not in the dictionary
            # Generate a response from the model
            print("Answering from GPT!")
            response = generate_response(text)

            # Print the response
            print(">> Chatbot: " + response)

            # Add the question and answer to the dictionary
            questions[text] = response

        # Check if the user wants to add new data to the training dataset
        if step % 5 == 0:
            if input("Do you want to add new data to the training dataset? (Y/N) ") in ["Y", "y"]:
                # Add the new data to the training dataset
                fine_tune_model(questions)

    save_dictionary_to_csv("main.csv", questions)


if __name__ == "__main__":
    main()
