from datasets import load_dataset

# Import spam dataset
dataset = load_dataset("ucirvine/sms_spam", split="train")

# Display dataset information
print(dataset)

# Display the first message
print(dataset[0])