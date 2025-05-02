from datasets import Dataset



print("Loading train dataset")
dataset = Dataset.from_file("logs_and_outputs/order_1/17925c6a81583bc3da5a4b7d196aba0a/uie_instructions/default-8a6abe4434e90f8a/2.0.0/c490e7f13dec80785fc335819009163a45c86ae2816040c8d81800108e7e4374/uie_instructions-train.arrow")

# 查看前几条
print(dataset[0])
print(dataset[1])


print("Loading test dataset")
dataset = Dataset.from_file("logs_and_outputs/order_1/17925c6a81583bc3da5a4b7d196aba0a/uie_instructions/default-8a6abe4434e90f8a/2.0.0/c490e7f13dec80785fc335819009163a45c86ae2816040c8d81800108e7e4374/uie_instructions-test.arrow")

# 查看前几条
print(dataset[0])
print(dataset[1])