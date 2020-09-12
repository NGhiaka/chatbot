import pandas as pd
from utils.preprocessing import preprocess_senctence

def load_conversation(file_name):
    questions = list()
    answers = list()
    ftype = file_name[-3:]
    if ftype == 'csv':
        conversations = pd.read_csv(file_name)
        check_nan = conversations.notna()
        for i in range(len(conversations)):
            if check_nan['Question'][i] and check_nan['answer'][i]:
                question = preprocess_senctence(conversations['Question'][i])
                answer = preprocess_senctence(conversations['answer'][i])
                questions.append(question)
                answers.append(answer)
    else:

        with open(file_path, 'rb') as rf:
            line = rf.readline().decode('utf-8')
            while line != '':
                if "*" in line:
                    question = self.preprocess_senctence(line)
                    questions.append(question)
                if "+" in line:
                    answer = self.preprocess_senctence(line)
                    answers.append(answer)
                line = rf.readline().decode('utf-8')
    return questions, answers

    # print(len(df))
    # print(df['Question'].dtypes)
    # print(df['answer'].dtypes)