from response import predict
while True:
    message =  input("User: ")
    res = predict(message)
    print("Bot: {}".format(res))