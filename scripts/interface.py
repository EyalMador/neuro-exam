from automation import train_model, predict_result
while True:
  while True:
    chosen_operation = input("Choose train/predict/exit: ")
    if chosen_operation in ["train", "predict", "exit"]:
      break

  if chosen_operation == "train":
    while True:
      chosen_model = input("Enter requested model to train: ")
      if chosen_model not in supported_models:
        print(f"Error: {chosen_model} model not supported.")
      else:
        break
    train_model(chosen_model)

  if chosen_operation == "predict":
    while True:
      chosen_model = input("Enter requested model to predict: ")
      if chosen_model not in supported_models:
        print(f"Error: {chosen_model} model not supported.")
      else:
        break
    filename = input("Enter name of video file: ")
    try:
      predict_result(chosen_model, filename)
    except Exception as e:
      print(e)
      continue

  if chosen_operation == "exit":
    break
