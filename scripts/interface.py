from scripts.automation import train_model, classify_video
from IPython.display import clear_output

supported_tests = ["straight_walk", "heel_to_toe_walk", "raise_hands", "finger_to_nose"]

while True:
  while True:
    chosen_operation = input("Choose train/classify/exit: ")
    if chosen_operation in ["train", "predict", "exit"]:
      clear_output(wait=True)
      break

  if chosen_operation == "train":
    while True:
      chosen_test = input("Enter requested model to train: ")
      if chosen_test not in supported_tests:
        print(f"Error: {chosen_test} test not supported.")
      else:
        break
    train_model(chosen_test)

  if chosen_operation == "classify":
    while True:
      chosen_test = input("Enter requested test to classify: ")
      if chosen_test not in supported_tests:
        print(f"Error: {chosen_test} test not supported.")
      else:
        break
    filename = input("Enter name of video file: ")
    try:
      classify_video(chosen_test, filename)
    except Exception as e:
      print(e)
      continue

  if chosen_operation == "exit":
    break
