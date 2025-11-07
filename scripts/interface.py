import ipywidgets as widgets
from IPython.display import display, clear_output
from scripts.automation import train, classify_video, test

supported_tests = ["straight_walk", "heel_to_toe_walk", "raise_hands", "finger_to_nose"]

operation_dropdown = widgets.Dropdown(
    options=["train", "classify", "test"],
    description="Operation:",
)

test_dropdown = widgets.Dropdown(
    options=supported_tests,
    description="Test:",
)

file_input = widgets.Text(
    description="Video file:",
    placeholder="example.mov",
)

run_button = widgets.Button(description="Run", button_style="success")
output = widgets.Output()

def on_run_clicked(b):
    clear_output(wait=False)
    with output:
        chosen_operation = operation_dropdown.value
        chosen_test = test_dropdown.value

        if chosen_operation == "train":
            train(chosen_test)
        elif chosen_operation == "test":
            test(chosen_test)
        elif chosen_operation == "classify":
            classify_video(chosen_test, file_input.value)

run_button.on_click(on_run_clicked)

ui = widgets.VBox([operation_dropdown, test_dropdown, file_input, run_button, output])
display(ui)
